"""Reproducible random test and object-balanced train/validation image splits."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scripts import dataset_pipeline as dp
from scripts import dataset_eda as eda


def split_summary(records):
    rows = []
    labeled_total = sum(r["objects"] is not None for r in records)
    object_total = sum(len(r["objects"]) for r in records if r["objects"] is not None)
    for split in ("train", "val", "test", "fpb_test_unlabeled"):
        selected = [r for r in records if r["split"] == split]
        labeled = [r for r in selected if r["objects"] is not None]
        objects = None if selected and not labeled else sum(len(r["objects"]) for r in labeled)
        rows.append({"split":split,"images":len(selected),"labeled_images":len(labeled),"objects":objects,
                     "labeled_image_share": len(labeled)/max(labeled_total,1) if split != "fpb_test_unlabeled" else None,
                     "object_share": objects/max(object_total,1) if objects is not None else None})
    return pd.DataFrame(rows).set_index("split")


def split_plan(records, n_classes, *, fractions=(0.8, 0.15, 0.05), seed=42, max_swaps=5000):
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Нужны три положительные доли train/valid/test с суммой 1.")
    labeled = sorted((i for i,r in enumerate(records) if r["objects"] is not None),
                     key=lambda i: (records[i]["source"], records[i]["source_id"]))
    n = len(labeled)
    n_test, n_val = round(n*fractions[2]), round(n*fractions[1])
    n_train = n-n_test-n_val
    if min(n_train, n_val) < 1:
        raise ValueError("Недостаточно размеченных изображений для непустых train и valid.")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(labeled)[rng.permutation(n)]
    test, remaining = shuffled[:n_test], shuffled[n_test:]
    image_indices, classes, counts = [], [], []
    for i, original_i in enumerate(remaining):
        for class_id, count in Counter(o["class_id"] for o in records[original_i]["objects"]).items():
            image_indices.append(i); classes.append(class_id); counts.append(count)
    image_indices, classes = np.asarray(image_indices, dtype=int), np.asarray(classes, dtype=int)
    counts = np.asarray(counts, dtype=float)
    matrix = np.zeros((len(remaining), n_classes), dtype=np.float32)
    matrix[image_indices, classes] = counts
    totals = matrix.sum(axis=0, dtype=float)
    target = totals * (n_val / len(remaining))
    weights = 1/np.maximum(totals, 1)
    validation = np.arange(len(remaining)) < n_val
    error = matrix[validation].sum(axis=0, dtype=float)-target
    initial_objective = float(np.dot(error*error, weights))
    norm = np.bincount(image_indices, weights=counts*counts*weights[classes], minlength=len(remaining))
    swaps = 0
    with tqdm(total=max_swaps, desc="Балансировка train/valid", unit="обмен") as progress:
        for _ in range(max_swaps):
            linear = 2*np.bincount(image_indices, weights=counts*(error*weights)[classes], minlength=len(remaining))
            add, remove = norm+linear, norm-linear
            train_ids, val_ids = np.flatnonzero(~validation), np.flatnonzero(validation)
            best = None
            for candidate_count in (32, 128):
                ka, kb = min(candidate_count, len(train_ids)), min(candidate_count, len(val_ids))
                a = train_ids[np.argpartition(add[train_ids], ka-1)[:ka]]
                b = val_ids[np.argpartition(remove[val_ids], kb-1)[:kb]]
                cross = (matrix[a].astype(float)*weights) @ matrix[b].T
                changes = add[a, None]+remove[b][None, :]-2*cross
                x,y = np.unravel_index(np.argmin(changes), changes.shape)
                if changes[x,y] < -1e-10:
                    best = int(a[x]), int(b[y])
                    break
            if best is None:
                break
            a,b = best
            error += matrix[a]-matrix[b]
            validation[a], validation[b] = True, False
            swaps += 1
            progress.update()
    assignments = {int(i): "test" for i in test}
    assignments.update({int(i): "val" if validation[j] else "train" for j,i in enumerate(remaining)})
    val_counts = matrix[validation].sum(axis=0, dtype=float)
    train_counts = totals-val_counts
    val_share = val_counts/max(val_counts.sum(), 1)
    train_share = train_counts/max(train_counts.sum(), 1)
    comparison = pd.DataFrame({"class_id": np.arange(n_classes), "train_objects": train_counts.astype(int),
                               "valid_objects": val_counts.astype(int), "train_share": train_share, "valid_share": val_share,
                               "difference_pp": 100*np.abs(train_share-val_share)})
    return assignments, comparison, {
        "labeled_images": n, "train_images": n_train, "valid_images": n_val, "test_images": n_test,
        "unlabeled_images_unchanged": len(records)-n, "swaps": swaps,
        "objective_before": initial_objective, "objective_after": float(np.dot(error*error, weights)),
        "max_class_difference_pp": float(comparison.difference_pp.max()),
        "mean_class_difference_pp": float(comparison.difference_pp.mean()),
        "classes_missing_train": int(((totals > 0) & (train_counts == 0)).sum()),
        "classes_missing_valid": int(((totals > 0) & (val_counts == 0)).sum()),
    }


def resplit_export(data, output, *, fractions=(0.8, 0.15, 0.05), seed=42, max_swaps=5000):
    """Move labeled files in place, update metadata/lists/YAML and roll back on failure."""
    data, output = Path(data).resolve(), Path(output).resolve()
    if not output.is_relative_to(data / "processed"):
        raise ValueError("Переразбиение разрешено только внутри data/processed/.")
    manifest_path = output / "manifest.jsonl"
    before = manifest_path.read_bytes()
    old_hash = hashlib.sha256(before).hexdigest()
    params = {"fractions": list(fractions), "seed": seed, "method": "object_count_swaps_v1"}
    split_state_path = data / "metadata" / (output.name + ".eda_split.json")
    if split_state_path.exists():
        saved = dp.read_json(split_state_path)
        if saved["parameters"] == params and saved["manifest_sha256"] == old_hash:
            return saved["summary"] | {"status": "REUSED", "files_moved": 0}
    records, _, _, names = eda.load_export(output)
    assignments, _, summary = split_plan(records, len(names), fractions=fractions, seed=seed, max_swaps=max_swaps)
    transformed, operations = [], []
    for i, original in enumerate(records):
        row = dict(original)
        if i in assignments:
            row["split"] = assignments[i]
            for field, folder in (("image", "images"), ("label", "labels")):
                row[field] = f"{folder}/{row['split']}/{Path(original[field]).name}"
                if row[field] != original[field]:
                    source, target = output / original[field], output / row[field]
                    if not source.resolve().is_relative_to(output) or not target.resolve().is_relative_to(output):
                        raise ValueError("Путь файла выходит за пределы сборки.")
                    if not source.is_file() or target.exists():
                        raise ValueError(f"Источник отсутствует или целевой файл уже существует: {source} -> {target}")
                    operations.append((source,target))
        transformed.append(row)
    for field in ("image", "label"):
        paths = [r[field] for r in transformed if r[field] is not None]
        if len(paths) != len(set(paths)):
            raise ValueError("Переразбиение создаёт совпадающие имена файлов.")
    manifest = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in transformed).encode("utf-8")
    new_hash = hashlib.sha256(manifest).hexdigest()
    build_state_path = data / "metadata" / (output.name + ".build.json")
    build_state = dp.read_json(build_state_path)
    if build_state["manifest_sha256"] != old_hash:
        raise ValueError("Manifest отличается от завершённой сборки.")
    build_state["manifest_sha256"] = new_hash
    summary["files_moved"] = len(operations)
    for key,rows in (("before",records),("after",transformed)):
        table = split_summary(rows).reset_index().astype(object)
        summary[key] = table.where(pd.notna(table),None).to_dict("records")
    updates = {manifest_path: manifest, build_state_path: json.dumps(build_state,ensure_ascii=False,indent=2).encode("utf-8")}
    config = dp.yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        listing = "".join("./"+r["image"]+"\n" for r in transformed if r["split"] == split and r["label"] is not None)
        updates[output / (split+".txt")] = listing.encode("utf-8")
        if listing:
            config[split] = str(output / (split+".txt"))
        else:
            config.pop(split, None)
    updates[output / "data.yaml"] = dp.yaml.safe_dump(config,allow_unicode=True,sort_keys=False).encode("utf-8")
    fpb_val = "".join("./"+r["image"]+"\n" for r in transformed if r["source"] == "fpb" and r["split"] == "val")
    if (output / "fpb_val.txt").exists():
        updates[output / "fpb_val.txt"] = fpb_val.encode("utf-8")
        updates[output / "fpb_eval.yaml"] = dp.yaml.safe_dump(config | {"val":str(output / "fpb_val.txt")}, allow_unicode=True).encode("utf-8")
    split_state = {"parameters":params,"manifest_sha256":new_hash,"summary":summary}
    updates[split_state_path] = json.dumps(split_state,ensure_ascii=False,indent=2).encode("utf-8")
    backups = {path:path.read_bytes() if path.exists() else None for path in updates}
    moved = []
    try:
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != old_hash:
            raise ValueError("Сборка изменилась во время расчёта разбиения.")
        for source,target in tqdm(operations,desc="Перенос изображений и меток",unit="файл"):
            target.parent.mkdir(parents=True,exist_ok=True)
            source.replace(target)
            moved.append((source,target))
        for path,content in updates.items():
            path.write_bytes(content)
        dp.validate_export(output)
    except BaseException:
        for path,content in backups.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        for source,target in reversed(moved):
            target.replace(source)
        raise
    return summary | {"status":"APPLIED"}
