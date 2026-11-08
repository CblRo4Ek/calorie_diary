"""EDA and optional exact/perceptual duplicate cleanup of a YOLO export."""
from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from scripts import dataset_pipeline as dp


def _visual_signature(rgb):
    thumbnail = np.asarray(rgb.resize((32, 32), Image.Resampling.BILINEAR)).copy()
    small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (small[:, :-1] > small[:, 1:]).ravel()
    return thumbnail, sum(int(value) << i for i, value in enumerate(bits))


def load_export(output):
    output = Path(output)
    manifest = output / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    names = dp.yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))["names"]
    images, objects = [], []
    for i, record in enumerate(records):
        images.append({k: v for k, v in record.items() if k != "objects"} | {
            "record_i": i, "key": record["source"] + ":" + record["source_id"],
            "labeled": record["objects"] is not None,
            "n_objects": len(record["objects"]) if record["objects"] is not None else np.nan,
        })
        for obj in record["objects"] or []:
            x, y, w, h = obj["bbox"]
            objects.append({"record_i": i, "key": images[-1]["key"], "source": record["source"],
                            "split": record["split"], "class_id": obj["class_id"], "class": names[obj["class_id"]],
                            "x": x, "y": y, "width_norm": w, "height_norm": h, "area_norm": w*h,
                            "width_px": w*record["width"], "height_px": h*record["height"]})
    return records, pd.DataFrame(images), pd.DataFrame(objects), names


def scan_images(data, output, images, *, workers=4, overwrite=False):
    """Read actual image dimensions/quality; cache features and RGB thumbnails."""
    data, output = Path(data), Path(output)
    signature = dp.fingerprint({"manifest": hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest(),
                                "features": 1, "thumbnail": 32})
    prefix = data / "metadata" / (output.name + ".eda")
    state_path, csv_path, thumbs_path = (Path(str(prefix) + suffix) for suffix in (".json", ".csv", ".npy"))
    if not overwrite and state_path.exists() and csv_path.exists() and thumbs_path.exists():
        if dp.read_json(state_path).get("signature") == signature:
            frame = pd.read_csv(csv_path, keep_default_na=False, dtype={"dhash": str})
            for column in ("brightness", "contrast", "sharpness", "entropy"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            thumbnails = np.load(thumbs_path, mmap_mode="r")
            if len(frame) == len(images) and thumbnails.shape == (len(images), 32, 32, 3):
                tqdm.write(f"EDA: кеш характеристик {len(frame):,} изображений.")
                return frame, thumbnails
    prefix.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(thumbs_path) + ".part")
    thumbnails = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(len(images), 32, 32, 3))

    def inspect(row):
        result = {"record_i": row["record_i"], "key": row["key"], "source": row["source"], "split": row["split"],
                  "width": 0, "height": 0, "mode": "", "brightness": np.nan, "contrast": np.nan,
                  "sharpness": np.nan, "entropy": np.nan, "dhash": "", "error": ""}
        thumbnail = np.zeros((32, 32, 3), dtype=np.uint8)
        try:
            path = output / row["image"]
            with Image.open(path) as photo:
                photo.verify()
            with Image.open(path) as photo:
                result.update(width=photo.width, height=photo.height, mode=photo.mode)
                photo.draft("RGB", (128, 128))
                rgb = photo.convert("RGB")
                thumbnail, digest = _visual_signature(rgb)
                gray_image = rgb.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
                gray = np.asarray(gray_image, dtype=np.float32)
                laplacian = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
                             - 4 * gray[1:-1, 1:-1])
                result.update(brightness=float(gray.mean()), contrast=float(gray.std()),
                              sharpness=float(laplacian.var()), entropy=float(gray_image.entropy()), dhash=f"{digest:016x}")
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        return result, thumbnail

    features = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Batches bound the number of pending futures and memory used by decoded photos.
        rows = images.to_dict("records")
        for start in tqdm(range(0, len(rows), 256), desc="EDA: чтение изображений", unit="пакет"):
            for result, thumbnail in pool.map(inspect, rows[start:start+256]):
                thumbnails[result["record_i"]] = thumbnail
                features.append(result)
    thumbnails.flush()
    del thumbnails
    temporary.replace(thumbs_path)
    frame = pd.DataFrame(features)
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    dp.dump_json(state_path, {"signature": signature, "images": len(frame),
                             "note": "Set overwrite=True after editing image files outside the pipeline."})
    return frame, np.load(thumbs_path, mmap_mode="r")


def similar_pairs(features, thumbnails, *, hamming=3, mae=0.02, aspect_tolerance=0.03):
    """Four 16-bit bands cover all 64-bit dHash pairs at distance <= 3."""
    if not 0 <= hamming <= 3:
        raise ValueError("hamming должен находиться между 0 и 3.")
    if mae < 0 or aspect_tolerance < 0:
        raise ValueError("Пороги RGB MAE и пропорций не могут быть отрицательными.")
    bands = [defaultdict(list) for _ in range(4)]
    hashes, aspects, found = {}, {}, []
    valid = features[features.error == ""]
    for row in tqdm(valid.to_dict("records"), desc="EDA: похожие изображения", unit="фото"):
        i, digest = int(row["record_i"]), int(row["dhash"], 16)
        aspect = row["width"] / row["height"]
        candidates = set()
        for b in range(4):
            candidates.update(bands[b][(digest >> (16*b)) & 65535])
        candidates = [j for j in sorted(candidates) if (digest ^ hashes[j]).bit_count() <= hamming
                      and abs(np.log(aspect / aspects[j])) <= aspect_tolerance]
        for start in range(0, len(candidates), 128):
            block = candidates[start:start+128]
            differences = np.mean(np.abs(np.asarray(thumbnails[block], dtype=np.int16)
                                         - np.asarray(thumbnails[i], dtype=np.int16)), axis=(1, 2, 3)) / 255
            for j, distance in zip(block, differences):
                if distance <= mae:
                    found.append({"a": j, "b": i, "hamming": (digest ^ hashes[j]).bit_count(), "rgb_mae": float(distance)})
        hashes[i], aspects[i] = digest, aspect
        for b in range(4):
            bands[b][(digest >> (16*b)) & 65535].append(i)
    return pd.DataFrame(found, columns=["a", "b", "hamming", "rgb_mae"])


def outlier_candidates(features, images, objects):
    """Flag heuristic quality/size tails for review; never delete them."""
    reasons = defaultdict(list)
    def flag(mask, reason):
        for i in features.loc[mask, "record_i"]:
            reasons[int(i)].append(reason)
    valid = features.error == ""
    flag(~valid, "unreadable_image")
    flag(valid & ((features.width < 128) | (features.height < 128)), "short_side_below_128")
    ratio = features.width / features.height.replace(0, np.nan)
    flag(valid & ((ratio > 3) | (ratio < 1/3)), "extreme_aspect_ratio")
    flag(valid & (features.brightness < 10) & (features.contrast < 10), "nearly_black")
    flag(valid & (features.brightness > 245) & (features.contrast < 10), "nearly_white")
    original = images.set_index("record_i")
    flag(valid & ((features.width != features.record_i.map(original.width))
                  | (features.height != features.record_i.map(original.height))), "manifest_size_mismatch")
    for _, group in features[valid].groupby("source"):
        for column in ("width", "height"):
            values = np.log(group[column])
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3-q1
            if spread > 0:
                for i in group.loc[(values < q1-3*spread) | (values > q3+3*spread), "record_i"]:
                    reasons[int(i)].append("unusual_size_for_source")
        threshold = group.sharpness.quantile(0.005)
        for i in group.loc[group.sharpness < threshold, "record_i"]:
            reasons[int(i)].append("lowest_0.5pct_sharpness_in_source")
    for row in objects[objects.area_norm < 0.0001].itertuples():
        reasons[int(row.record_i)].append("bbox_area_below_0.01pct")
    return pd.DataFrame([{"record_i": i, "reasons": ";".join(sorted(set(why)))} for i, why in sorted(reasons.items())],
                        columns=["record_i", "reasons"])


def class_distribution(records, names):
    rows = []
    for split in ("all", "train", "val", "test"):
        selected = records if split == "all" else [r for r in records if r["split"] == split or
                                                    (split == "test" and r["split"] == "fpb_test_unlabeled")]
        labeled = [r for r in selected if r["objects"] is not None]
        objects, images = Counter(), Counter()
        for record in labeled:
            objects.update(o["class_id"] for o in record["objects"])
            images.update({o["class_id"] for o in record["objects"]})
        for class_id, name in enumerate(names):
            rows.append({"class_id": class_id, "class": name, "split": split,
                         "objects": objects[class_id] if labeled else np.nan,
                         "images": images[class_id] if labeled else np.nan})
    return pd.DataFrame(rows)


def remove_exact_duplicates(data, output):
    """Remove only verified pixel copies with equivalent labels; keep original raw data."""
    data, output = Path(data).resolve(), Path(output).resolve()
    if not output.is_relative_to(data / "processed"):
        raise ValueError("Очистка разрешена только внутри data/processed/.")
    records, _, _, _ = load_export(output)
    groups = defaultdict(list)
    for record in records:
        groups[record["pixel_sha256"]].append(record)
    removed, conflicts = [], []
    def signature(record):
        if record["objects"] is None:
            return None
        return tuple(sorted((o["class_id"], tuple(round(v, 8) for v in o["bbox"])) for o in record["objects"]))
    for group in tqdm([g for g in groups.values() if len(g) > 1], desc="EDA: точные дубликаты", unit="группа"):
        for record in group:
            if dp.image_info(output / record["image"])[2] != record["pixel_sha256"]:
                raise ValueError("Изображение изменилось после сборки: автоматическое удаление остановлено.")
        signatures = {signature(r) for r in group if r["objects"] is not None}
        if len(signatures) > 1:
            conflicts.extend(r["source"] + ":" + r["source_id"] for r in group)
            continue
        ordered = sorted(group, key=dp.duplicate_priority)
        removed.extend(ordered[1:])
    audit = [{"source": r["source"], "source_id": r["source_id"], "reason": "pixel_duplicate", "stage": "eda"} for r in removed]
    count = _apply_removals(data, output, records, removed, audit)
    return {"removed": count, "conflicting_exact_records_for_review": conflicts}


def similar_removal_plan(records, pairs, *, hamming=3, mae=0.02, aspect_tolerance=0.03):
    """Greedy selection: each removed image must match a directly retained image."""
    if not 0 <= hamming <= 3 or mae < 0 or aspect_tolerance < 0:
        raise ValueError("Нужны hamming 0–3 и неотрицательные пороги RGB/пропорций.")
    neighbors = defaultdict(dict)
    for pair in pairs.itertuples():
        a, b = int(pair.a), int(pair.b)
        if a == b or not 0 <= a < len(records) or not 0 <= b < len(records):
            raise ValueError("Некорректные индексы пар; повторите поиск на текущем manifest.")
        left, right = records[a], records[b]
        aspect = abs(np.log((left["width"] / left["height"]) / (right["width"] / right["height"])))
        if pair.hamming <= hamming and pair.rgb_mae <= mae and aspect <= aspect_tolerance:
            edge = {"hamming": int(pair.hamming), "rgb_mae": float(pair.rgb_mae), "aspect_difference": float(aspect),
                    "reason": "pixel_duplicate" if left["pixel_sha256"] == right["pixel_sha256"] else "similar_image"}
            neighbors[a][b] = neighbors[b][a] = edge
    # Pixel copies are also included if differing file encodings affect reduced JPEG decoding.
    exact = defaultdict(list)
    for i, record in enumerate(records):
        exact[record["pixel_sha256"]].append(i)
    for group in exact.values():
        for offset, a in enumerate(group):
            for b in group[offset+1:]:
                neighbors[a][b] = neighbors[b][a] = {"hamming": 0, "rgb_mae": 0.0, "aspect_difference": 0.0,
                                                    "reason": "pixel_duplicate"}
    order = sorted(neighbors, key=lambda i: dp.duplicate_priority(records[i]))
    removed, retained, plan = set(), set(), []
    for i in order:
        if i in removed:
            continue
        retained.add(i)
        for j, edge in sorted(neighbors[i].items()):
            if j in removed or j in retained:
                continue
            removed.add(j)
            left, right = records[i], records[j]
            classes = [None if r["objects"] is None else sorted(o["class_id"] for o in r["objects"]) for r in (left, right)]
            plan.append({"keep_i": i, "remove_i": j, "keep_key": left["source"] + ":" + left["source_id"],
                         "remove_key": right["source"] + ":" + right["source_id"], "keep_split": left["split"],
                         "remove_split": right["split"], "same_classes": classes[0] == classes[1] if None not in classes else None} | edge)
    return pd.DataFrame(plan, columns=["keep_i", "remove_i", "keep_key", "remove_key", "keep_split", "remove_split",
                                       "same_classes", "hamming", "rgb_mae", "aspect_difference", "reason"])


def remove_similar_images(data, output, pairs, *, manifest_hash, hamming=3, mae=0.02, aspect_tolerance=0.03):
    """Apply the thresholds to the current manifest; recheck chosen pairs before moving files."""
    data, output = Path(data).resolve(), Path(output).resolve()
    if not output.is_relative_to(data / "processed"):
        raise ValueError("Очистка разрешена только внутри data/processed/.")
    if hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("Manifest изменился после поиска пар. Повторите EDA перед удалением.")
    records, _, _, _ = load_export(output)
    plan = similar_removal_plan(records, pairs, hamming=hamming, mae=mae, aspect_tolerance=aspect_tolerance)
    signatures, audit = {}, []
    def read(i):
        if i not in signatures:
            with Image.open(output / records[i]["image"]) as photo:
                aspect = photo.width / photo.height
                photo.draft("RGB", (128, 128))
                thumbnail, digest = _visual_signature(photo.convert("RGB"))
            signatures[i] = thumbnail, digest, aspect
        return signatures[i]
    for row in tqdm(plan.itertuples(), total=len(plan), desc="EDA: проверка удаляемых пар", unit="пара"):
        keep, remove = records[row.keep_i], records[row.remove_i]
        if row.reason == "pixel_duplicate":
            actual = [dp.image_info(output / r["image"])[2] for r in (keep, remove)]
            if actual[0] != actual[1] or actual[0] != keep["pixel_sha256"]:
                raise ValueError("Пиксели точной копии изменились; повторите EDA.")
            distance, difference, aspect = 0, 0.0, 0.0
        else:
            a, b = read(row.keep_i), read(row.remove_i)
            distance = (a[1] ^ b[1]).bit_count()
            difference = float(np.abs(a[0].astype(np.int16) - b[0].astype(np.int16)).mean() / 255)
            aspect = float(abs(np.log(a[2] / b[2])))
            if distance > hamming or difference > mae or aspect > aspect_tolerance:
                raise ValueError("Пара больше не соответствует порогам; обновите характеристики и поиск пар.")
        audit.append({"source": remove["source"], "source_id": remove["source_id"], "stage": "eda", "reason": row.reason,
                      "retained_source": keep["source"], "retained_source_id": keep["source_id"],
                      "hamming": distance, "rgb_mae": difference, "aspect_difference": aspect,
                      "thresholds": {"hamming": hamming, "rgb_mae": mae, "aspect_tolerance": aspect_tolerance}})
    removed = [records[i] for i in plan.remove_i]
    count = _apply_removals(data, output, records, removed, audit, expected_hash=manifest_hash)
    return {"removed": count, "by_reason": dict(Counter(row["reason"] for row in audit))}


def _apply_removals(data, output, records, removed, audit_entries, *, expected_hash=None):
    if not removed:
        return 0
    keys = {(r["source"], r["source_id"]) for r in removed}
    kept = [r for r in records if (r["source"], r["source_id"]) not in keys]
    if not all(any(r["split"] == split and r["label"] is not None for r in kept) for split in ("train", "val")):
        raise ValueError("Очистка оставит пустой train/val; требуется ручная проверка.")
    state_path = data / "metadata" / (output.name + ".build.json")
    state = dp.read_json(state_path)
    before = (output / "manifest.jsonl").read_bytes()
    current_hash = hashlib.sha256(before).hexdigest()
    if state["manifest_sha256"] != current_hash or (expected_hash is not None and current_hash != expected_hash):
        raise ValueError("Manifest изменён вне пайплайна; очистка остановлена.")
    manifest = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode("utf-8")
    updates = {output / "manifest.jsonl": manifest}
    for split in ("train", "val", "test"):
        listing = "".join("./" + r["image"] + "\n" for r in kept if r["split"] == split and r["label"] is not None)
        updates[output / (split + ".txt")] = listing.encode("utf-8")
    if (output / "fpb_val.txt").exists():
        updates[output / "fpb_val.txt"] = "".join("./" + r["image"] + "\n" for r in kept
                                                 if r["source"] == "fpb" and r["split"] == "val").encode("utf-8")
    audit_path = output / "excluded_records.json"
    audit = dp.read_json(audit_path) if audit_path.exists() else []
    audit += audit_entries
    updates[audit_path] = json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8")
    state["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    updates[state_path] = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
    backups = {path: path.read_bytes() if path.exists() else None for path in updates}
    cache = data / "cache"
    if not cache.resolve().is_relative_to(data):
        raise ValueError("Кеш выходит за пределы data/.")
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eda_cleanup_", dir=cache) as temporary:
        temporary = Path(temporary).resolve()
        if not temporary.is_relative_to(cache.resolve()):
            raise ValueError("Временная папка вне data/cache/.")
        moved = []
        try:
            for record in removed:
                for relative in (record["image"], record["label"]):
                    if relative is None:
                        continue
                    source, target = output / relative, temporary / relative
                    if not source.resolve().is_relative_to(output) or not target.resolve().is_relative_to(temporary):
                        raise ValueError("Путь изображения/метки выходит за пределы датасета.")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(target)
                    moved.append((source, target))
            for path, content in updates.items():
                path.write_bytes(content)
            dp.validate_export(output)
        except BaseException:
            for path, content in backups.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
            for source, target in reversed(moved):
                target.replace(source)
            raise
    return len(removed)
