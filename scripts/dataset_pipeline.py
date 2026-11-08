"""FPB + MyFoodRepo v2.1 ingestion. Source labels are never silently treated as background.

All downloaded assets, caches, decisions and exports live under the supplied data/.
Food-101 is an optional unlabeled inference sample, never a training source.
Only the first five YOLO fields survive FPB ingestion; weights are discarded.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import random
import shutil
import ssl
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml
from PIL import Image
from tqdm.auto import tqdm

FPB_REPO = "issai/Food_Portion_Benchmark"
FOOD_REPO = "ethz/food101"
FPB_PREFIX = "FPB_Dataset/RGB"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_sessions = threading.local()
_catalog_cache = {}
_tree_cache = {}
_preview_cache = {}


def configure_https(data_dir):
    """Use Windows system trust (including a corporate CA), without disabling TLS."""
    if os.name != "nt" or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    path = Path(data_dir).resolve() / "cache/system-ca.pem"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        certs = ssl.create_default_context().get_ca_certs(binary_form=True)
        content = Path(requests.certs.where()).read_text(encoding="ascii")
        content += "\n" + "".join(ssl.DER_cert_to_PEM_cert(cert) for cert in certs)
        path.write_text(content, encoding="ascii")
    os.environ["REQUESTS_CA_BUNDLE"] = str(path)
    # Expose the same trust bundle to other Python HTTP clients in notebooks.
    os.environ.setdefault("SSL_CERT_FILE", str(path))


def project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "README.md").is_file() and (path / "scripts/dataset_pipeline.py").is_file():
            return path
    raise FileNotFoundError("Откройте ноутбук из корня calorie_diary или notebooks/.")


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def request(url, *, params=None, stream=False, headers=None):
    if not hasattr(_sessions, "session"):
        _sessions.session = requests.Session()
        _sessions.session.headers["User-Agent"] = "calorie-diary-dataset-research/1.0"
    for attempt in range(4):
        try:
            response = _sessions.session.get(url, params=params, timeout=(15, 60), stream=stream, headers=headers)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if exc.response is not None and exc.response.status_code in {400, 401, 403, 404, 422}:
                raise
            if attempt == 3:
                raise
            retry_after = exc.response.headers.get("Retry-After", "0") if exc.response is not None else "0"
            delay = float(retry_after) if retry_after.isdigit() else 0
            time.sleep(min(max(delay, 2 ** attempt), 30))


def download(url, path):
    """Atomic download; an interrupted .part file is never considered a valid image."""
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with request(url, stream=True) as response, temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
            temporary.replace(path)
            return path
        except requests.RequestException:
            if attempt == 2:
                raise


def catalogs(data_dir, offline=False, *, myfoodrepo_source="download", train_archive=None, valid_archive=None):
    """Pin FPB and MyFoodRepo's selected input source and author splits."""
    from scripts import myfoodrepo as mfr
    configure_https(data_dir)
    if myfoodrepo_source not in {"download", "archives"}:
        raise ValueError("MYFOODREPO_SOURCE должен быть download или archives.")
    archives = {}
    if myfoodrepo_source == "archives":
        if train_archive is None or valid_archive is None:
            raise ValueError("Укажите train и valid архивы MyFoodRepo v2.1.")
        archives["train"] = mfr.archive_info(train_archive)
    if valid_archive is not None:
        archives["val"] = mfr.archive_info(valid_archive)
    inputs = {"input_source": myfoodrepo_source, "archives": archives,
              "splits": ["train", "val"] if "val" in archives else ["train"]}
    key = (str(Path(data_dir).resolve()), fingerprint(inputs))
    if key in _catalog_cache:
        return _catalog_cache[key]
    lock_name = "fpb_myfoodrepo_archives_sources.lock.json" if archives else "fpb_myfoodrepo_sources.lock.json"
    path = Path(data_dir) / "metadata" / lock_name
    if path.exists():
        saved = read_json(path)
        if (saved["myfoodrepo"].get("input_source", "download") == myfoodrepo_source
                and saved["myfoodrepo"].get("archives", {}) == archives):
            _catalog_cache[key] = saved
            return saved
    fpb_cache = next((p for p in (path, Path(data_dir) / "metadata/fpb_myfoodrepo_sources.lock.json",
                                 Path(data_dir) / "metadata/sources.lock.json") if p.exists()), None)
    if fpb_cache is not None:
        fpb = read_json(fpb_cache)["fpb"]
    else:
        if offline:
            raise FileNotFoundError("Отсутствует кеш каталога FPB.")
        info = request(f"https://huggingface.co/api/datasets/{FPB_REPO}").json()
        config = yaml.safe_load(request(f"https://huggingface.co/datasets/{FPB_REPO}/raw/{info['sha']}/{FPB_PREFIX}/data.yaml").text)
        names = config["names"]
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
        if len(names) != 138:
            raise ValueError("Изменился словарь классов FPB.")
        fpb = {"repo": FPB_REPO, "revision": info["sha"], "names": names}
    coco = mfr.load_annotations(data_dir, offline=offline, catalog=inputs)
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    if "val" in archives:
        validation = mfr.load_annotations(data_dir, offline=offline, catalog=inputs, split="val")
        if {c["id"]: c["name"] for c in validation["categories"]} != {c["id"]: c["name"] for c in categories}:
            raise ValueError("Словари классов train и valid архивов не совпадают.")
    result = {"fpb": fpb, "myfoodrepo": {
        "repo": mfr.DATASET_REF if myfoodrepo_source == "download" else "AIcrowd/MyFoodRepo-v2.1",
        "revision": str(mfr.MIRROR_VERSION) if myfoodrepo_source == "download" else fingerprint(archives)[:12],
        "dataset_version": "2.1", "source": mfr.SOURCE_PAGE, "mirror": mfr.MIRROR_PAGE,
        "names": [c["name"] for c in categories], "category_ids": [c["id"] for c in categories],
    } | inputs}
    dump_json(path, result)
    _catalog_cache[key] = result
    return result


def repo_tree(data_dir, catalog, subpath):
    key = (str(Path(data_dir).resolve()), catalog["repo"], catalog["revision"], subpath)
    if key in _tree_cache:
        return _tree_cache[key]
    path = Path(data_dir) / "metadata/trees" / catalog["revision"] / (subpath.replace("/", "__") + ".json")
    if path.exists():
        _tree_cache[key] = read_json(path)
        return _tree_cache[key]
    url = f'https://huggingface.co/api/datasets/{catalog["repo"]}/tree/{catalog["revision"]}/{subpath}'
    entries = []
    params = {"limit": 1000}
    while url:
        response = request(url, params=params)
        entries.extend(item for item in response.json() if item["type"] == "file")
        url = response.links.get("next", {}).get("url")
        params = None
    _tree_cache[key] = entries
    dump_json(path, entries)
    return entries


def parse_yolo(text, num_classes):
    """Accept exactly 5/6 fields, discard field 6 without parsing or storing it."""
    objects = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) not in {5, 6}:
            raise ValueError(f"YOLO строка {line_number}: нужно 5/6 полей, получено {len(fields)}")
        class_id = int(fields[0])
        box = list(map(float, fields[1:5]))
        if not 0 <= class_id < num_classes or not all(math.isfinite(x) for x in box):
            raise ValueError(f"Некорректный класс/координаты: строка {line_number}")
        x, y, w, h = box
        if not (0 < w <= 1+1e-6 and 0 < h <= 1+1e-6 and -1e-6 <= x-w/2 <= x+w/2 <= 1+1e-6
                and -1e-6 <= y-h/2 <= y+h/2 <= 1+1e-6):
            raise ValueError(f"Рамка вне изображения: строка {line_number}, {box}")
        # Decimal serialization can move an edge by ~1e-10. Snap only this tolerance.
        left, right = max(0.0, x-w/2), min(1.0, x+w/2)
        top, bottom = max(0.0, y-h/2), min(1.0, y+h/2)
        box = [(left+right)/2, (top+bottom)/2, right-left, bottom-top]
        objects.append({"source_class_id": class_id, "bbox": box})
    return objects


def yolo_text(objects, key="source_class_id"):
    return "".join(f'{item[key]} ' + " ".join(f"{v:.10f}" for v in item["bbox"]) + "\n" for item in objects)


def fpb_index(data_dir, catalog, splits=("train", "val", "test")):
    records = []
    for split in splits:
        image_files = repo_tree(data_dir, catalog, f"{FPB_PREFIX}/{split}/images")
        # Hidden test labels are not assumed to exist; even an empty label is meaningful.
        label_files = [] if split == "test" else repo_tree(data_dir, catalog, f"{FPB_PREFIX}/{split}/labels")
        label_paths = {Path(item["path"]).stem: item["path"] for item in label_files}
        seen = set()
        for item in image_files:
            remote = item["path"]
            if Path(remote).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = Path(remote).stem
            if stem in seen:
                raise ValueError(f"Повторяющийся stem в FPB {split}: {stem}")
            seen.add(stem)
            label = label_paths.get(stem)
            if split != "test" and label is None:
                raise ValueError(f"Отсутствует метка FPB {split}: {remote}")
            base = re.sub(r"\.rf\..*$", "", stem)
            base = re.sub(r"_\d+_(?:png|jpg|jpeg)$", "", base, flags=re.I)
            records.append({"source": "fpb", "source_id": f"{split}/{stem}", "source_split": split,
                            "remote_image": remote, "remote_label": label,
                            "group_id": f"fpb/{base}", "box_origin": "native"})
    return records


def fpb_objects(data_dir, catalog, record):
    if record["source_split"] == "test":
        return None
    relative = Path(record["remote_label"]).relative_to(FPB_PREFIX)
    path = Path(data_dir) / "raw/fpb" / catalog["revision"] / relative
    if path.exists():
        return parse_yolo(path.read_text(encoding="utf-8"), len(catalog["names"]))
    url = f'https://huggingface.co/datasets/{catalog["repo"]}/raw/{catalog["revision"]}/{record["remote_label"]}'
    objects = parse_yolo(request(url).text, len(catalog["names"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yolo_text(objects), encoding="utf-8")
    return objects


def fpb_materialize(data_dir, catalog, record, objects):
    relative = Path(record["remote_image"]).relative_to(FPB_PREFIX)
    path = Path(data_dir) / "raw/fpb" / catalog["revision"] / relative
    url = f'https://huggingface.co/datasets/{catalog["repo"]}/resolve/{catalog["revision"]}/{record["remote_image"]}'
    download(url, path)
    return {**record, "image": str(path.resolve()), "objects": objects}


def prepare_fpb_preview(data_dir, catalog, wanted, per_class=3, split="train", max_scan=1200, workers=8):
    records = fpb_index(data_dir, catalog, (split,))
    records.sort(key=lambda r: fingerprint(r["source_id"]))
    wanted = set(wanted)
    counts = Counter()
    selected = []
    scanned = 0
    limit = min(len(records), max_scan)
    with tqdm(total=limit, desc="FPB: поиск примеров", unit="метка") as progress:
        for offset in range(0, limit, 64):
            batch = records[offset:min(offset+64, max_scan)]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                annotations = list(pool.map(lambda r: fpb_objects(data_dir, catalog, r), batch))
            for record, objects in zip(batch, annotations):
                scanned += 1
                names = {catalog["names"][obj["source_class_id"]] for obj in objects}
                needed = {name for name in names & wanted if counts[name] < per_class}
                if needed:
                    selected.append((record, objects))
                    counts.update(names & wanted)
            progress.update(len(batch))
            if all(counts[name] >= per_class for name in wanted):
                progress.total = scanned
                progress.refresh()
                break
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(tqdm(
            pool.map(lambda args: fpb_materialize(data_dir, catalog, *args), selected),
            total=len(selected), desc="FPB preview: подготовка фото", unit="фото",
        ))
    return result, {"scanned": scanned, "counts": dict(counts),
                    "missing": sorted(name for name in wanted if counts[name] < per_class)}


def food_rows(catalog, split, offset, length):
    result = request("https://datasets-server.huggingface.co/rows", params={
        "dataset": catalog["repo"], "config": "default", "split": split, "offset": offset, "length": length,
    }).json()
    for item in result["rows"]:
        src = item["row"]["image"]["src"]
        if catalog["revision"] not in src:
            raise ValueError("Viewer переключился на другую revision. Используйте полную загрузку Parquet.")
    return result["rows"]


def preview_samples(data_dir, cats, per_class=2, max_scan=1600, workers=8):
    from scripts import myfoodrepo as mfr
    frame = review_table(cats)
    config = {"catalogs": cats, "per_class": per_class, "max_scan": max_scan, "pipeline": 2}
    key = (str(Path(data_dir).resolve()), fingerprint(config))
    cache = Path(data_dir) / "metadata" / f"fpb_myfoodrepo_preview_samples_{fingerprint(config)[:12]}.json"
    if key not in _preview_cache and cache.exists():
        _preview_cache[key] = read_json(cache)
    if key in _preview_cache:
        payload = _preview_cache[key]
        if all(Path(r["image"]).is_file() for r in payload["records"]):
            tqdm.write(f"Preview: {len(payload['records'])} фото уже подготовлены.")
            return payload["records"], payload["coverage"]
    fpb_records, coverage = prepare_fpb_preview(data_dir, cats["fpb"], frame["fpb_class"],
                                               per_class=per_class, max_scan=max_scan, workers=workers)
    wanted = sorted(set(frame["myfoodrepo_class"]) | {"avocado", "banana", "tomato"})
    other = mfr.prepare(data_dir, cats["myfoodrepo"], wanted=wanted, per_class=per_class, workers=workers, valid_limit=6)
    for split, limit in (("val", 6), ("test", 2)):
        for record in tqdm(fpb_index(data_dir, cats["fpb"], (split,))[:limit], desc=f"FPB {split}: preview", unit="фото"):
            fpb_records.append(fpb_materialize(data_dir, cats["fpb"], record, fpb_objects(data_dir, cats["fpb"], record)))
    payload = {"records": fpb_records + other, "coverage": coverage}
    dump_json(cache, payload)
    _preview_cache[key] = payload
    return payload["records"], coverage


def prepare_full(data_dir, cats, workers=8):
    from scripts import myfoodrepo as mfr
    fpb_records = fpb_index(data_dir, cats["fpb"])
    def prepare(record):
        return fpb_materialize(data_dir, cats["fpb"], record, fpb_objects(data_dir, cats["fpb"], record))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(prepare, fpb_records), total=len(fpb_records),
                            desc="FPB: фото и разметка", unit="фото"))
    return records + mfr.prepare(data_dir, cats["myfoodrepo"], workers=workers)


# These are hypotheses for visual review, never automatic synonym merges.
SYNONYM_CANDIDATES = [
    ("almond", "almonds", "likely_alias", "Единственное/множественное число"),
    ("apple_juice", "juice_apple", "likely_alias", "Яблочный сок"),
    ("capuccino", "cappuccino", "likely_alias", "Разное написание капучино"),
    ("french_fries", "chips_french_fries", "likely_alias", "Картофель фри"),
    ("icecream", "ice_cream_n_s", "likely_alias", "Общий класс мороженого"),
    ("raisins", "raisins_dried", "likely_alias", "Изюм"),
    ("toast", "bread_toast", "likely_alias", "Тостовый хлеб"),
    ("beef", "beef_n_s", "likely_alias", "Общий класс говядины"),
    ("chocolate", "chocolate_n_s", "likely_alias", "Общий класс шоколада"),
    ("kefir", "kefir_drink", "likely_alias", "Кефир"),
    ("burger", "hamburger", "related", "Общий класс и более узкая категория"),
    ("bliny", "crepe_with_flour_plain", "related", "Проверить вид блинов и состав"),
    ("bliny", "pancake", "related", "Блины и панкейки могут различаться"),
    ("black_tea", "tea", "related", "Tea включает несколько видов чая"),
    ("pizza", "pizza_margherita_baked", "related", "Общий класс и вид пиццы"),
    ("pizza", "pizza_with_vegetables_baked", "related", "Общий класс и вид пиццы"),
    ("mashed_potato", "mashed_potatoes_prepared_with_full_fat_milk_with_butter", "related", "Конкретная рецептура"),
    ("raspberry_jam", "jam", "related", "Малиновое варенье и общий класс"),
    ("lentiil_soup", "soup_of_lentils_dahl_dhal", "related", "Чечевичный суп и конкретный вид"),
    ("coke", "cola_based_drink", "related", "Бренд и общая категория"),
    ("cupcake", "muffin", "related", "Разные виды выпечки"),
    ("dried_apricot", "apricots", "related", "Сушёный и свежий продукт"),
]

def review_table(cats):
    import pandas as pd
    rows = [{"fpb_class": name, "myfoodrepo_class": name, "kind": "exact_name",
             "decision": "merge", "reviewed": True, "note": "Одинаковое имя; проверьте визуально"}
            for name in sorted(set(cats["fpb"]["names"]) & set(cats["myfoodrepo"]["names"]))]
    rows += [{"fpb_class": a, "myfoodrepo_class": b, "kind": kind, "decision": "keep_separate",
              "reviewed": False, "note": note} for a, b, kind, note in SYNONYM_CANDIDATES
             if a in cats["fpb"]["names"] and b in cats["myfoodrepo"]["names"]]
    return pd.DataFrame(rows)


def save_decisions(frame, path, cats):
    validate_decisions(frame, cats)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def validate_decisions(frame, cats):
    if "myfoodrepo_class" not in frame:
        raise ValueError("Нужна отдельная таблица решений FPB + MyFoodRepo.")
    if frame["myfoodrepo_class"].duplicated().any():
        raise ValueError("Класс MyFoodRepo указан в решениях несколько раз.")
    for row in frame.to_dict("records"):
        if row["fpb_class"] not in cats["fpb"]["names"] or row["myfoodrepo_class"] not in cats["myfoodrepo"]["names"]:
            raise ValueError("Неизвестный класс в решениях.")
        if row["decision"] not in {"merge", "keep_separate"}:
            raise ValueError("Решение должно быть merge или keep_separate.")
        if row["decision"] == "merge" and str(row["reviewed"]).lower() != "true":
            raise ValueError(f"Подтвердите reviewed=True для {row['myfoodrepo_class']}")


def class_mapping(cats, decisions):
    validate_decisions(decisions, cats)
    names = list(cats["fpb"]["names"])
    targets = {row["myfoodrepo_class"]: row for row in decisions.to_dict("records")}
    mapping = {("fpb", idx): idx for idx in range(len(names))}
    rows = [{"source": "fpb", "source_class_id": idx, "source_class": name,
             "class_id": idx, "class_name": name, "rule": "fpb_base"} for idx, name in enumerate(names)]
    for source_id, name in zip(cats["myfoodrepo"]["category_ids"], cats["myfoodrepo"]["names"]):
        decision = targets.get(name)
        if decision and decision["decision"] == "merge":
            target = decision["fpb_class"]
            rule = "exact_name" if name == target else "reviewed_alias"
        else:
            target = f"myfoodrepo__{name}" if name in names else name
            rule = "myfoodrepo_separate"
        if target not in names:
            names.append(target)
        class_id = names.index(target)
        mapping[("myfoodrepo", source_id)] = class_id
        rows.append({"source": "myfoodrepo", "source_class_id": source_id, "source_class": name,
                     "class_id": class_id, "class_name": target, "rule": rule})
    return names, mapping, rows


def assign_splits(records, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction должен находиться между 0 и 1.")
    result = [dict(record) for record in records]
    other = []
    has_native_val = any(r["source"] == "myfoodrepo" and r["source_split"] in {"val", "valid"} for r in result)
    for record in result:
        if record["source"] == "fpb":
            record["split"] = "fpb_test_unlabeled" if record["source_split"] == "test" else record["source_split"]
        elif record["source"] == "myfoodrepo" and record["source_split"] in {"val", "valid"}:
            record["split"] = "val"
        elif record["source"] == "myfoodrepo" and record["source_split"] == "train":
            if has_native_val or record.get("preserve_source_split", False):
                record["split"] = "train"
            else:
                other.append(record)
        else:
            raise ValueError(f"Источник не включён в обучение: {record['source']}")
    ordered = sorted(other, key=lambda r: fingerprint([seed, r["group_id"]]))
    val_count = max(1, round(len(ordered) * val_fraction)) if len(ordered) > 1 else 0
    for i, record in enumerate(tqdm(ordered, desc="Разбиение MyFoodRepo", unit="фото", disable=not ordered)):
        record["split"] = "val" if i < val_count else "train"
    return result


def image_info(path):
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256(f"{width}x{height}".encode() + rgb.tobytes()).hexdigest()
        # dHash is an audit hint, not an automatic assertion of equal portions.
        small = rgb.convert("L").resize((9, 8))
        pixels = list(small.tobytes())
        dhash = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
        orientation = image.getexif().get(274, 1)
    return width, height, digest, f"{dhash:016x}", orientation


def deduplicate(records):
    """Deduplicate exact pixels and quarantine conflicting annotations."""
    rank = {"train": 0, "val": 1, "test": 2, "fpb_test_unlabeled": 3}
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["pixel_sha256"]].append(record)
    kept, audit = [], []
    for group in tqdm(by_hash.values(), desc="Сборка: удаление дубликатов", unit="группа"):
        signatures = {fingerprint(sorted((o["class_id"], tuple(round(v, 8) for v in o["bbox"])) for o in record["objects"])) for record in group if record["objects"] is not None}
        if len(signatures) > 1:
            audit.extend({"source": r["source"], "source_id": r["source_id"], "reason": "conflicting_duplicate"} for r in group)
            continue
        ordered = sorted(group, key=lambda r: (-rank[r["split"]], r["source"], r["source_id"]))
        kept.append(ordered[0])
        audit.extend({"source": r["source"], "source_id": r["source_id"], "reason": "pixel_duplicate"} for r in ordered[1:])
    return sorted(kept, key=lambda r: (r["source"], r["source_id"])), audit


def dataset_path(data_dir, mode):
    if mode not in {"preview", "full"}:
        raise ValueError("MODE должен быть preview или full")
    return Path(data_dir).resolve() / "processed" / f"fpb_myfoodrepo_{mode}"


def _build_signature(cats, decisions, mode, val_fraction, seed):
    return fingerprint({"pipeline": "fpb_myfoodrepo_v2_1_pixels_only", "catalogs": cats, "decisions": decisions.to_dict("records"),
                        "mode": mode, "val_fraction": val_fraction, "seed": seed})


def _dataset_summary(mode, manifest, num_classes, excluded):
    return {"mode": mode, "classes": num_classes, "images": len(manifest), "excluded": excluded,
            "by_source_split": dict(Counter(f'{r["source"]}/{r["split"]}' for r in manifest)),
            "unlabeled_fpb_test": sum(r["label"] is None for r in manifest),
            "portion_grouping": "Exact pixel duplicates only; group_id does not filter splits."}


def load_dataset(data_dir, cats, decisions, mode="preview", val_fraction=0.1, seed=42):
    """Reuse a completed export without downloading, splitting or hashing images."""
    import pandas as pd
    output = dataset_path(data_dir, mode)
    if not output.exists():
        return None
    state_path = Path(data_dir) / "metadata" / f"{output.name}.build.json"
    required = [output / name for name in ("data.yaml", "manifest.jsonl", "class_mapping.csv", "train.txt", "val.txt")]
    if not state_path.is_file() or not all(path.is_file() for path in required):
        raise ValueError("Сборка не завершена. Установите OVERWRITE = True для повторной подготовки.")
    state = read_json(state_path)
    if state["signature"] != _build_signature(cats, decisions, mode, val_fraction, seed):
        raise ValueError("Параметры сборки изменились. Установите OVERWRITE = True для пересборки.")
    manifest_bytes = (output / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != state["manifest_sha256"]:
        raise ValueError("Manifest отличается от завершённой сборки. Установите OVERWRITE = True.")
    manifest = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines()]
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    audit_path = output / "excluded_records.json"
    summary = _dataset_summary(mode, manifest, len(config["names"]), len(read_json(audit_path)) if audit_path.exists() else 0)
    summary["validation"] = {"status": "REUSED", "checked_images": 0}
    tqdm.write(f"Используется готовая сборка: {output.name} ({len(manifest)} изображений).")
    return output, pd.DataFrame(manifest), summary


def build_dataset(data_dir, cats, records, decisions, mode="preview", val_fraction=0.1, seed=42, overwrite=False):
    """Keep one export per mode; reuse it unless overwrite is explicitly enabled."""
    data_dir = Path(data_dir).resolve()
    output = dataset_path(data_dir, mode)
    if not overwrite:
        saved = load_dataset(data_dir, cats, decisions, mode, val_fraction, seed)
        if saved is not None:
            return saved
    cache = data_dir / "cache"
    # Resolve before creating/moving/removing anything, including Windows junctions.
    for target in (output, cache):
        if not target.resolve().is_relative_to(data_dir):
            raise ValueError(f"Путь выходит за пределы data/: {target}")
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fpb_myfoodrepo_build_", dir=cache) as temporary:
        staging = Path(temporary) / "dataset"
        backup = Path(temporary) / "previous"
        _, merged, summary = _write_dataset(
            cats, records, decisions, staging, output,
            mode=mode, val_fraction=val_fraction, seed=seed,
        )
        if output.exists():
            output.replace(backup)
        try:
            staging.replace(output)
        except OSError:
            if backup.exists():
                backup.replace(output)
            raise
    dump_json(data_dir / "metadata" / f"{output.name}.build.json", {
        "signature": _build_signature(cats, decisions, mode, val_fraction, seed),
        "manifest_sha256": hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest(),
    })
    return output, merged, summary


def _write_dataset(cats, records, decisions, output, final_output, *, mode, val_fraction, seed):
    """Write and validate a temporary export; YAML paths target its final location."""
    import pandas as pd
    names, mapping, mapping_rows = class_mapping(cats, decisions)
    records = assign_splits(records, val_fraction, seed)
    seen = set()
    for record in tqdm(records, desc="Сборка: проверка и хеширование", unit="фото"):
        key = (record["source"], record["source_id"])
        if key in seen:
            raise ValueError(f"Повторяющийся source_id: {key}")
        seen.add(key)
        width, height, digest, dhash, orientation = image_info(record["image"])
        record.update(width=width, height=height, pixel_sha256=digest, dhash=dhash, orientation=orientation)
        if record["objects"] is not None:
            record["objects"] = [{"class_id": mapping[(record["source"], obj["source_class_id"])],
                                  "source_class_id": obj["source_class_id"], "bbox": list(obj["bbox"])}
                                 for obj in record["objects"]]
    kept, audit = deduplicate(records)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for record in tqdm(kept, desc="Сборка: запись изображений и меток", unit="фото"):
        uid = record["source"] + "__" + fingerprint(record["source_id"])[:24]
        source = Path(record["image"])
        destination = output / "images" / record["split"] / (uid + source.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if record["orientation"] not in (None, 1):
                # Preserve the raw pixel axes used by boxes, remove misleading EXIF rotation.
                with Image.open(source) as image:
                    image.convert("RGB").save(destination)
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        label = None
        if record["objects"] is not None:
            label = output / "labels" / record["split"] / (uid + ".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(yolo_text(record["objects"], "class_id"), encoding="utf-8")
        manifest.append({k: v for k, v in record.items() if k not in {"image", "remote_image", "remote_label", "orientation"}} |
                        {"image": destination.relative_to(output).as_posix(),
                         "label": label.relative_to(output).as_posix() if label else None})
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in manifest:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(mapping_rows).to_csv(output / "class_mapping.csv", index=False, encoding="utf-8")
    if audit:
        dump_json(output / "excluded_records.json", audit)
    config = {"path": str(final_output), "names": names, "nc": len(names)}
    for split in ("train", "val", "test"):
        paths = [r["image"] for r in manifest if r["split"] == split and r["label"] is not None]
        listing = output / f"{split}.txt"
        listing.write_text("".join(f"./{path}\n" for path in paths), encoding="utf-8")
        if paths:
            config[split] = str(final_output / listing.name)
    if "train" not in config or "val" not in config:
        raise ValueError("После очистки нет train/val: увеличьте preview или проверьте дубликаты.")
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    fpb_val = [r["image"] for r in manifest if r["source"] == "fpb" and r["split"] == "val"]
    if fpb_val:
        listing = output / "fpb_val.txt"
        listing.write_text("".join(f"./{path}\n" for path in fpb_val), encoding="utf-8")
        (output / "fpb_eval.yaml").write_text(yaml.safe_dump(config | {"val": str(final_output / listing.name)}, allow_unicode=True), encoding="utf-8")
    summary = _dataset_summary(mode, manifest, len(names), len(audit))
    summary["validation"] = validate_export(output)
    return output, pd.DataFrame(manifest), summary


def validate_export(output):
    output = Path(output)
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    splits_by_hash = defaultdict(set)
    for record in tqdm(records, desc="Проверка сохранённого датасета", unit="фото"):
        image_path = output / record["image"]
        with Image.open(image_path) as image:
            image.verify()
        splits_by_hash[record["pixel_sha256"]].add(record["split"])
        if record["label"] is None:
            if record["source"] != "fpb" or record["split"] != "fpb_test_unlabeled":
                raise AssertionError("Неожиданное изображение без метки.")
            continue
        text = (output / record["label"]).read_text(encoding="utf-8")
        if any(len(line.split()) != 5 for line in text.splitlines()):
            raise AssertionError("В экспорте должно быть ровно 5 полей YOLO.")
        objects = parse_yolo(text, len(config["names"]))
        expected = [{"source_class_id": obj["class_id"], "bbox": obj["bbox"]} for obj in record["objects"]]
        if len(objects) != len(expected):
            raise AssertionError("Количество объектов в TXT и manifest не совпало.")
        for actual, target in zip(objects, expected):
            if actual["source_class_id"] != target["source_class_id"] or any(abs(a-b) > 1e-9 for a,b in zip(actual["bbox"], target["bbox"])):
                raise AssertionError("ID/координаты TXT и manifest не совпали.")
        if record["source"] not in {"fpb", "myfoodrepo"}:
            raise AssertionError("Источник не должен входить в обучающий датасет.")
    if any(len(x) != 1 for x in splits_by_hash.values()):
        raise AssertionError("Обнаружено пересечение splits по пикселям.")
    return {"checked_images": len(records), "classes": len(config["names"]), "status": "OK"}


def show_records(records, cats, *, output_root=None, count=8, title=""):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    records = list(records)[:count]
    if not records:
        print("Нет примеров для отображения.")
        return None
    columns = min(4, len(records))
    fig, axes = plt.subplots(math.ceil(len(records)/columns), columns, figsize=(4*columns, 4*math.ceil(len(records)/columns)), squeeze=False)
    for axis, record in zip(axes.flat, records):
        path = Path(output_root) / record["image"] if output_root else Path(record["image"])
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
            width, height = image.size
        for obj in record["objects"] or []:
            if "class_id" in obj and output_root:
                names = cats
                name = names[obj["class_id"]]
            else:
                info = cats[record["source"]]
                name = dict(zip(info.get("category_ids", range(len(info["names"]))), info["names"]))[obj["source_class_id"]]
            x,y,w,h = obj["bbox"]
            axis.add_patch(Rectangle(((x-w/2)*width,(y-h/2)*height),w*width,h*height,fill=False,edgecolor="lime",linewidth=2))
            axis.text(max(0,(x-w/2)*width),max(0,(y-h/2)*height),name,fontsize=8,color="black",bbox={"facecolor":"lime","alpha":0.8,"pad":1})
        axis.set_title(f'{record["source"]}: {record.get("split", record["source_split"])}\n{record["box_origin"]}', fontsize=9)
        axis.axis("off")
    for axis in list(axes.flat)[len(records):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return fig


def show_class_pair(records, cats, fpb_class, myfoodrepo_class, samples=3):
    """Compare object crops from both training sources."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, samples, figsize=(4*samples, 7), squeeze=False)
    for row, (source, name) in enumerate((("fpb", fpb_class), ("myfoodrepo", myfoodrepo_class))):
        info = cats[source]
        source_id = dict(zip(info["names"], info.get("category_ids", range(len(info["names"])))))[name]
        candidates = [r for r in records if r["source"] == source and r["source_split"] == "train"
                      and any(o["source_class_id"] == source_id for o in r["objects"] or [])]
        for col in range(samples):
            axis = axes[row, col]
            if col < len(candidates):
                record = candidates[col]
                with Image.open(record["image"]) as image:
                    image = image.convert("RGB")
                    if source in {"fpb", "myfoodrepo"}:
                        obj = next(o for o in record["objects"] if o["source_class_id"] == source_id)
                        x,y,w,h = obj["bbox"]
                        width,height = image.size
                        image = image.crop((max(0, math.floor((x-w/2)*width)), max(0, math.floor((y-h/2)*height)),
                                            min(width, math.ceil((x+w/2)*width)), min(height, math.ceil((y+h/2)*height))))
                    axis.imshow(image)
                axis.set_title(f'{source}: {name}\n{"object bbox crop"}', fontsize=10)
            else:
                axis.text(0.5,0.5,"No sample in preview\nIncrease max_scan / use full data",ha="center",va="center")
            axis.axis("off")
    fig.suptitle(f"{fpb_class}  /  {myfoodrepo_class}")
    fig.tight_layout()
    plt.show()
    return fig


def food101_catalog(data_dir):
    configure_https(data_dir)
    path = Path(data_dir) / "metadata/food101_inference_sources.lock.json"
    if path.exists():
        return read_json(path)
    legacy = Path(data_dir) / "metadata/sources.lock.json"
    if legacy.exists() and "food101" in read_json(legacy):
        old = read_json(legacy)["food101"]
        info = {"repo": old["repo"], "revision": old["revision"]}
    else:
        info = {"repo": FOOD_REPO, "revision": request(f"https://huggingface.co/api/datasets/{FOOD_REPO}").json()["sha"]}
    dump_json(path, info)
    return info


def food101_inference_sample(data_dir, n_images=20, seed=42, overwrite=False):
    if not 1 <= n_images <= 25250:
        raise ValueError("N_IMAGES должен быть от 1 до 25250.")
    catalog = food101_catalog(data_dir)
    output = Path(data_dir).resolve() / "raw/food101_inference_sample"
    manifest = output / "manifest.jsonl"
    state_path = Path(data_dir) / "metadata/food101_inference_sample.json"
    settings = {"catalog": catalog, "n_images": n_images, "seed": seed}
    if manifest.exists() and not overwrite:
        if not state_path.exists() or read_json(state_path) != settings:
            raise ValueError("Параметры Food-101 сэмпла изменились; установите OVERWRITE_FOOD101 = True.")
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        if all((output / row["image"]).is_file() for row in rows):
            return output, rows
    previous = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()] if manifest.exists() else []
    rows = []
    for index in tqdm(sorted(random.Random(seed).sample(range(25250), n_images)), desc="Food-101: фото без разметки", unit="фото"):
        local = output / "images" / f"{index:06d}.jpg"
        cached = Path(data_dir) / "raw/food101" / catalog["revision"] / "validation" / local.name
        if not local.exists() and cached.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(cached, local)
            except OSError:
                shutil.copy2(cached, local)
        if not local.exists():
            item = food_rows(catalog, "validation", index, 1)[0]
            download(item["row"]["image"]["src"], local)
        with Image.open(local) as image:
            image.verify()
        rows.append({"image": local.relative_to(output).as_posix(), "source": "food101",
                     "source_id": f"validation/{index:06d}"})
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dump_json(state_path, settings)
    kept = {row["image"] for row in rows}
    for row in previous:
        path = (output / row["image"]).resolve()
        if not path.is_relative_to(output):
            raise ValueError("Путь выходит за пределы сэмпла.")
        if row["image"] not in kept:
            path.unlink(missing_ok=True)
    return output, rows
