"""Pinned MyFoodRepo v2.1 import: images and object boxes derived from polygons."""
from __future__ import annotations

import math
import shutil
import os
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from PIL import Image
from tqdm.auto import tqdm

from scripts import dataset_pipeline as dp

DATASET_REF = "rinatkharisov/detectron2ra"
MIRROR_VERSION = 24
PREFIX = "public_training_set_release_2.1"
SOURCE_PAGE = "https://www.aicrowd.com/challenges/food-recognition-benchmark-2022"
MIRROR_PAGE = f"https://www.kaggle.com/datasets/{DATASET_REF}/versions/{MIRROR_VERSION}"
_annotations = {}
_indexes = {}
_archive_indexes = {}
_extracted = {}


def archive_info(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Архив MyFoodRepo не найден: {path}")
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def archive_members(info):
    key = dp.fingerprint(info)
    if key not in _archive_indexes:
        with tarfile.open(info["path"], "r:*") as archive:
            members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name.replace("\\", "/"))
            if (name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts)
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Небезопасный элемент tar: {member.name}")
        annotations = [m for m in members if m.isfile() and PurePosixPath(m.name).name in {"annotations.json", "annotation.json"}]
        if len(annotations) != 1:
            raise ValueError("В архиве должен быть один COCO annotations.json.")
        _archive_indexes[key] = (members, annotations[0])
    return _archive_indexes[key]


def archive_annotations(data_dir, info, split):
    folder = Path(data_dir) / "metadata/myfoodrepo_v2_1_archives" / split
    path, state = folder / "annotations.json", folder / "source.json"
    if not (path.is_file() and state.is_file() and dp.read_json(state) == info):
        _, member = archive_members(info)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.part")
        with tarfile.open(info["path"], "r:*") as archive, archive.extractfile(member) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(path)
        dp.dump_json(state, info)
    return path


def extract_archive(data_dir, info, split):
    data_dir = Path(data_dir).resolve()
    key = (str(data_dir), dp.fingerprint(info), split)
    if key in _extracted:
        return _extracted[key]
    destination = data_dir / "raw/myfoodrepo_v2_1_archives" / split
    if not destination.resolve().is_relative_to(data_dir):
        raise ValueError("Папка распаковки выходит за пределы data/.")
    state_path = data_dir / "metadata/myfoodrepo_v2_1_archives" / split / "extracted.json"
    state = dp.read_json(state_path) if state_path.exists() else {}
    if state.get("archive") == info and state.get("complete"):
        image_root = destination / state["image_root"]
        if image_root.is_dir():
            tqdm.write(f"MyFoodRepo {split}: архив уже распакован.")
            _extracted[key] = image_root
            return image_root
    members, annotation = archive_members(info)
    files = [m for m in members if m.isfile() and "images" in PurePosixPath(m.name).parts
             and PurePosixPath(m.name).suffix.lower() in dp.IMAGE_SUFFIXES]
    for member in files:
        if not (destination / member.name).resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"Путь tar выходит за пределы распаковки: {member.name}")
    same_archive = state.get("archive") == info
    image_root = PurePosixPath(annotation.name).parent / "images"
    dp.dump_json(state_path, {"archive": info, "complete": False, "image_root": str(image_root)})
    with tarfile.open(info["path"], "r:*") as archive:
        for member in tqdm(files, desc=f"MyFoodRepo {split}: распаковка", unit="файл"):
            target = destination / member.name
            if same_archive and target.is_file() and target.stat().st_size == member.size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with archive.extractfile(member) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            temporary.replace(target)
    if not (destination / image_root).is_dir():
        raise ValueError("Рядом с COCO JSON в архиве не найдена папка images/.")
    dp.dump_json(state_path, {"archive": info, "complete": True, "image_root": str(image_root)})
    _extracted[key] = destination / image_root
    return _extracted[key]


def file_url(filename):
    return (f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}/"
            f"{quote(filename, safe='')}?datasetVersionNumber={MIRROR_VERSION}")


def download_file(filename, destination, *, show_progress=False):
    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with dp.request(file_url(filename), stream=True) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Kaggle вернул страницу вместо файла.")
        total = int(response.headers.get("Content-Length", 0)) or None
        with temporary.open("wb") as stream, tqdm(total=total, desc=PurePosixPath(filename).name,
                                                  unit="B", unit_scale=True, disable=not show_progress) as progress:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
                progress.update(len(chunk))
        if total is not None and temporary.stat().st_size != total:
            raise RuntimeError("Файл загружен не полностью; повторите запуск.")
    temporary.replace(destination)
    return destination


def load_annotations(data_dir, offline=False, *, catalog=None, split="train"):
    archive = (catalog or {}).get("archives", {}).get(split)
    key = (str(Path(data_dir).resolve()), split, dp.fingerprint(archive))
    if key not in _annotations:
        dp.configure_https(data_dir)
        if archive is not None:
            path = archive_annotations(data_dir, archive, split)
        else:
            if split != "train":
                raise ValueError("Для авторской validation укажите локальный valid-архив.")
            path = Path(data_dir) / "metadata/myfoodrepo_v2_1/annotations.json"
            if offline and not path.exists():
                raise FileNotFoundError(path)
            download_file(f"{PREFIX}/annotations.json", path, show_progress=True)
        coco = dp.read_json(path)
        counts = tuple(len(coco[k]) for k in ("categories", "images", "annotations"))
        expected = (323, 54392, 100256) if split == "train" else (323, 946, 1708)
        if counts != expected:
            raise ValueError(f"MyFoodRepo v2.1 {split}: ожидалось {expected}; получено {counts}.")
        for name in ("categories", "images", "annotations"):
            if len({item["id"] for item in coco[name]}) != len(coco[name]):
                raise ValueError(f"Повторяющиеся ID в {name}.")
        _annotations[key] = coco
    return _annotations[key]


def polygon_box(polygons, width, height):
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("Нет полигональной сегментации.")
    if any(len(p) % 2 for p in polygons):
        raise ValueError("Нечётное число координат полигона.")
    valid = [p for p in polygons if len(p) >= 6]
    if not valid:
        raise ValueError("Нет полигона с тремя вершинами.")
    xs = [x for p in valid for x in p[::2]]
    ys = [y for p in valid for y in p[1::2]]
    if not all(math.isfinite(v) for v in xs + ys):
        raise ValueError("Некорректные координаты полигона.")
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Полигон выходит за границы изображения.")
    return [(left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height], valid


def index(data_dir, catalog, split="train"):
    key = (str(Path(data_dir).resolve()), dp.fingerprint(catalog), split)
    if key in _indexes:
        return _indexes[key]
    coco = load_annotations(data_dir, catalog=catalog, split=split)
    images = {image["id"]: image for image in coco["images"]}
    categories = set(catalog["category_ids"])
    objects, seen, audit = defaultdict(list), defaultdict(set), []
    for annotation in tqdm(coco["annotations"], desc="MyFoodRepo: нормализация разметки", unit="объект"):
        image = images[annotation["image_id"]]
        category_id = annotation["category_id"]
        if category_id not in categories:
            raise ValueError(f"Неизвестный category_id={category_id}")
        box, polygons = polygon_box(annotation["segmentation"], image["width"], image["height"])
        common = {"image_id": image["id"], "annotation_id": annotation["id"]}
        if len(polygons) != len(annotation["segmentation"]):
            audit.append(common | {"reason": "degenerate_polygon_components"})
        signature = dp.fingerprint([category_id, polygons])
        if signature in seen[image["id"]]:
            audit.append(common | {"reason": "duplicate_object"})
            continue
        seen[image["id"]].add(signature)
        objects[image["id"]].append({"source_class_id": category_id, "bbox": box})
    records = []
    for image in coco["images"]:
        if not objects[image["id"]]:
            raise ValueError(f"Изображение без объектов: {image['id']}")
        records.append({"source": "myfoodrepo", "source_id": f"{split}/{image['id']}", "source_split": split,
                        "group_id": f"myfoodrepo/{image['id']}", "box_origin": "segmentation_polygon",
                        "preserve_source_split": "val" in catalog.get("splits", []),
                        "image_id": image["id"], "file_name": image["file_name"],
                        "width": image["width"], "height": image["height"], "objects": objects[image["id"]]})
    if audit:
        if catalog.get("archives", {}).get(split):
            audit_path = Path(data_dir) / "metadata/myfoodrepo_v2_1_archives" / split / "import_audit.json"
        else:
            audit_path = Path(data_dir) / "metadata/myfoodrepo_v2_1/import_audit.json"
        dp.dump_json(audit_path, audit)
    _indexes[key] = records
    return records


def materialize(data_dir, record, catalog=None):
    filename = PurePosixPath(record["file_name"]).name
    archive = (catalog or {}).get("archives", {}).get(record["source_split"])
    if archive is not None:
        path = extract_archive(data_dir, archive, record["source_split"]) / filename
    else:
        path = Path(data_dir) / "raw/myfoodrepo_v2_1/images" / filename
    old_sample = Path(data_dir) / "raw/myfoodrepo_v2_1_sample/images" / f"{record['image_id']}.jpg"
    if archive is None and not path.exists() and old_sample.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(old_sample, path)
        except OSError:
            shutil.copy2(old_sample, path)
    if archive is None:
        download_file(f"{PREFIX}/images/{filename}", path)
    with Image.open(path) as image:
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"Размеры не совпали с COCO: {filename}")
        image.verify()
    return {k: v for k, v in record.items() if k != "file_name"} | {"image": str(path.resolve())}


def prepare(data_dir, catalog, wanted=None, per_class=2, seed=42, workers=8, valid_limit=None):
    records = list(index(data_dir, catalog))
    if wanted is not None:
        name_to_id = dict(zip(catalog["names"], catalog["category_ids"]))
        wanted_ids = {name_to_id[name] for name in wanted}
        records.sort(key=lambda r: dp.fingerprint([seed, r["source_id"]]))
        selected, counts = [], Counter()
        for record in records:
            ids = {obj["source_class_id"] for obj in record["objects"]}
            if any(counts[category] < per_class for category in ids & wanted_ids):
                selected.append(record)
                counts.update(ids & wanted_ids)
            if all(counts[category] >= per_class for category in wanted_ids):
                break
        records = selected
    if "val" in catalog.get("splits", []):
        validation = list(index(data_dir, catalog, "val"))
        validation.sort(key=lambda r: dp.fingerprint([seed, r["source_id"]]))
        records += validation[:valid_limit] if valid_limit is not None else validation
    for split, info in catalog.get("archives", {}).items():
        extract_archive(data_dir, info, split)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(lambda r: materialize(data_dir, r, catalog), records), total=len(records),
                         desc="MyFoodRepo v2.1: подготовка фото", unit="фото"))
