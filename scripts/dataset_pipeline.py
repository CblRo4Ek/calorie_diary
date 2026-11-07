"""FPB + Food-101 ingestion. Source labels are never silently treated as background.

All downloaded assets, caches, decisions and exports live under the supplied data/.
Food-101 uses a full-image proxy box by the project's explicit baseline decision.
Only the first five YOLO fields survive FPB ingestion; weights are discarded.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import ssl
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
import yaml
from PIL import Image

FPB_REPO = "issai/Food_Portion_Benchmark"
FOOD_REPO = "ethz/food101"
FPB_PREFIX = "FPB_Dataset/RGB"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PROXY_BOX = [0.5, 0.5, 1.0, 1.0]
_sessions = threading.local()
_catalog_cache = {}
_tree_cache = {}
_food_index_cache = {}
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
    # huggingface_hub 1.x uses HTTPX, which reads the standard SSL_CERT_FILE.
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


def catalogs(data_dir, offline=False):
    """Cache pinned catalogs in data/metadata for subsequent runs."""
    configure_https(data_dir)
    key = str(Path(data_dir).resolve())
    if key in _catalog_cache:
        return _catalog_cache[key]
    path = Path(data_dir) / "metadata/sources.lock.json"
    if path.exists():
        _catalog_cache[key] = read_json(path)
        return _catalog_cache[key]
    if offline:
        raise FileNotFoundError(f"Для первого запуска нужен интернет: отсутствует {path}")
    infos = {}
    for source, repo in (("fpb", FPB_REPO), ("food101", FOOD_REPO)):
        infos[source] = request(f"https://huggingface.co/api/datasets/{repo}").json()
    fpb_sha = infos["fpb"]["sha"]
    fpb_config = yaml.safe_load(request(
        f"https://huggingface.co/datasets/{FPB_REPO}/raw/{fpb_sha}/{FPB_PREFIX}/data.yaml"
    ).text)
    fpb_names = fpb_config["names"]
    if isinstance(fpb_names, dict):
        fpb_names = [fpb_names[i] for i in range(len(fpb_names))]
    features = infos["food101"]["cardData"]["dataset_info"]["features"]
    labels = next(item for item in features if item["name"] == "label")["dtype"]["class_label"]["names"]
    food_names = labels if isinstance(labels, list) else [labels[str(i)] for i in range(len(labels))]
    if len(fpb_names) != 138 or len(food_names) != 101:
        raise ValueError("Изменился опубликованный словарь классов. Проверьте новый релиз.")
    result = {
        "fpb": {"repo": FPB_REPO, "revision": fpb_sha, "names": fpb_names},
        "food101": {"repo": FOOD_REPO, "revision": infos["food101"]["sha"], "names": food_names,
                    "files": [x["rfilename"] for x in infos["food101"]["siblings"]
                              if x["rfilename"].endswith(".parquet")]},
    }
    _catalog_cache[key] = result
    dump_json(path, result)
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
    for offset in range(0, min(len(records), max_scan), 64):
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
        if all(counts[name] >= per_class for name in wanted):
            break
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(lambda args: fpb_materialize(data_dir, catalog, *args), selected))
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


class HTTPRangeReader(io.RawIOBase):
    """Read only requested Parquet column chunks; never buffer a whole image shard."""
    def __init__(self, url):
        super().__init__()
        self.url, self.position = url, 0
        with request(url, headers={"Range": "bytes=0-0"}, stream=True) as response:
            if response.status_code != 206 or "Content-Range" not in response.headers:
                raise ValueError("Сервер не поддерживает Range. Используйте полную загрузку Parquet.")
            self.size = int(response.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if value < 0:
            raise ValueError("Отрицательная позиция файла")
        self.position = value
        return value

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0:
            return b""
        start = self.position
        with request(self.url, headers={"Range": f"bytes={start}-{start+size-1}"}, stream=True) as response:
            if response.status_code != 206:
                raise ValueError("Сервер проигнорировал Range; полная загрузка не выполнялась.")
            content = response.content
        if len(content) != size:
            raise IOError("Неполный ответ Range")
        self.position += len(content)
        return content


def food_preview_blocks(data_dir, catalog, split="train", workers=8):
    """Cache the exact Parquet class index and per-shard labels on disk."""
    import pyarrow.parquet as pq
    key = (str(Path(data_dir).resolve()), catalog["repo"], catalog["revision"], split)
    if key in _food_index_cache:
        return _food_index_cache[key]
    path = Path(data_dir) / "metadata" / f'food101_row_index_{catalog["revision"]}_{split}.json'
    if path.exists():
        _food_index_cache[key] = read_json(path)
        return _food_index_cache[key]
    def inspect(remote):
        cache = path.parent / f'food101_labels_{catalog["revision"]}_{Path(remote).stem}.json'
        if cache.exists():
            return read_json(cache)
        url = f'https://huggingface.co/datasets/{catalog["repo"]}/resolve/{catalog["revision"]}/{remote}'
        with HTTPRangeReader(url) as stream:
            parquet = pq.ParquetFile(stream)
            values = []
            for group_id in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(group_id)
                column = next(group.column(i) for i in range(group.num_columns)
                              if group.column(i).path_in_schema == "label")
                stats = column.statistics
                if stats and stats.has_min_max and stats.min == stats.max and stats.null_count == 0:
                    values.extend([int(stats.min)] * group.num_rows)
                else:
                    values.extend(parquet.read_row_group(group_id, columns=["label"])["label"].to_pylist())
        print(f"Indexed labels: {Path(remote).name} ({len(values)} rows)", flush=True)
        dump_json(cache, values)
        return values
    files = sorted(remote for remote in catalog["files"] if remote.startswith(f"data/{split}-"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shards = list(pool.map(inspect, files))
    index = defaultdict(list)
    offset = 0
    for labels in shards:
        for label in labels:
            index[str(label)].append(offset)
            offset += 1
    expected_count = 750 if split == "train" else 250
    if len(index) != len(catalog["names"]) or any(len(rows) != expected_count for rows in index.values()):
        raise ValueError("Food-101: изменилась полнота исходной разметки. Проверьте новый релиз.")
    _food_index_cache[key] = dict(index)
    dump_json(path, _food_index_cache[key])
    return _food_index_cache[key]


def prepare_food_preview(data_dir, catalog, wanted, per_class=4, split="train", workers=8):
    if not 1 <= per_class <= 100:
        raise ValueError("Preview per_class должен быть от 1 до 100.")
    blocks = food_preview_blocks(data_dir, catalog, split, workers)
    def prepare(name):
        class_id = catalog["names"].index(name)
        indices = blocks[str(class_id)][:per_class]
        local_paths = [Path(data_dir) / "raw/food101" / catalog["revision"] / split / f"{i:06d}.jpg" for i in indices]
        if all(path.exists() for path in local_paths):
            return [food_record(path, split, i, class_id) for path, i in zip(local_paths, indices)]
        if indices == list(range(indices[0], indices[0] + len(indices))):
            rows = food_rows(catalog, split, indices[0], len(indices))
        else:
            rows = [food_rows(catalog, split, index, 1)[0] for index in indices]
        result = []
        for item in rows:
            row_idx, row = item["row_idx"], item["row"]
            if row["label"] != class_id:
                raise ValueError("Изменился индекс классов Food-101 preview.")
            path = Path(data_dir) / "raw/food101" / catalog["revision"] / split / f"{row_idx:06d}.jpg"
            download(row["image"]["src"], path)
            result.append(food_record(path, split, row_idx, class_id))
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [record for group in pool.map(prepare, sorted(set(wanted))) for record in group]


def prepare_food_slice(data_dir, catalog, split="validation", offset=0, length=4):
    """Small original test slice, without a second class-block discovery pass."""
    records = []
    for item in food_rows(catalog, split, offset, length):
        row_idx, row = item["row_idx"], item["row"]
        path = Path(data_dir) / "raw/food101" / catalog["revision"] / split / f"{row_idx:06d}.jpg"
        download(row["image"]["src"], path)
        records.append(food_record(path, split, row_idx, row["label"]))
    return records


def preview_samples(data_dir, cats, per_class=3, max_scan=1600, workers=8):
    """Shared, real-image preview for both notebooks; full preparation is separate."""
    frame = review_table(cats)
    config = {"catalogs": cats, "per_class": per_class, "max_scan": max_scan}
    key = (str(Path(data_dir).resolve()), fingerprint(config))
    cache = Path(data_dir) / "metadata" / f"preview_samples_{fingerprint(config)[:12]}.json"
    if key not in _preview_cache and cache.exists():
        _preview_cache[key] = read_json(cache)
    if key in _preview_cache:
        payload = _preview_cache[key]
        if all(Path(r["image"]).is_file() for r in payload["records"]):
            return payload["records"], payload["coverage"]
    fpb_records, coverage = prepare_fpb_preview(
        data_dir, cats["fpb"], frame["fpb_class"], per_class=per_class,
        max_scan=max_scan, workers=workers,
    )
    wanted_food = sorted(set(frame["food101_class"]) | {"apple_pie", "sushi", "waffles"})
    food_records = prepare_food_preview(data_dir, cats["food101"], wanted_food, per_class=per_class+1, workers=workers)
    # Native FPB validation and unlabelled test retain their identities.
    for split, limit in (("val", 6), ("test", 2)):
        for record in fpb_index(data_dir, cats["fpb"], (split,))[:limit]:
            fpb_records.append(fpb_materialize(data_dir, cats["fpb"], record,
                                             fpb_objects(data_dir, cats["fpb"], record)))
    food_records += prepare_food_slice(data_dir, cats["food101"])
    payload = {"records": fpb_records + food_records, "coverage": coverage}
    _preview_cache[key] = payload
    dump_json(cache, payload)
    return payload["records"], coverage


def food_record(path, split, row_idx, class_id):
    uid = f"{split}/{row_idx:06d}"
    return {"source": "food101", "source_id": uid, "source_split": split,
            "group_id": f"food101/{uid}", "box_origin": "full_image_proxy",
            "image": str(Path(path).resolve()),
            "objects": [{"source_class_id": class_id, "bbox": PROXY_BOX.copy()}]}


def prepare_full(data_dir, cats, workers=8):
    """Download FPB images/clean labels and Food-101 Parquet into data/, resumably."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    fpb_records = fpb_index(data_dir, cats["fpb"])
    def prepare(record):
        objects = fpb_objects(data_dir, cats["fpb"], record)
        return fpb_materialize(data_dir, cats["fpb"], record, objects)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(prepare, fpb_records))
    for native_split in ("train", "validation"):
        offset = 0
        for remote in sorted(x for x in cats["food101"]["files"] if x.startswith(f"data/{native_split}-")):
            local = hf_hub_download(
                repo_id=FOOD_REPO, repo_type="dataset", revision=cats["food101"]["revision"],
                filename=remote, cache_dir=str(Path(data_dir) / "cache/huggingface"),
            )
            for batch in pq.ParquetFile(local).iter_batches(batch_size=128, columns=["image", "label"]):
                for row in batch.to_pylist():
                    class_id = int(row["label"])
                    if not 0 <= class_id < len(cats["food101"]["names"]):
                        raise ValueError("Неизвестный класс Food-101")
                    path = Path(data_dir) / "raw/food101" / cats["food101"]["revision"] / native_split / f"{offset:06d}.jpg"
                    if not path.exists():
                        image_bytes = row["image"]["bytes"]
                        if image_bytes is None:
                            raise ValueError("Parquet не содержит байты изображения Food-101.")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(image_bytes)
                    result.append(food_record(path, native_split, offset, class_id))
                    offset += 1
        expected = 75750 if native_split == "train" else 25250
        if offset != expected:
            raise ValueError(f"Неполная загрузка Food-101 {native_split}: {offset}, ожидалось {expected}")
    return result


# These are hypotheses for visual review, never automatic synonym merges.
SYNONYM_CANDIDATES = [
    ("icecream", "ice_cream", "likely_alias", "Разное написание мороженого"),
    ("cupcake", "cup_cakes", "likely_alias", "Единственное/множественное число"),
    ("pahlava", "baklava", "likely_alias", "Варианты названия пахлавы"),
    ("burger", "hamburger", "needs_review", "Бургер может быть шире гамбургера"),
    ("bliny", "pancakes", "related", "Блины и панкейки могут различаться"),
    ("salmon", "grilled_salmon", "related", "Второй класс задаёт способ приготовления"),
    ("fried_dumplings", "dumplings", "related", "Общий класс и жареный подтип"),
    ("samsa", "samosa", "related", "Разные блюда; созвучие не означает синонимию"),
]


def review_table(cats):
    import pandas as pd
    rows = [{"fpb_class": name, "food101_class": name, "kind": "exact_name",
             "decision": "merge", "reviewed": True, "note": "Одинаковое имя; визуальная проверка рекомендуется"}
            for name in sorted(set(cats["fpb"]["names"]) & set(cats["food101"]["names"]))]
    rows += [{"fpb_class": a, "food101_class": b, "kind": kind, "decision": "keep_separate",
              "reviewed": False, "note": note} for a, b, kind, note in SYNONYM_CANDIDATES
             if a in cats["fpb"]["names"] and b in cats["food101"]["names"]]
    return pd.DataFrame(rows)


def save_decisions(frame, path, cats):
    validate_decisions(frame, cats)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def validate_decisions(frame, cats):
    if frame["food101_class"].duplicated().any():
        raise ValueError("Один класс Food-101 указан в решениях несколько раз.")
    for row in frame.to_dict("records"):
        if row["fpb_class"] not in cats["fpb"]["names"] or row["food101_class"] not in cats["food101"]["names"]:
            raise ValueError("Неизвестный класс в решениях.")
        if row["decision"] not in {"merge", "keep_separate"}:
            raise ValueError("Решение должно быть merge или keep_separate.")
        if row["decision"] == "merge" and str(row["reviewed"]).lower() != "true":
            raise ValueError(f'Сначала подтвердите reviewed=True для {row["food101_class"]}')


def class_mapping(cats, decisions):
    validate_decisions(decisions, cats)
    names = list(cats["fpb"]["names"])
    targets = {row["food101_class"]: row for row in decisions.to_dict("records")}
    mapping = {("fpb", idx): idx for idx in range(len(names))}
    rows = [{"source": "fpb", "source_class_id": idx, "source_class": name,
             "class_id": idx, "class_name": name, "rule": "fpb_base"} for idx, name in enumerate(names)]
    for idx, name in enumerate(cats["food101"]["names"]):
        decision = targets.get(name)
        if decision and decision["decision"] == "merge":
            target = decision["fpb_class"]
            rule = "exact_name" if name == target else "reviewed_alias"
        else:
            # A deliberately rejected same-name match must remain distinguishable.
            target = f"food101__{name}" if name in names else name
            rule = "food101_separate"
        if target not in names:
            names.append(target)
        class_id = names.index(target)
        mapping[("food101", idx)] = class_id
        rows.append({"source": "food101", "source_class_id": idx, "source_class": name,
                     "class_id": class_id, "class_name": target, "rule": rule})
    return names, mapping, rows


def assign_splits(records, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction должен находиться между 0 и 1.")
    result = [dict(record) for record in records]
    food_train = defaultdict(list)
    for record in result:
        if record["source"] == "fpb":
            record["split"] = "fpb_test_unlabeled" if record["source_split"] == "test" else record["source_split"]
        elif record["source_split"] == "validation":
            record["split"] = "test"  # HF validation is the original Food-101 test.
        elif record["source_split"] == "train":
            food_train[record["objects"][0]["source_class_id"]].append(record)
        else:
            raise ValueError(f'Неизвестный split: {record["source_split"]}')
    for group in food_train.values():
        ordered = sorted(group, key=lambda r: fingerprint([seed, r["group_id"]]))
        val_count = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        for i, record in enumerate(ordered):
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
    """Protect holdouts. Conflicting duplicate annotations are quarantined."""
    rank = {"train": 0, "val": 1, "test": 2, "fpb_test_unlabeled": 3}
    group_rank = defaultdict(int)
    for record in records:
        group_rank[record["group_id"]] = max(group_rank[record["group_id"]], rank[record["split"]])
    candidates, audit = [], []
    for record in records:
        if rank[record["split"]] < group_rank[record["group_id"]]:
            audit.append({"source_id": record["source_id"], "source": record["source"], "reason": "group_in_holdout"})
        else:
            candidates.append(record)
    by_hash = defaultdict(list)
    for record in candidates:
        by_hash[record["pixel_sha256"]].append(record)
    kept = []
    for group in by_hash.values():
        signatures = {fingerprint(record["objects"]) for record in group if record["objects"] is not None}
        if len(signatures) > 1:
            audit.extend({"source": r["source"], "source_id": r["source_id"], "reason": "conflicting_duplicate"} for r in group)
            continue
        ordered = sorted(group, key=lambda r: (-rank[r["split"]], r["source"], r["source_id"]))
        kept.append(ordered[0])
        audit.extend({"source": r["source"], "source_id": r["source_id"], "reason": "pixel_duplicate"} for r in ordered[1:])
    return sorted(kept, key=lambda r: (r["source"], r["source_id"])), audit


def build_dataset(data_dir, cats, records, decisions, mode="preview", val_fraction=0.1, seed=42):
    """Replace the single current export only after the new export validates."""
    data_dir = Path(data_dir).resolve()
    output = data_dir / "processed/fpb_food101"
    cache = data_dir / "cache"
    # Resolve before creating/moving/removing anything, including Windows junctions.
    for target in (output, cache):
        if not target.resolve().is_relative_to(data_dir):
            raise ValueError(f"Путь выходит за пределы data/: {target}")
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fpb_food101_build_", dir=cache) as temporary:
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
    # Remove generated version folders from the previous layout after success.
    for previous in output.parent.glob("fpb_food101_*"):
        if previous.is_dir():
            if not previous.resolve().is_relative_to(output.parent.resolve()):
                raise ValueError(f"Путь старой сборки выходит за пределы processed/: {previous}")
            shutil.rmtree(previous)
    return output, merged, summary


def _write_dataset(cats, records, decisions, output, final_output, *, mode, val_fraction, seed):
    """Write and validate a temporary export; YAML paths target its final location."""
    import pandas as pd
    names, mapping, mapping_rows = class_mapping(cats, decisions)
    records = assign_splits(records, val_fraction, seed)
    seen = set()
    for record in records:
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
            if record["box_origin"] == "full_image_proxy" and any(obj["bbox"] != PROXY_BOX for obj in record["objects"]):
                raise ValueError("Food-101 должен иметь рамку всего изображения.")
    kept, audit = deduplicate(records)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for record in kept:
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
    summary = {"mode": mode, "classes": len(names), "images": len(manifest), "excluded": len(audit),
               "by_source_split": dict(Counter(f'{r["source"]}/{r["split"]}' for r in manifest)),
               "unlabeled_fpb_test": sum(r["label"] is None for r in manifest),
               "portion_grouping": "Only filename variants and exact pixels; physical plate IDs are not provided."}
    summary["validation"] = validate_export(output)
    return output, pd.DataFrame(manifest), summary


def validate_export(output):
    output = Path(output)
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    splits_by_hash, splits_by_group = defaultdict(set), defaultdict(set)
    for record in records:
        image_path = output / record["image"]
        with Image.open(image_path) as image:
            image.verify()
        splits_by_hash[record["pixel_sha256"]].add(record["split"])
        splits_by_group[record["group_id"]].add(record["split"])
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
        if record["source"] == "food101" and any(obj["bbox"] != PROXY_BOX for obj in objects):
            raise AssertionError("Неполная рамка Food-101.")
    if any(len(x) != 1 for x in splits_by_hash.values()) or any(len(x) != 1 for x in splits_by_group.values()):
        raise AssertionError("Обнаружено пересечение splits по пикселям/группе.")
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
                name = cats[record["source"]]["names"][obj["source_class_id"]]
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


def show_class_pair(records, cats, fpb_class, food101_class, samples=3):
    """Compare target-object crops in FPB to full Food-101 images, training only."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, samples, figsize=(4*samples, 7), squeeze=False)
    for row, (source, name) in enumerate((("fpb", fpb_class), ("food101", food101_class))):
        source_id = cats[source]["names"].index(name)
        candidates = [r for r in records if r["source"] == source and r["source_split"] == "train"
                      and any(o["source_class_id"] == source_id for o in r["objects"] or [])]
        for col in range(samples):
            axis = axes[row, col]
            if col < len(candidates):
                record = candidates[col]
                with Image.open(record["image"]) as image:
                    image = image.convert("RGB")
                    if source == "fpb":
                        obj = next(o for o in record["objects"] if o["source_class_id"] == source_id)
                        x,y,w,h = obj["bbox"]
                        width,height = image.size
                        image = image.crop((max(0, math.floor((x-w/2)*width)), max(0, math.floor((y-h/2)*height)),
                                            min(width, math.ceil((x+w/2)*width)), min(height, math.ceil((y+h/2)*height))))
                    axis.imshow(image)
                axis.set_title(f'{source}: {name}\n{"native bbox crop" if source == "fpb" else "full image proxy"}', fontsize=10)
            else:
                axis.text(0.5,0.5,"No sample in preview\nIncrease max_scan / use full data",ha="center",va="center")
            axis.axis("off")
    fig.suptitle(f"{fpb_class}  /  {food101_class}")
    fig.tight_layout()
    plt.show()
    return fig
