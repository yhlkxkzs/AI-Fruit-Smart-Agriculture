#!/usr/bin/env python3
"""Build fruit_classification_multistate manifests (species + state, dual-head training)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data" / "fruit_classification_multistate"
DB_ROOT = Path("/home/yuhanlin/Database/datasets/database")
TEMP_ROOT = Path("/home/yuhanlin/Database/datasets/temp")
SPECIES_MAP_FILE = TEMP_ROOT / "species_mapping.json"

BASE_SPECIES = [
    "apple",
    "cherry",
    "grape",
    "lemon",
    "mango",
    "orange",
    "pear",
    "pistachio",
    "strawberry",
    "tomato",
    "banana",
    "coconut",
    "guava",
    "papaya",
    "pomegranate",
    "litchi",
    "avocado",
    "bean",
    "broccoli",
    "chestnut",
    "cucumber",
    "eggplant",
    "hogplum",
    "jackfruit",
    "soybean",
    "other",
]
STATES = ["healthy", "immature", "mature", "aged", "diseased", "pest", "unknown"]

SPECIES_ALIASES = {
    "apple": "apple",
    "apples": "apple",
    "cherry": "cherry",
    "cherries": "cherry",
    "grape": "grape",
    "grapes": "grape",
    "lemon": "lemon",
    "lemons": "lemon",
    "mango": "mango",
    "mangoes": "mango",
    "orange": "orange",
    "oranges": "orange",
    "pear": "pear",
    "pears": "pear",
    "pistachio": "pistachio",
    "strawberry": "strawberry",
    "strawberries": "strawberry",
    "tomato": "tomato",
    "tomatoes": "tomato",
    "banana": "banana",
    "bananas": "banana",
    "coconut": "coconut",
    "coconuts": "coconut",
    "guava": "guava",
    "guavas": "guava",
    "papaya": "papaya",
    "papayas": "papaya",
    "pomegranate": "pomegranate",
    "pomegranates": "pomegranate",
    "litchi": "litchi",
    "litchis": "litchi",
    "avocado": "avocado",
    "avocados": "avocado",
    "bean": "bean",
    "beans": "bean",
    "broccoli": "broccoli",
    "broccolis": "broccoli",
    "chestnut": "chestnut",
    "chestnuts": "chestnut",
    "cucumber": "cucumber",
    "cucumbers": "cucumber",
    "eggplant": "eggplant",
    "eggplants": "eggplant",
    "hogplum": "hogplum",
    "hogpulm": "hogplum",
    "jackfruit": "jackfruit",
    "jackfruits": "jackfruit",
    "soybean": "soybean",
    "soybeans": "soybean",
}

DATASET_DEFAULT_SPECIES: dict[str, str] = {}


def load_temp_species_mapping() -> None:
    global SPECIES, SPECIES_ALIASES, STATE_ALIASES, DATASET_DEFAULT_SPECIES
    SPECIES = list(BASE_SPECIES)
    if not SPECIES_MAP_FILE.is_file():
        return
    data = json.loads(SPECIES_MAP_FILE.read_text(encoding="utf-8"))
    SPECIES = data.get("species") or SPECIES
    SPECIES_ALIASES.update(data.get("species_aliases") or {})
    STATE_ALIASES.update(data.get("state_aliases_extra") or {})
    DATASET_DEFAULT_SPECIES.update(data.get("dataset_default_species") or {})


SPECIES = list(BASE_SPECIES)

STATE_ALIASES = {
    "fresh": "healthy",
    "healthy": "healthy",
    "raw": "immature",
    "unripe": "immature",
    "green": "immature",
    "b_green": "immature",
    "l_green": "immature",
    "freshunripe": "immature",
    "white": "immature",
    "early-turning": "immature",
    "ripe": "mature",
    "red": "mature",
    "freshripe": "mature",
    "b_fully_ripened": "mature",
    "l_fully_ripened": "mature",
    "turning": "mature",
    "b_half_ripened": "mature",
    "l_half_ripened": "mature",
    "late-turning": "mature",
    "rotten": "aged",
    "overripe": "aged",
    "formalin_mixed": "aged",
    "formalin-mixed": "aged",
    "scab": "diseased",
    "anomalous": "diseased",
    "illness": "diseased",
    "mould": "diseased",
    "gangrene": "diseased",
    "blemish": "diseased",
    "greening": "diseased",
    "occluded": "unknown",
    "rejected": "diseased",
    "angular_leaf_spot": "diseased",
    "bean_rust": "diseased",
    "anthracnose": "diseased",
    "bacterial_wilt": "diseased",
    "belly_rot": "diseased",
    "downy_mildew": "diseased",
    "gummy_stem_blight": "diseased",
    "pythium_fruit_rot": "diseased",
    "fresh_cucumber": "healthy",
    "fresh_leaf": "healthy",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_ROOT = Path("/home/yuhanlin")


def manifest_path(path: Path) -> str:
    resolved = str(path.resolve())
    if resolved.startswith("/data1/yuhanlin"):
        return str(MANIFEST_ROOT / Path(resolved).relative_to("/data1/yuhanlin"))
    return resolved


def find_image(stem: str, folder: Path) -> Path | None:
    for ext in IMG_EXT:
        for variant in (ext, ext.upper()):
            p = folder / f"{stem}{variant}"
            if p.is_file():
                return p
    return None


def norm_species(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return SPECIES_ALIASES.get(key, "other")


def norm_state(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return STATE_ALIASES.get(key, "unknown")


def iter_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out.append(p)
    return out


def add_record(
    records: list[dict],
    path: Path,
    species: str,
    state: str,
    source: str,
) -> None:
    if not path.is_file():
        return
    sp = norm_species(species)
    st = norm_state(state)
    if sp not in SPECIES or st not in STATES:
        return
    records.append(
        {
            "path": manifest_path(path),
            "species": sp,
            "state": st,
            "source": source,
        }
    )


def collect_flat_images(
    records: list[dict],
    images_root: Path,
    source: str,
    species: str,
    state: str = "unknown",
) -> int:
    if not images_root.is_dir():
        return 0
    n = 0
    for img in iter_images(images_root):
        add_record(records, img, species, state, source)
        n += 1
    return n


def collect_fruitvision(records: list[dict]) -> int:
    """Use per-fruit folders Apple/Banana/.../Fresh|Rotten|Formalin-mixed only."""
    root = DB_ROOT / (
        "FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection"
    )
    if not root.is_dir():
        return 0
    skip = {"fruits", "annotations", "Fruits Original", "data", "docs", "scripts"}
    n = 0
    for fruit_dir in root.iterdir():
        if not fruit_dir.is_dir() or fruit_dir.name in skip:
            continue
        sp = fruit_dir.name
        for state_dir in fruit_dir.iterdir():
            if not state_dir.is_dir():
                continue
            st = state_dir.name.replace(" ", "_").replace("-", "_")
            for img in iter_images(state_dir):
                add_record(records, img, sp, st, "fruitvision")
                n += 1
    return n


def collect_rawripe(records: list[dict]) -> int:
    root = DB_ROOT / "RawRipe"
    if not root.is_dir():
        return 0
    n = 0
    for fruit in root.iterdir():
        if not fruit.is_dir() or fruit.name in ("annotations", "docs", "data"):
            continue
        for state_dir in fruit.iterdir():
            if not state_dir.is_dir():
                continue
            img_dir = state_dir / "images"
            if not img_dir.is_dir():
                continue
            for img in iter_images(img_dir):
                add_record(records, img, fruit.name, state_dir.name, "rawripe")
                n += 1
    return n


def collect_papaya_subdirs(
    records: list[dict],
    root: Path,
    source: str,
    species: str | None = None,
) -> int:
    """Papaya-style: {group}/{subcategory}/images/ ; subcategory name = state."""
    if not root.is_dir():
        return 0
    skip = {
        "labelmap.json",
        "annotations",
        "data",
        "docs",
        "scripts",
        "sets",
        "csv",
        "json",
        "images",
        "segmentations",
        "LICENSE",
    }
    n = 0
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name in skip:
            continue
        images = sub / "images"
        if not images.is_dir():
            images = sub / "color" / "images"
        if not images.is_dir():
            continue
        st = sub.name
        sp = species if species else sub.name
        for img in iter_images(images):
            add_record(records, img, sp, st, source)
            n += 1
    return n


def collect_banana_ripening(records: list[dict]) -> int:
    root = DB_ROOT / "Banana_Ripening_Process" / "banana"
    return collect_papaya_subdirs(records, root, "banana_ripening", "banana")


def collect_tomato_ripeness(records: list[dict]) -> int:
    root = DB_ROOT / "tomato_ripeness_detection"
    n = collect_papaya_subdirs(records, root / "tomatoes", "tomato_ripeness", "tomato")
    if n:
        return n
    return collect_papaya_subdirs(records, root, "tomato_ripeness", "tomato")


def collect_riseholme(records: list[dict]) -> int:
    root = DB_ROOT / "riseholme_strawberry_classification_2021"
    return collect_papaya_subdirs(records, root / "strawberries", "riseholme_strawberry", "strawberry")


def collect_tomato_plant(records: list[dict]) -> int:
    root = DB_ROOT / "tomato_plant" / "tomatoes"
    images = root / "images"
    if not images.is_dir():
        return 0
    n = 0
    json_dir = root / "json"
    if json_dir.is_dir():
        for jf in json_dir.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            anns = data.get("annotations") or []
            if not anns:
                continue
            cid = Counter(a.get("category_id", 0) for a in anns).most_common(1)[0][0]
            st = "green" if cid == 1 else "red" if cid == 2 else "unknown"
            stem = jf.stem
            img = find_image(stem, images)
            if img:
                add_record(records, img, "tomato", st, "tomato_plant")
                n += 1
    else:
        for img in iter_images(images):
            add_record(records, img, "tomato", "unknown", "tomato_plant")
            n += 1
    return n


def collect_lemon_qc(records: list[dict]) -> int:
    root = DB_ROOT / "lemon-datase" / "lemons"
    if not root.is_dir():
        return 0
    n = collect_papaya_subdirs(records, root, "lemon_qc", "lemon")
    images = root / "images"
    if images.is_dir():
        for img in iter_images(images):
            add_record(records, img, "lemon", "unknown", "lemon_qc")
            n += 1
    return n


def collect_apple_scab(records: list[dict]) -> int:
    root = DB_ROOT / "AppleScabFDs" / "apples"
    if not root.is_dir():
        root = DB_ROOT / "AppleScabFDs"
    return collect_papaya_subdirs(records, root, "apple_scab", "apple")


def collect_afsa_species(records: list[dict]) -> int:
    root = REPO_ROOT / "data" / "fruit_classification"
    n = 0
    for split in ("train", "val", "test"):
        sp_root = root / split
        if not sp_root.is_dir():
            continue
        for sp_dir in sp_root.iterdir():
            if not sp_dir.is_dir():
                continue
            for img in iter_images(sp_dir):
                add_record(records, img, sp_dir.name, "unknown", "afsa_species_only")
                n += 1
    return n


def collect_rangeland_avocado(records: list[dict]) -> int:
    root = DB_ROOT / "rangeland_weeds_australia" / "avocados" / "images"
    return collect_flat_images(records, root, "rangeland_avocado", "avocado")


def collect_bean_disease(records: list[dict]) -> int:
    root = DB_ROOT / "bean_disease_uganda" / "beans"
    return collect_papaya_subdirs(records, root, "bean_disease_uganda", "bean")


def collect_broccoli_detection(records: list[dict]) -> int:
    root = DB_ROOT / "ghai_broccoli_detection" / "broccolis" / "images"
    return collect_flat_images(records, root, "ghai_broccoli_detection", "broccoli")


def collect_chestnut(records: list[dict]) -> int:
    root = DB_ROOT / "chestnut" / "chestnuts"
    return collect_papaya_subdirs(records, root, "chestnut", "chestnut")


def collect_cucumber_disease(records: list[dict]) -> int:
    root = DB_ROOT / "cucumber_disease_classification" / "cucumbers"
    return collect_papaya_subdirs(records, root, "cucumber_disease", "cucumber")


def collect_eggplant(records: list[dict]) -> int:
    root = DB_ROOT / "eggplant_preprocessed_2026" / "eggplants"
    return collect_papaya_subdirs(records, root, "eggplant_preprocessed", "eggplant")


def collect_fruits_od_hogplum_jackfruit(records: list[dict]) -> int:
    fruits_root = DB_ROOT / "Fruits Images Dataset_Object Detection" / "fruits"
    n = 0
    for species in ("hogplum", "jackfruit"):
        images = fruits_root / species / "color" / "images"
        n += collect_flat_images(records, images, "fruits_od_detection", species)
    return n


def collect_vegann_soybean(records: list[dict]) -> int:
    root = DB_ROOT / "vegann_multicrop_presence_segmentation" / "soybeans" / "images"
    return collect_flat_images(records, root, "vegann_soybean", "soybean")


def collect_temp_datasets(records: list[dict]) -> int:
    """Standardized PV datasets under Database/datasets/temp/."""
    if not TEMP_ROOT.is_dir():
        return 0
    skip = {"scripts", "__pycache__"}
    n = 0
    for ds in sorted(TEMP_ROOT.iterdir()):
        if not ds.is_dir() or ds.name in skip:
            continue
        if not (ds / ".standardized.json").is_file():
            continue
        mains = [
            p
            for p in ds.iterdir()
            if p.is_dir() and p.name not in {"data", "annotations", "scripts"}
        ]
        if not mains:
            continue
        main = mains[0]
        source = f"temp/{ds.name}"

        if ds.name == "Fruits-262" or main.name.startswith("fruits_262"):
            for cls_dir in sorted(main.iterdir()):
                if not cls_dir.is_dir():
                    continue
                images = cls_dir / "images"
                if not images.is_dir():
                    continue
                for img in iter_images(images):
                    add_record(records, img, cls_dir.name, "unknown", source)
                    n += 1
            continue

        default_sp = DATASET_DEFAULT_SPECIES.get(ds.name, ds.name)
        for cls_dir in sorted(main.iterdir()):
            if not cls_dir.is_dir():
                continue
            images = cls_dir / "images"
            if not images.is_dir():
                continue
            cls_sp = norm_species(cls_dir.name)
            sp = norm_species(default_sp)
            if sp == "other" and cls_sp != "other":
                sp = cls_sp
            if cls_sp == sp or cls_sp == "other":
                st = cls_dir.name
            else:
                st = cls_dir.name
            for img in iter_images(images):
                add_record(records, img, sp, st, source)
                n += 1
    return n


def dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = r["path"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_records(records: list[dict], seed: int) -> None:
    rng = random.Random(seed)
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_key[(r["species"], r["state"])].append(r)
    for group in by_key.values():
        rng.shuffle(group)
    train, val, test = [], [], []
    for group in by_key.values():
        n = len(group)
        if n == 0:
            continue
        n_train = max(1, int(n * 0.7)) if n >= 3 else max(0, n - 2)
        n_val = max(1, int(n * 0.15)) if n >= 3 else (1 if n - n_train > 1 else 0)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = max(0, n - n_train)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    for r, sp in ((train, "train"), (val, "val"), (test, "test")):
        for x in r:
            x["split"] = sp


def write_labelmaps(out: Path) -> None:
    species = [
        {"object_id": i, "label_id": i, "object_name": n}
        for i, n in enumerate(SPECIES)
    ]
    states = [
        {"object_id": i, "label_id": i, "object_name": n}
        for i, n in enumerate(STATES)
    ]
    (out / "labelmap_species.json").write_text(
        json.dumps(species, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "labelmap_state.json").write_text(
        json.dumps(states, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_manifests(out: Path, records: list[dict]) -> None:
    manifest = out / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)
    fields = ["path", "species", "state", "source", "split"]
    for name, subset in (
        ("all", records),
        ("train", [r for r in records if r.get("split") == "train"]),
        ("val", [r for r in records if r.get("split") == "val"]),
        ("test", [r for r in records if r.get("split") == "test"]),
    ):
        with open(manifest / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)


def copy_images(out: Path, records: list[dict]) -> None:
    img_root = out / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    for r in records:
        src = Path(r["path"])
        h = hashlib.md5(r["path"].encode()).hexdigest()[:8]
        name = f"{r['species']}_{r['state']}_{r['source']}_{h}{src.suffix.lower()}"
        dst = img_root / r["split"] / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Build fruit_classification_multistate manifests.")
    p.add_argument("--out", type=Path, default=OUT_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_true")
    args = p.parse_args()

    load_temp_species_mapping()

    records: list[dict] = []
    collectors = [
        ("fruitvision", collect_fruitvision),
        ("rawripe", collect_rawripe),
        ("banana_ripening", collect_banana_ripening),
        ("tomato_ripeness", collect_tomato_ripeness),
        ("riseholme_strawberry", collect_riseholme),
        ("tomato_plant", collect_tomato_plant),
        ("lemon_qc", collect_lemon_qc),
        ("apple_scab", collect_apple_scab),
        ("afsa_species_only", collect_afsa_species),
        ("rangeland_avocado", collect_rangeland_avocado),
        ("bean_disease_uganda", collect_bean_disease),
        ("ghai_broccoli_detection", collect_broccoli_detection),
        ("chestnut", collect_chestnut),
        ("cucumber_disease", collect_cucumber_disease),
        ("eggplant_preprocessed", collect_eggplant),
        ("fruits_od_detection", collect_fruits_od_hogplum_jackfruit),
        ("vegann_soybean", collect_vegann_soybean),
        ("temp_datasets", collect_temp_datasets),
    ]
    source_counts: dict[str, int] = {}
    for name, fn in collectors:
        c = fn(records)
        source_counts[name] = c
        print(f"  {name}: {c}")

    records = dedupe_records(records)
    split_records(records, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    write_labelmaps(args.out)
    write_manifests(args.out, records)

    stats = {
        "total": len(records),
        "by_source": source_counts,
        "by_species": dict(Counter(r["species"] for r in records)),
        "by_state": dict(Counter(r["state"] for r in records)),
        "by_split": dict(Counter(r["split"] for r in records)),
    }
    (args.out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.copy_images:
        copy_images(args.out, records)
        print(f"Copied images under {args.out / 'images'}")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
