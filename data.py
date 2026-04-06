import glob
import json
import os
import random
from itertools import combinations
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ────────── KeyExtraction ──────────────────────────────────────────────────
def _extract_key(path: str, key_type: str = "subject") -> str:
    assert key_type in ["subject", "id"], (
        f"Invalid key type '{key_type}'. Choose from: ['subject', 'id']"
    )

    norm = path.replace("\\", "/")
    if "/LivDet/" in norm:
        parts = norm.split("/")
        livdet_idx = parts.index("LivDet")
        year = parts[livdet_idx + 1]
        sensor = parts[livdet_idx + 2]

        stem = os.path.splitext(parts[-1])[0]
        tokens = stem.split("_")

        if len(tokens) == 3:  # subject_finger_impression
            subject, finger, _ = tokens
            if key_type == "subject":
                """
                'data/LivDet/livdet2015/CrossMatch/Train/Live/0310542_L1_1.bmp' → 'livdet2015_CrossMatch_0310542'
                """
                return f"{year}_{sensor}_{subject}"
            else:  # "id"
                """
                'data/LivDet/livdet2015/CrossMatch/Train/Live/0310542_L1_1.bmp' → 'livdet2015_CrossMatch_0310542_L1'
                """
                return f"{year}_{sensor}_{subject}_{finger}"

        else:  # id_impression
            id_, impression = tokens
            """
            'data/LivDet/livdet2011/Biometrika/Train/Live/1_1.png' → 'livdet2011_Biometrika_1'
            """
            return f"{year}_{sensor}_{id_}"

    raise ValueError(path)


def _extract_id(path: str, id_type: str = "subject") -> str:
    assert id_type in ("subject", "finger"), (
        f"Invalid id type '{id_type}'. Choose from: ('subject', 'finger'])"
    )

    norm = path.replace("\\", "/")

    if "CASIA-FSA" in norm:
        filename = os.path.basename(norm)  # RRRR_IIIIF_XXXX_Z_S.bmp
        parts = filename.split("_")
        assert len(parts) == 5, f"Unexpected filename format: {filename}"

        iiiif = parts[1]  # IIIIF
        subject_id = iiiif[:-1]  # IIII
        finger_id = iiiif[-1]  # F

        dst = "casiafsa"

    elif "CASIA-FV5" in norm:
        filename = os.path.basename(norm)  # subject_finger_impression.bmp
        parts = filename.split("_")
        assert len(parts) == 3, f"Unexpected filename format: {filename}"

        subject_id = parts[0]
        finger_id = parts[1]

        dst = "casiafv5"

    elif "Neurotechnology" in norm and "CrossMatch" in norm:
        filename = os.path.basename(norm)  # subject_finger_impression.bmp
        parts = filename.split("_")
        assert len(parts) == 3, f"Unexpected filename format: {filename}"

        subject_id = parts[0]
        finger_id = parts[1]

        dst = "neurocm"

    elif "Neurotechnology" in norm and "UareU" in norm:
        filename = os.path.basename(norm)  # subject_finger_impression.bmp
        parts = filename.split("_")
        assert len(parts) == 3, f"Unexpected filename format: {filename}"

        subject_id = parts[0]
        finger_id = parts[1]

        dst = "neurouau"

    elif "SD301a" in norm:
        filename = os.path.basename(
            norm
        )  # SUBJECT_ENCOUNTER_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT
        parts = filename.split("_")
        assert len(parts) == 6, f"Unexpected filename format: {filename}"

        subject_id = parts[0]
        finger_id = parts[-1].split(".")[0]

        dst = "sd301a"

    elif "FVC" in norm:
        # data/FVC/FVC2000/Dbs/Db1_a/1_1.tif
        path_parts = norm.split("/")
        assert len(path_parts) == 6, f"Unexpected path format: {norm}"

        year = path_parts[-4]
        db = path_parts[-2]

        filename = os.path.basename(norm)
        parts = filename.split("_")
        assert len(parts) == 2, f"Unexpected filename format: {filename}"

        finger_id = parts[0]
        subject_id = finger_id

        dst = f"{year}_{db}"

    elif "SD302" in norm:
        """
        SUBJECT_DEVICE_CAPTURE_FRGP.EXT
        SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT
        """
        filename = os.path.basename(norm)
        parts = filename.split("_")
        assert 4 <= len(parts) <= 5, f"Unexpected filename format: {filename}"

        subject_id = parts[0]
        finger_id = parts[-1].split(".")[0]

        dst = "sd302"

    elif "PolyU" in norm:
        """
        data/PolyU/contact-based_fingerprints/first_session/1_1.jpg
        data/PolyU/contactless_2d_fingerprint_images/first_session/p1/p1.bmp
        """
        path_parts = norm.split("/")
        assert 5 <= len(path_parts) <= 6, f"Unexpected path format: {norm}"

        if len(path_parts) == 5:
            filename = os.path.basename(norm)  # X_Y.jpg
            parts = filename.split("_")
            assert len(parts) == 2, f"Unexpected filename format: {filename}"

            finger_id = parts[0]
            subject_id = finger_id
        else:
            pX = path_parts[-2]

            finger_id = pX[1:]
            subject_id = finger_id

        dst = "polyu"

    if id_type == "subject":
        return f"{dst}_{subject_id}"
    else:  # "finger"
        return f"{dst}_{subject_id}_{finger_id}"


def create_recog_splits(
    data_root: str,
    output_path: str,
    split_ratio: tuple = (0.8, 0.0, 0.2),
    seed: int = 42,
    min_samples: Optional[int] = 3,
    max_samples: Optional[int] = None,
) -> dict:
    assert len(split_ratio) == 3, "split_ratio must have 3 values (train, val, test)"
    assert all(r >= 0 for r in split_ratio), "split_ratio must be non-negative"
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1"

    rng = random.Random(seed)
    print(f"Creating splits for {data_root}")

    exts = ("*.bmp", "*.tif", "*.png", "*.jpg")
    all_paths = [
        p
        for ext in exts
        for p in glob.glob(os.path.join(data_root, "**", ext), recursive=True)
    ]

    subject_finger_paths = {}
    for path in all_paths:
        subject = _extract_id(path, "subject")
        finger = _extract_id(path, "finger")
        if subject not in subject_finger_paths:
            subject_finger_paths[subject] = {}
        if finger not in subject_finger_paths[subject]:
            subject_finger_paths[subject][finger] = []
        subject_finger_paths[subject][finger].append(path)

    if min_samples is not None or max_samples is not None:
        filtered_subject_finger_paths = {}
        removed_count = 0

        for subject, fingers in subject_finger_paths.items():
            valid_fingers = {}
            for finger, paths in fingers.items():
                count = len(paths)
                # Check bounds
                if (min_samples is None or count >= min_samples) and (
                    max_samples is None or count <= max_samples
                ):
                    valid_fingers[finger] = paths
                else:
                    removed_count += 1

            if valid_fingers:
                filtered_subject_finger_paths[subject] = valid_fingers

        subject_finger_paths = filtered_subject_finger_paths
        print(f"Filtered out {removed_count} fingers based on sample constraints.")

    all_subjects = sorted(subject_finger_paths.keys())
    rng.shuffle(all_subjects)

    n_total = len(all_subjects)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_subjects = set(all_subjects[:n_train])
    val_subjects = set(all_subjects[n_train : n_train + n_val])

    splits = {
        "train": {},
        "val": {},
        "test": {},
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
    }

    for subject, fingers in subject_finger_paths.items():
        target = (
            "train"
            if subject in train_subjects
            else "val"
            if subject in val_subjects
            else "test"
        )
        for finger, paths in fingers.items():
            splits[target][finger] = paths
            splits[f"{target}_samples"] += len(paths)

    splits.update(
        {
            "train_subjects": n_train,
            "val_subjects": n_val,
            "test_subjects": n_total - n_train - n_val,
            "train_fingers": len(splits["train"]),
            "val_fingers": len(splits["val"]),
            "test_fingers": len(splits["test"]),
            "total_subjects": n_total,
            "total_fingers": len(splits["train"])
            + len(splits["val"])
            + len(splits["test"]),
            "total_samples": splits["train_samples"]
            + splits["val_samples"]
            + splits["test_samples"],
        }
    )

    print(
        f"• Train: {splits['train_samples']} samples ({splits['train_subjects']} subjects / {splits['train_fingers']} fingers)\n"
        f"• Val: {splits['val_samples']} samples ({splits['val_subjects']} subjects / {splits['val_fingers']} fingers)\n"
        f"• Test: {splits['test_samples']} samples ({splits['test_subjects']} subjects / {splits['test_fingers']} fingers)\n"
        f"Total: {splits['total_samples']} samples ({splits['total_subjects']} subjects / {splits['total_fingers']} fingers)"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


# ─────LivDet─────
def create_LivDet_recog_splits(
    data_root: str = "data/LivDet/",
    output_path: str = "data/LivDet/livdet_recog_splits.json",
    val_ratio: float = 0.2,
    seed: int = 42,
    min_samples: int = 3,
    max_samples: int = 20,
) -> dict:

    def _collect_live_paths(sensor_dir: str, split: str) -> list[str]:
        base = os.path.join(sensor_dir, split, "Live")
        return sorted(
            glob.glob(os.path.join(base, "*.png"))
            + glob.glob(os.path.join(base, "*.bmp"))
        )

    def _filter_by_id(paths: list[str], rng: random.Random) -> list[str]:
        id_to_paths: dict[str, list[str]] = {}
        for p in paths:
            id_to_paths.setdefault(_extract_key(p, "id"), []).append(p)

        kept: list[str] = []
        for id_paths in id_to_paths.values():
            if len(id_paths) < min_samples:
                continue
            if len(id_paths) > max_samples:
                id_paths = rng.sample(id_paths, max_samples)
            kept.extend(id_paths)

        return sorted(kept)

    assert 0.0 < val_ratio < 1.0
    assert 0 < min_samples <= max_samples

    sensors = {
        "livdet2011_Biometrika": os.path.join(data_root, "livdet2011", "Biometrika"),
        "livdet2011_Digital": os.path.join(data_root, "livdet2011", "Digital"),
        "livdet2011_Italdata": os.path.join(data_root, "livdet2011", "Italdata"),
        "livdet2011_Sagem": os.path.join(data_root, "livdet2011", "Sagem"),
        "livdet2013_Biometrika": os.path.join(data_root, "livdet2013", "Biometrika"),
        "livdet2013_CrossMatch": os.path.join(data_root, "livdet2013", "CrossMatch"),
        "livdet2013_Italdata": os.path.join(data_root, "livdet2013", "Italdata"),
        "livdet2015_CrossMatch": os.path.join(data_root, "livdet2015", "CrossMatch"),
        "livdet2015_DigitalPersona": os.path.join(
            data_root, "livdet2015", "DigitalPersona"
        ),
        "livdet2015_GreenBit": os.path.join(data_root, "livdet2015", "GreenBit"),
        "livdet2015_HiScan": os.path.join(data_root, "livdet2015", "HiScan"),
    }
    rng = random.Random(seed)
    unified = {"train": [], "val": [], "train_subjects": 0, "val_subjects": 0}

    for sensor_key, sensor_dir in sensors.items():
        test_paths = _filter_by_id(_collect_live_paths(sensor_dir, "Test"), rng)
        unified[f"test_{sensor_key}"] = test_paths

        train_paths = _filter_by_id(_collect_live_paths(sensor_dir, "Train"), rng)

        subject_to_paths: dict[str, list[str]] = {}
        for p in train_paths:
            subject_to_paths.setdefault(_extract_key(p, "subject"), []).append(p)

        all_subjects = sorted(subject_to_paths.keys())
        rng.shuffle(all_subjects)

        n_val = int(len(all_subjects) * val_ratio)
        n_train = len(all_subjects) - n_val
        val_set = set(all_subjects[n_train:])

        for sid, paths in subject_to_paths.items():
            target = unified["val"] if sid in val_set else unified["train"]
            target.extend(paths)

        unified["train_subjects"] += n_train
        unified["val_subjects"] += n_val

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    test_keys = [k for k in unified if k.startswith("test_")]
    n_test = sum(len(unified[k]) for k in test_keys)

    print(
        f"LivDet Recognition Splits:\n"
        f"• Train: {len(unified['train'])} samples ({unified['train_subjects']} subjects)\n"
        f"• Val: {len(unified['val'])} samples ({unified['val_subjects']} subjects)\n"
        f"• Test: {n_test} samples across {len(test_keys)} sensors"
    )

    return unified


def create_LivDet_PAD_splits(
    data_root: str = "data/LivDet/",
    recog_json_path: str = "data/LivDet/livdet_recog_splits.json",
    output_path: str = "data/LivDet/livdet_pad_splits.json",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    def _collect_spoof_paths(sensor_dir: str, split: str) -> list[str]:
        base = os.path.join(sensor_dir, split, "Spoof")
        return sorted(
            glob.glob(os.path.join(base, "*", "*.png"))
            + glob.glob(os.path.join(base, "*", "*.bmp"))
        )

    def _cnt(entries, lbl):
        return sum(1 for e in entries if e["label"] == lbl)

    assert 0.0 < val_ratio < 1.0

    with open(recog_json_path) as f:
        recog = json.load(f)

    unified = {}
    for key, value in recog.items():
        if isinstance(value, list):
            unified[key] = [{"path": p, "label": 0} for p in value]

    sensors = {
        "livdet2011_Biometrika": os.path.join(data_root, "livdet2011", "Biometrika"),
        "livdet2011_Digital": os.path.join(data_root, "livdet2011", "Digital"),
        "livdet2011_Italdata": os.path.join(data_root, "livdet2011", "Italdata"),
        "livdet2011_Sagem": os.path.join(data_root, "livdet2011", "Sagem"),
        "livdet2013_Biometrika": os.path.join(data_root, "livdet2013", "Biometrika"),
        "livdet2013_CrossMatch": os.path.join(data_root, "livdet2013", "CrossMatch"),
        "livdet2013_Italdata": os.path.join(data_root, "livdet2013", "Italdata"),
        "livdet2015_CrossMatch": os.path.join(data_root, "livdet2015", "CrossMatch"),
        "livdet2015_DigitalPersona": os.path.join(
            data_root, "livdet2015", "DigitalPersona"
        ),
        "livdet2015_GreenBit": os.path.join(data_root, "livdet2015", "GreenBit"),
        "livdet2015_HiScan": os.path.join(data_root, "livdet2015", "HiScan"),
    }
    rng = random.Random(seed)

    for sensor_key, sensor_dir in sensors.items():
        test_key = f"test_{sensor_key}"
        for p in _collect_spoof_paths(sensor_dir, "Test"):
            unified[test_key].append({"path": p, "label": 1})

        spoof_paths = _collect_spoof_paths(sensor_dir, "Train")
        if not spoof_paths:
            continue

        shuffled = spoof_paths[:]
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_ratio))
        for p in shuffled[n_val:]:
            unified["train"].append({"path": p, "label": 1})
        for p in shuffled[:n_val]:
            unified["val"].append({"path": p, "label": 1})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    test_keys = [k for k in unified if k.startswith("test_")]
    n_test = sum(len(unified[k]) for k in test_keys)

    print(
        f"LivDet PAD Splits:\n"
        f"• Train: {len(unified['train'])} samples (live={_cnt(unified['train'], 0)}, spoof={_cnt(unified['train'], 1)})\n"
        f"• Val: {len(unified['val'])} samples (live={_cnt(unified['val'], 0)}, spoof={_cnt(unified['val'], 1)})\n"
        f"• Test: {n_test} imgs across {len(test_keys)} sensors"
    )

    return unified


def unify_recog_splits(
    splits_paths: list,
    output_path: str = "data/splits.json",
):
    print(f"Unifying splits from {splits_paths}")

    unified = {
        "train": {},
        "val": {},
        "test": {},
        "train_subjects": 0,
        "val_subjects": 0,
        "test_subjects": 0,
        "train_fingers": 0,
        "val_fingers": 0,
        "test_fingers": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "total_subjects": 0,
        "total_fingers": 0,
        "total_samples": 0,
    }
    for splits_path in splits_paths:
        with open(splits_path, "r") as f:
            splits = json.load(f)

        unified["train"].update(splits["train"])
        unified["val"].update(splits["val"])
        unified["test"].update(splits["test"])

        unified["train_subjects"] += splits["train_subjects"]
        unified["val_subjects"] += splits["val_subjects"]
        unified["test_subjects"] += splits["test_subjects"]
        unified["train_fingers"] += splits["train_fingers"]
        unified["val_fingers"] += splits["val_fingers"]
        unified["test_fingers"] += splits["test_fingers"]
        unified["train_samples"] += splits["train_samples"]
        unified["val_samples"] += splits["val_samples"]
        unified["test_samples"] += splits["test_samples"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    unified["total_subjects"] = (
        unified["train_subjects"] + unified["val_subjects"] + unified["test_subjects"]
    )
    unified["total_fingers"] = (
        unified["train_fingers"] + unified["val_fingers"] + unified["test_fingers"]
    )
    unified["total_samples"] = (
        unified["train_samples"] + unified["val_samples"] + unified["test_samples"]
    )

    print(
        f"• Train: {unified['train_samples']} samples ({unified['train_subjects']} subjects / {unified['train_fingers']} fingers)\n"
        f"• Val: {unified['val_samples']} samples ({unified['val_subjects']} subjects / {unified['val_fingers']} fingers)\n"
        f"• Test: {unified['test_samples']} samples ({unified['test_subjects']} subjects / {unified['test_fingers']} fingers)\n"
        f"Total: {unified['total_samples']} samples ({unified['total_subjects']} subjects / {unified['total_fingers']} fingers)"
    )

    return unified


class RecogTrainingDataset(Dataset):
    def __init__(
        self,
        splits_path: str = "data/splits.json",
        transform: Optional[Callable] = None,
    ):
        self.transform = transform

        with open(splits_path, "r") as f:
            finger_to_paths = json.load(f)["train"]

        self.paths = []
        finger_ids = []
        for finger, paths in finger_to_paths.items():
            self.paths.extend(paths)
            finger_ids.extend([finger] * len(paths))

        unique_ids = sorted(set(finger_ids))
        self.id_to_label = {id_: idx for idx, id_ in enumerate(unique_ids)}
        self.labels = [self.id_to_label[k] for k in finger_ids]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __repr__(self):
        return (
            f"RecogTrainingDataset: {len(self)} samples ({len(self.id_to_label)} ids)"
        )


class RecogEvaluationDataset(Dataset):
    def __init__(
        self,
        splits_path: str = "data/splits.json",
        split: str = "test",
        n_genuine_impressions: int = 2,
        n_impostor_impressions: int = 1,
        impostor_mode: str = "all",
        n_impostor_subset: Optional[int] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.split = split
        self.n_genuine_impressions = n_genuine_impressions
        self.n_impostor_impressions = n_impostor_impressions
        self.impostor_mode = impostor_mode
        self.transform = transform

        assert impostor_mode in ("all", "sub"), f"Invalid mode: {impostor_mode}"
        assert n_genuine_impressions >= 2, "n_genuine_impressions must be >= 2"
        assert n_genuine_impressions >= 1, "n_impostor_impressions must be >= 1"
        assert split in ("test", "val"), f"Invalid split: {split}"

        with open(splits_path, "r") as f:
            splits = json.load(f)

        finger_to_paths = splits[split]
        self.n_ids = len(finger_to_paths)

        rng = random.Random(seed)
        genuine_pairs = []
        for paths in finger_to_paths.values():
            num_to_sample = min(len(paths), self.n_genuine_impressions)
            selected = rng.sample(paths, num_to_sample)
            for path_a, path_b in combinations(selected, 2):
                genuine_pairs.append((path_a, path_b, 1))

        finger_paths = list(finger_to_paths.values())
        impostor_pairs = []
        if impostor_mode == "all":
            for _ in range(n_impostor_impressions):
                impression_slice = [rng.choice(p) for p in finger_paths]
                for path_a, path_b in combinations(impression_slice, 2):
                    impostor_pairs.append((path_a, path_b, 0))
        else:  # "sub"
            assert n_impostor_subset is not None, (
                "Number of impostor subset must be not None if impostor mode is 'sub'"
            )
            assert 1 <= n_impostor_subset < self.n_ids, (
                "Number of impostor subset must be greater than or equal to 1 and lower than number of fingers if impostor mode is 'sub'"
            )

            for finger_idx, paths in enumerate(finger_paths):
                other_indices = list(range(self.n_ids))
                other_indices.remove(finger_idx)
                for _ in range(n_impostor_impressions):
                    path_a = rng.choice(paths)
                    sampled = rng.sample(other_indices, n_impostor_subset)
                    for other_idx in sampled:
                        path_b = rng.choice(finger_paths[other_idx])
                        impostor_pairs.append((path_a, path_b, 0))

        self.pairs = genuine_pairs + impostor_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_a, path_b, label = self.pairs[idx]
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b, label

    def __repr__(self):
        n_genuine = sum(1 for *_, lbl in self.pairs if lbl == 1)
        n_impostor = sum(1 for *_, lbl in self.pairs if lbl == 0)
        return (
            f"RecogEvaluationDataset:\n"
            f"• n_pairs: {len(self)}\n"
            f"• n_genuine: {n_genuine}\n"
            f"• n_impostor: {n_impostor}\n"
            f"• n_ids: {self.n_ids}"
        )


# ─────PAD─────
class PADDataset(Dataset):
    def __init__(
        self,
        json_path: str = "data/LivDet/livdet_pad_splits.json",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.transform = transform
        self.split = split

        with open(json_path) as f:
            splits = json.load(f)
        assert split in splits, (
            f"Split '{split}' not found. Available: {list(splits.keys())}"
        )

        entries = splits[split]
        self.paths = [e["path"] for e in entries]
        self.labels = [e["label"] for e in entries]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __repr__(self):
        return (
            f"PADDataset:\n"
            f"• split: '{self.split}'\n"
            f"• n_samples: {len(self)}\n"
            f"• live: {self.labels.count(0)}\n"
            f"• spoof: {self.labels.count(1)}"
        )


# ─────Dataloader─────
def create_dataloaders(
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader | tuple:
    def _loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    loaders = tuple(
        _loader(ds, shuffle=(ds is train_dataset))
        for ds in (train_dataset, val_dataset, test_dataset)
        if ds is not None
    )

    return loaders[0] if len(loaders) == 1 else loaders


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # data_root = "data/FVC"
    # output_path = "data/FVC/fvc_splits.json"
    # create_recog_splits(data_root=data_root, output_path=output_path, split_ratio=[0.6, 0.2, 0.2])

    # data_roots = [
    #     "CASIA-FSA",
    #     "CASIA-FV5",
    #     "FVC",
    #     "Neurotechnology/CrossMatch",
    #     "Neurotechnology/UareU",
    #     "PolyU",
    #     "SD301a",
    #     "SD302",
    # ]
    # output_paths = [
    #     "casiafsa",
    #     "casiafv5",
    #     "fvc",
    #     "neurocm",
    #     "neurouau",
    #     "polyu",
    #     "sd301a",
    #     "sd302",
    # ]
    # for data_root, output_path in zip(data_roots, output_paths):
    #     create_recog_splits(
    #         data_root="data/" + data_root,
    #         output_path=f"data/{data_root}/{output_path}_splits.json",
    #         split_ratio=[0.6, 0.2, 0.2]
    #     )
    #     print()

    # unify_recog_splits(
    #     splits_paths=[
    #         "data/CASIA-FSA/casiafsa_splits.json",
    #         "data/CASIA-FV5/casiafv5_splits.json",
    #         "data/FVC/fvc_splits.json",
    #         "data/Neurotechnology/CrossMatch/neurocm_splits.json",
    #         "data/Neurotechnology/UareU/neurouau_splits.json",
    #         "data/PolyU/polyu_splits.json",
    #         "data/SD301a/sd301a_splits.json",
    #         "data/SD302/sd302_splits.json",
    #     ],
    #     output_path="data/splits.json",
    # )

    # train_dataset = RecogTrainingDataset(
    #     splits_path="data/splits.json", transform=transform
    # )
    # print(train_dataset)

    # train_dataloader = create_dataloaders(train_dataset=train_dataset, batch_size=4)
    # for imgs, labels in train_dataloader:
    #     print(imgs.shape)
    #     print(labels)
    #     break

    val_dataset = RecogEvaluationDataset(
        splits_path="data/splits.json",
        transform=transform,
        split="val",
        n_genuine_impressions=8,
        impostor_mode="sub",
        n_impostor_subset=100
    )
    print(val_dataset)

    # test_dataset = RecogEvaluationDataset(
    #     splits_path="data/splits.json", transform=transform
    # )
    # print(test_dataset)

    # train_dataloader = create_dataloaders(test_dataset=test_dataset, batch_size=4)
    # for img_as, img_bs, labels in train_dataloader:
    #     print(img_as.shape)
    #     print(img_bs.shape)
    #     print(labels)
    #     break
