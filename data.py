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

    if "/FVC/" in norm:
        """
        'data/FVC/fvc2000/db1/100_1.tif' → 'fvc2000_db1_100'
        """
        parts = norm.split("/")
        year = parts[-3]
        db = parts[-2]
        id = os.path.basename(norm).split("_")[0]
        return f"{year}_{db}_{id}"

    if "/SD302/" in norm:
        parts = os.path.basename(norm).split("_")
        subject = parts[0]
        id = os.path.splitext(parts[-1])[0]
        if key_type == "subject":
            """
            'data/SD302/a/A/00002500_A_roll_01.png'     → 'sd302_00002500'
            'data/SD302/b/U/00002546_U_500_roll_07.png' → 'sd302_00002546'
            """
            return f"sd302_{subject}"
        else:  # "id"
            """
            'data/SD302/a/A/00002500_A_roll_01.png'     → 'sd302_00002500_01'
            """
            return f"sd302_{subject}_{id}"

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


# ────────── FVC ──────────────────────────────────────────────────
def create_FVC_splits(
    data_root: str = "data/FVC/",
    output_path: str = "data/FVC/fvc_splits.json",
    split_ratio: tuple = (0.6, 0.1, 0.3),
    seed: int = 42,
) -> dict:
    assert round(sum(split_ratio), 10) == 1.0

    rng = random.Random(seed)

    databases = {
        "fvc2000_db1": os.path.join(data_root, "fvc2000", "db1"),
        "fvc2000_db2": os.path.join(data_root, "fvc2000", "db2"),
        "fvc2002_db1": os.path.join(data_root, "fvc2002", "db1"),
        "fvc2002_db2": os.path.join(data_root, "fvc2002", "db2"),
        "fvc2002_db3": os.path.join(data_root, "fvc2002", "db3"),
        "fvc2004_db1": os.path.join(data_root, "fvc2004", "db1"),
        "fvc2004_db2": os.path.join(data_root, "fvc2004", "db2"),
    }

    unified = {"train": [], "val": [], "train_subjects": 0, "val_subjects": 0}

    for db_key, db_dir in databases.items():
        subjects_to_paths: dict[str, list[str]] = {}
        for path in sorted(glob.glob(os.path.join(db_dir, "*.tif"))):
            subjects_to_paths.setdefault(_extract_key(path, "subject"), []).append(path)

        all_subjects = sorted(subjects_to_paths.keys())
        rng.shuffle(all_subjects)

        n_total = len(all_subjects)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        train_subjects = set(all_subjects[:n_train])
        val_subjects = set(all_subjects[n_train : n_train + n_val])
        test_subjects = set(all_subjects[n_train + n_val :])

        db_test_samples = []
        for sid, paths in subjects_to_paths.items():
            if sid in train_subjects:
                unified["train"].extend(paths)
            elif sid in val_subjects:
                unified["val"].extend(paths)
            else:
                db_test_samples.extend(paths)

        unified[f"test_{db_key}"] = db_test_samples
        unified[f"test_subjects_{db_key}"] = len(test_subjects)

        unified["train_subjects"] += n_train
        unified["val_subjects"] += n_val

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    n_test = sum(
        len(unified[k])
        for k in unified
        if k.startswith("test_") and isinstance(unified[k], list)
    )

    print(
        f"FVC Dataset Splits:\n"
        f"• Train: {len(unified['train'])} samples ({unified['train_subjects']} subjects)\n"
        f"• Val: {len(unified['val'])} samples ({unified['val_subjects']} subjects)\n"
        f"• Test: {n_test} samples split across {len(databases)} DBs"
    )

    return unified


# ────────── SD302 ──────────────────────────────────────────────────
def create_SD302_splits(
    data_root: str = "data/SD302/",
    output_path: str = "data/SD302/sd302_splits.json",
    split_ratio: tuple = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict:
    assert round(sum(split_ratio), 10) == 1.0

    modalities = {
        "sd302a_A": os.path.join(data_root, "sd302a", "A"),
        "sd302a_B": os.path.join(data_root, "sd302a", "B"),
        "sd302a_C": os.path.join(data_root, "sd302a", "C"),
        "sd302a_E": os.path.join(data_root, "sd302a", "E"),
        "sd302a_F": os.path.join(data_root, "sd302a", "F"),
        "sd302a_G": os.path.join(data_root, "sd302a", "G"),
        "sd302b_R": os.path.join(data_root, "sd302b", "R"),
        "sd302b_S": os.path.join(data_root, "sd302b", "S"),
        "sd302b_U": os.path.join(data_root, "sd302b", "U"),
        "sd302b_V": os.path.join(data_root, "sd302b", "V"),
        "sd302c_J": os.path.join(data_root, "sd302c", "J"),
        "sd302c_N": os.path.join(data_root, "sd302c", "N"),
        "sd302c_Q": os.path.join(data_root, "sd302c", "Q"),
        "sd302d_K": os.path.join(data_root, "sd302d", "K"),
        "sd302d_L": os.path.join(data_root, "sd302d", "L"),
        "sd302d_M": os.path.join(data_root, "sd302d", "M"),
        "sd302d_P": os.path.join(data_root, "sd302d", "P"),
    }

    subject_to_paths = {}

    for modal_key, modal_dir in modalities.items():
        for path in sorted(glob.glob(os.path.join(modal_dir, "*.png"))):
            subject_to_paths.setdefault(_extract_key(path, "subject"), []).append(path)

    all_subjects = sorted(subject_to_paths.keys())
    rng = random.Random(seed)
    rng.shuffle(all_subjects)

    n_total = len(all_subjects)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_subjects = set(all_subjects[:n_train])
    val_subjects = set(all_subjects[n_train : n_train + n_val])

    unified: dict[str, list[str] | int] = {"train": [], "val": [], "test": []}
    for sid, paths in subject_to_paths.items():
        if sid in train_subjects:
            unified["train"].extend(paths)
        elif sid in val_subjects:
            unified["val"].extend(paths)
        else:
            unified["test"].extend(paths)

    unified["train_subjects"] = n_train
    unified["val_subjects"] = n_val
    unified["test_subjects"] = n_total - n_train - n_val

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    print(
        f"SD302 Splits:\n"
        f"• Train: {len(unified['train'])} samples ({unified['train_subjects']} subjects)\n"
        f"• Val: {len(unified['val'])} samples ({unified['val_subjects']} subjects)\n"
        f"• Test: {len(unified['test'])} samples ({unified['test_subjects']} subjects)"
    )

    return unified


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


# ─────Unified─────
def unify_recog_splits(
    data_root: str = "data/",
    datasets: list = ["fvc", "sd302", "livdet"],
    output_path: str = "data/splits.json",
):
    json_paths = []

    for dataset in datasets:
        if dataset == "fvc":
            if not os.path.exists(os.path.join(data_root, "FVC/fvc_splits.json")):
                create_FVC_splits(
                    data_root=data_root + "FVC",
                    output_path=data_root + "FVC/fvc_splits.json",
                )
            json_paths.append(data_root + "FVC/fvc_splits.json")

        elif dataset == "sd302":
            if not os.path.exists(os.path.join(data_root, "SD302/sd302_splits.json")):
                create_SD302_splits(
                    data_root=data_root + "SD302",
                    output_path=data_root + "SD302/sd302_splits.json",
                )
            json_paths.append(data_root + "SD302/sd302_splits.json")

        elif dataset == "livdet":
            if not os.path.exists(
                os.path.join(data_root, "LivDet/livdet_recog_splits.json")
            ):
                create_LivDet_recog_splits(
                    data_root=data_root + "LivDet",
                    output_path=data_root + "LivDet/livdet_recog_splits.json",
                )
            json_paths.append(data_root + "LivDet/livdet_recog_splits.json")

        else:
            raise ValueError(dataset)

    unified = {"train": [], "val": [], "train_subjects": 0, "val_subjects": 0}

    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)

        unified["train"].extend(data["train"])
        unified["val"].extend(data["val"])

        unified["train_subjects"] += data["train_subjects"]
        unified["val_subjects"] += data["val_subjects"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    print(
        f"Unified Recognition Splits:\n"
        f"• Train: {len(unified['train'])} samples ({unified['train_subjects']} subjects)\n"
        f"• Val: {len(unified['val'])} samples ({unified['val_subjects']} subjects)\n"
    )

    return unified


# ─────Recognition─────
class RecogTrainingDataset(Dataset):
    def __init__(
        self,
        json_path: str = "data/splits.json",
        transform: Optional[Callable] = None,
    ):
        self.transform = transform

        with open(json_path, "r") as f:
            self.paths = json.load(f)["train"]
        id_keys: list[str] = [_extract_key(p, "id") for p in self.paths]

        unique_keys = sorted(set(id_keys))
        self.key_to_label = {key: idx for idx, key in enumerate(unique_keys)}
        self.labels = [self.key_to_label[k] for k in id_keys]

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
            f"RecogTrainingDataset:\n"
            f"• n_ids: {len(self.key_to_label)}\n"
            f"• n_samples: {len(self)}"
        )


class RecogEvaluationDataset(Dataset):
    def __init__(
        self,
        json_path: str = "data/splits.json",
        split: str = "val",
        n_genuine_impressions: int = 2,
        n_impostor_impressions: int = 1,
        impostor_mode: str = "all",
        n_impostor_subset: Optional[int] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        assert impostor_mode in ("all", "sub")

        with open(json_path, "r") as f:
            splits = json.load(f)
        assert split in splits

        paths = splits[split]
        self.transform = transform
        self.split = split
        self.impostor_mode = impostor_mode

        rng = random.Random(seed)

        id_to_paths = {}
        for path in paths:
            key = _extract_key(path, "id")
            id_to_paths.setdefault(key, []).append(path)
        for id in id_to_paths:
            id_to_paths[id].sort()

        min_impressions = min(len(v) for v in id_to_paths.values())
        assert 2 <= n_genuine_impressions <= min_impressions
        assert 1 <= n_impostor_impressions

        self.n_genuine_impressions = n_genuine_impressions
        self.n_impostor_impressions = n_impostor_impressions
        self.n_ids = len(id_to_paths)

        genuine_pairs = []
        for paths in id_to_paths.values():
            selected = rng.sample(paths, n_genuine_impressions)
            for path_a, path_b in combinations(selected, 2):
                genuine_pairs.append((path_a, path_b, 0))

        id_paths = list(id_to_paths.values())
        impostor_pairs = []
        if impostor_mode == "all":
            for _ in range(n_impostor_impressions):
                impression_slice = [rng.choice(p) for p in id_paths]
                for path_a, path_b in combinations(impression_slice, 2):
                    impostor_pairs.append((path_a, path_b, 1))
        else:  # "sub"
            assert n_impostor_subset is not None
            assert 1 <= n_impostor_subset < self.n_ids

            for id_idx, anchor_paths in enumerate(id_paths):
                other_indices = [i for i in range(self.n_ids) if i != id_idx]
                for _ in range(n_impostor_impressions):
                    path_a = rng.choice(anchor_paths)
                    sampled = rng.sample(other_indices, n_impostor_subset)
                    for other_idx in sampled:
                        path_b = rng.choice(id_paths[other_idx])
                        impostor_pairs.append((path_a, path_b, 1))

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
        n_genuine = sum(1 for *_, lbl in self.pairs if lbl == 0)
        n_impostor = sum(1 for *_, lbl in self.pairs if lbl == 1)
        return (
            f"RecogEvaluationDataset:\n"
            f"• split: '{self.split}'\n"
            f"• n_ids: {self.n_ids}\n"
            f"• n_genuine_impressions: {self.n_genuine_impressions}\n"
            f"• n_impostor_impressions: {self.n_impostor_impressions}\n"
            f"• impostor_mode: {self.impostor_mode}\n"
            f"• n_pairs: {len(self)}\n"
            f"• genuine: {n_genuine}\n"
            f"• impostor: {n_impostor}"
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
    pin_memory: bool = False,
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

    # create_FVC_splits()
    create_LivDet_recog_splits()
    create_LivDet_PAD_splits()
