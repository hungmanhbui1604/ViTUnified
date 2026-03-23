import os
import json
import random
import glob
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from torchvision import transforms


def _extract_key(path: str, key_type: str = "subject") -> str:
    valid_key_types = ["subject", "id"]
    assert key_type in valid_key_types, f"Invalid key type '{key_type}'. Choose from: {valid_key_types}"
    
    norm = path.replace("\\", "/")

    if "/FVC/" in norm:
        parts  = norm.split("/")
        year   = parts[-3]                          # fvc2000
        db     = parts[-2]                          # db1
        id = os.path.basename(norm).split("_")[0]   # 100
        # 'data/FVC/fvc2000/db1/100_1.tif' → 'fvc2000_db1_100
        return f"{year}_{db}_{id}"
    
    if "/SD302/" in norm:
        parts = os.path.basename(path).split("_")
        subject = parts[0]
        id = os.path.splitext(parts[-1])[0]
        if key_type == "subject":
            """
            'data/SD302/a/A/00002500_A_roll_01.png'     → 'sd302_00002500'
            'data/SD302/b/U/00002546_U_500_roll_07.png' → 'sd302_00002546'
            """
            return f"sd302_{subject}"
        else:
            # 'data/SD302/a/A/00002500_A_roll_01.png'   → 'sd302_00002500_01'
            return f"sd302_{subject}_{id}"
    
    if "/LivDet/" in norm:
        parts      = norm.split("/")
        livdet_idx = parts.index("LivDet")
        year       = parts[livdet_idx + 1]   # livdet2015
        sensor     = parts[livdet_idx + 2]   # CrossMatch

        stem       = os.path.splitext(parts[-1])[0]   # "0310542_L1_1" or "1_1"
        tokens     = stem.split("_")

        if len(tokens) == 3:
            # subject_finger_impression  →  e.g. "0310542_L1_1"
            subject, finger, _ = tokens
            if key_type == "subject":
                # 'data/LivDet/livdet2015/CrossMatch/Train/Live/0310542_L1_1.bmp' → 'livdet2015_CrossMatch_0310542'
                return f"{year}_{sensor}_{subject}"
            else:
                # 'data/LivDet/livdet2015/CrossMatch/Train/Live/0310542_L1_1.bmp' → 'livdet2015_CrossMatch_0310542_L1'
                return f"{year}_{sensor}_{subject}_{finger}"

        elif len(tokens) == 2:
            # id_impression  →  e.g. "1_1"
            id_, impression = tokens
            # 'data/LivDet/livdet2011/Biometrika/Train/Live/1_1.png' → 'livdet2011_Biometrika_1'
            return f"{year}_{sensor}_{id_}"

        else:
            raise ValueError(
                f"Unexpected LivDet filename format '{parts[-1]}' in path '{path}'. "
                f"Expected 'subject_finger_impression.png' or 'id_impression.png'."
            )
    
    raise ValueError(
        f"Cannot determine dataset for path '{path}'."
    )


# ──FVC───────────────────────────────────────────
def create_FVC_splits(
    data_root: str = "data/FVC/",
    output_path: str = "data/FVC/fvc_splits.json",
    split_ratio: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    assert round(sum(split_ratio), 10) == 1.0, "Split ratios must sum to 1."
 
    databases = {
        "fvc2000_db1": os.path.join(data_root, "fvc2000", "db1"),
        "fvc2000_db2": os.path.join(data_root, "fvc2000", "db2"),
        "fvc2002_db1": os.path.join(data_root, "fvc2002", "db1"),
        "fvc2002_db2": os.path.join(data_root, "fvc2002", "db2"),
        "fvc2002_db3": os.path.join(data_root, "fvc2002", "db3"),
        "fvc2004_db1": os.path.join(data_root, "fvc2004", "db1"),
        "fvc2004_db2": os.path.join(data_root, "fvc2004", "db2"),
    }
 
    unified: dict = {
        "train": [],
        "val": [],
        "train_subjects": 0,
        "val_subjects": 0,
    }
 
    rng = random.Random(seed)
 
    for db_key, db_dir in databases.items():
        # Collect subjects for this DB
        subjects_to_paths: dict[str, list[str]] = {}
        for path in sorted(glob.glob(os.path.join(db_dir, "*.tif"))):
            subjects_to_paths.setdefault(_extract_key(path, "subject"), []).append(path)
 
        all_subjects = sorted(subjects_to_paths.keys())
        rng.shuffle(all_subjects)
 
        n_total = len(all_subjects)
        n_train = int(n_total * split_ratio[0])
        n_val   = int(n_total * split_ratio[1])
 
        train_subjects = set(all_subjects[:n_train])
        val_subjects   = set(all_subjects[n_train:n_train + n_val])
        test_subjects  = set(all_subjects[n_train + n_val:])
 
        db_test_images = []
        for sid, paths in subjects_to_paths.items():
            if sid in train_subjects:
                unified["train"].extend(paths)
            elif sid in val_subjects:
                unified["val"].extend(paths)
            else:
                db_test_images.extend(paths)
 
        # Keep test split separate per DB
        unified[f"test_{db_key}"] = db_test_images
        unified[f"test_subjects_{db_key}"] = len(test_subjects)
 
        unified["train_subjects"] += n_train
        unified["val_subjects"]   += n_val
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)
 
    n_test = sum(
        len(unified[k]) for k in unified
        if k.startswith("test_") and isinstance(unified[k], list)
    )
    print(
        f"FVC splits created → "
        f"train: {len(unified['train'])} images ({unified['train_subjects']} subjects), "
        f"val: {len(unified['val'])} images ({unified['val_subjects']} subjects), "
        f"test: {n_test} images split across {len(databases)} DBs"
    )
 
    return unified


# ──SD302───────────────────────────────────────────
def create_SD302_splits(
    data_root: str = "data/SD302/",
    output_path: str = "data/SD302/sd302_splits.json",
    split_ratio: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    assert round(sum(split_ratio), 10) == 1.0, "Split ratios must sum to 1."

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
        "sd302b_V": os.path.join(data_root, "sd302b", "V")
    }

    subject_to_paths: dict[str, list[str]] = {}

    for modal_key, modal_dir in modalities.items():
        for path in sorted(glob.glob(os.path.join(modal_dir, "*.png"))):
            subject_to_paths.setdefault(_extract_key(path, "subject"), []).append(path)

    all_subjects = sorted(subject_to_paths.keys())
    rng = random.Random(seed)
    rng.shuffle(all_subjects)

    n_total = len(all_subjects)
    n_train = int(n_total * split_ratio[0])
    n_val   = int(n_total * split_ratio[1])

    train_subjects = set(all_subjects[:n_train])
    val_subjects   = set(all_subjects[n_train:n_train + n_val])

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
        f"SD302 splits created → "
        f"train: {len(unified['train'])} images ({unified["train_subjects"]} subjects), "
        f"val: {len(unified['val'])} images ({unified["val_subjects"]} subjects), "
        f"test: {len(unified['test'])} images ({unified["test_subjects"]} subjects)"
    )
    return unified


# ──LivDet───────────────────────────────────────────
def create_LivDet_recog_splits(
    data_root:   str   = "data/LivDet/",
    output_path: str   = "data/LivDet/livdet_recog_splits.json",
    val_ratio:   float = 0.2,
    seed:        int   = 42,
) -> dict:
    
    def _collect_live_paths(sensor_dir: str, split: str) -> list[str]:
        base = os.path.join(sensor_dir, split, "Live")
        return sorted(glob.glob(os.path.join(base, "*.png")) + glob.glob(os.path.join(base, "*.bmp")))
    
    assert 0.0 < val_ratio < 1.0

    sensors = {
        "livdet2011_Biometrika"     : os.path.join(data_root, "livdet2011", "Biometrika"),
        "livdet2011_Digital"        : os.path.join(data_root, "livdet2011", "Digital"),
        "livdet2011_Italdata"       : os.path.join(data_root, "livdet2011", "Italdata"),
        "livdet2011_Sagem"          : os.path.join(data_root, "livdet2011", "Sagem"),
        "livdet2013_Biometrika"     : os.path.join(data_root, "livdet2013", "Biometrika"),
        "livdet2013_CrossMatch"     : os.path.join(data_root, "livdet2013", "CrossMatch"),
        "livdet2013_Italdata"       : os.path.join(data_root, "livdet2013", "Italdata"),
        "livdet2015_CrossMatch"     : os.path.join(data_root, "livdet2015", "CrossMatch"),
        "livdet2015_DigitalPersona" : os.path.join(data_root, "livdet2015", "DigitalPersona"),
        "livdet2015_GreenBit"       : os.path.join(data_root, "livdet2015", "GreenBit"),
        "livdet2015_HiScan"         : os.path.join(data_root, "livdet2015", "HiScan")
    }
    rng     = random.Random(seed)
    unified: dict = {"train": [], "val": [], "train_subjects": 0, "val_subjects": 0}

    for sensor_key, sensor_dir in sensors.items():

        # ── Test: predefined split, kept as-is ───────────────────────────
        unified[f"test_{sensor_key}"] = _collect_live_paths(sensor_dir, "Test")

        # ── Train / Val: subject-level split of Train/Live ────────────────
        train_paths = _collect_live_paths(sensor_dir, "Train")
        if not train_paths:
            continue

        subject_to_paths: dict[str, list[str]] = {}
        for p in train_paths:
            subject_to_paths.setdefault(_extract_key(p, "subject"), []).append(p)

        all_subjects = sorted(subject_to_paths.keys())
        rng.shuffle(all_subjects)

        n_val   = max(1, int(len(all_subjects) * val_ratio))
        n_train = len(all_subjects) - n_val
        val_set = set(all_subjects[n_train:])

        for sid, paths in subject_to_paths.items():
            target = unified["val"] if sid in val_set else unified["train"]
            target.extend(paths)

        unified["train_subjects"] += n_train
        unified["val_subjects"]   += n_val

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    test_keys = [k for k in unified if k.startswith("test_")]
    n_test    = sum(len(unified[k]) for k in test_keys)
    print(
        f"LivDet RECOG splits -> "
        f"train: {len(unified['train'])} imgs ({unified['train_subjects']} subjects), "
        f"val: {len(unified['val'])} imgs ({unified['val_subjects']} subjects), "
        f"test: {n_test} imgs across {len(test_keys)} sensor(s)"
    )
    return unified


def create_LivDet_PAD_splits(
    data_root:        str   = "data/LivDet/",
    recog_json_path:  str   = "data/LivDet/livdet_recog_splits.json",
    output_path:      str   = "data/LivDet/livdet_pad_splits.json",
    val_ratio:        float = 0.2,
    seed:             int   = 42,
) -> dict:
    def _collect_spoof_paths(sensor_dir: str, split: str) -> list[str]:
        base = os.path.join(sensor_dir, split, "Spoof")
        return sorted(glob.glob(os.path.join(base, "*", "*.png")) + glob.glob(os.path.join(base, "*", "*.bmp")))
    
    def _cnt(entries, lbl):
        return sum(1 for e in entries if e["label"] == lbl)

    assert 0.0 < val_ratio < 1.0

    with open(recog_json_path) as f:
        recog = json.load(f)

    # Start from recog structure, wrapping live paths with label 0
    unified: dict = {}
    for key, value in recog.items():
        if isinstance(value, list):
            unified[key] = [{"path": p, "label": 0} for p in value]

    # Add spoof images
    sensors = {
        "livdet2011_Biometrika"     : os.path.join(data_root, "livdet2011", "Biometrika"),
        "livdet2011_Digital"        : os.path.join(data_root, "livdet2011", "Digital"),
        "livdet2011_Italdata"       : os.path.join(data_root, "livdet2011", "Italdata"),
        "livdet2011_Sagem"          : os.path.join(data_root, "livdet2011", "Sagem"),
        "livdet2013_Biometrika"     : os.path.join(data_root, "livdet2013", "Biometrika"),
        "livdet2013_CrossMatch"     : os.path.join(data_root, "livdet2013", "CrossMatch"),
        "livdet2013_Italdata"       : os.path.join(data_root, "livdet2013", "Italdata"),
        "livdet2015_CrossMatch"     : os.path.join(data_root, "livdet2015", "CrossMatch"),
        "livdet2015_DigitalPersona" : os.path.join(data_root, "livdet2015", "DigitalPersona"),
        "livdet2015_GreenBit"       : os.path.join(data_root, "livdet2015", "GreenBit"),
        "livdet2015_HiScan"         : os.path.join(data_root, "livdet2015", "HiScan")
    }
    rng = random.Random(seed)

    for sensor_key, sensor_dir in sensors.items():

        # Test — append spoof from predefined Test/Spoof directly
        test_key = f"test_{sensor_key}"
        for p in _collect_spoof_paths(sensor_dir, "Test"):
            unified[test_key].append({"path": p, "label": 1})

        # Train / Val — file-level split, no subject grouping needed
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
    n_test    = sum(len(unified[k]) for k in test_keys)
    print(
        f"LivDet PAD splits -> "
        f"train: {len(unified['train'])} imgs "
        f"(live={_cnt(unified['train'], 0)}, spoof={_cnt(unified['train'], 1)}), "
        f"val: {len(unified['val'])} imgs "
        f"(live={_cnt(unified['val'], 0)}, spoof={_cnt(unified['val'], 1)}), "
        f"test: {n_test} imgs across {len(test_keys)} sensor(s)"
    )
    return unified


# ──Unify───────────────────────────────────────────
def unify_splits(
    data_root: str = "data/",
    datasets: list[str] =["fvc", "sd302"],
    output_path: str ="data/splits.json",
    split_ratio: tuple[int] =(0.7, 0.15, 0.15),
    seed=42,
):

    json_paths = []

    for dataset in datasets:
        if dataset == "fvc":
            if not os.path.exists(os.path.join(data_root, "FVC/fvc_splits.json")):
                create_FVC_splits(
                    data_root=data_root+"FVC",
                    output_path=data_root+"FVC/fvc_splits.json",
                    split_ratio=split_ratio, 
                    seed=seed
                )
            json_paths.append(data_root+"FVC/fvc_splits.json")

        elif dataset == "sd302":
            if not os.path.exists(os.path.join(data_root, "SD302/sd302_splits.json")):
                create_SD302_splits(
                    data_root=data_root+"SD302",
                    output_path=data_root+"SD302/sd302_splits.json",
                    split_ratio=split_ratio, 
                    seed=seed)
            json_paths.append(data_root+"SD302/sd302_splits.json")

        else:
            raise ValueError(dataset)

    unified = {
        "train": [],
        "val": [],
        "train_subjects": 0,
        "val_subjects": 0
    }

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
        f"Unified splits created → "
        f"train: {len(unified['train'])} images ({unified["train_subjects"]} subjects), "
        f"val: {len(unified['val'])} images ({unified["val_subjects"]} subjects)"
    )

    return unified
    

# ──RecogDatasets───────────────────────────────────────────
class RecogTrainingDataset(Dataset):
    def __init__(
        self,
        json_path: str = "data/splits.json",
        transform=None,
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
        img   = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __repr__(self):
        return (
            f"RecogTrainingDataset(n_ids={len(self.key_to_label)}, n_samples={len(self)})"
        )
    

class RecogEvaluationDataset(Dataset):
    def __init__(
        self,
        json_path: str = "data/splits.json",
        split: str = "val",
        n_genuine_impressions: int = 2,
        n_impostor_impressions: int = 1,
        transform=None,
    ):
        with open(json_path, "r") as f:
            splits = json.load(f)

        assert split in splits, (
            f"Split '{split}' not found in '{json_path}'. "
            f"Available splits: {list(splits.keys())}"
        )

        paths          = splits[split]
        self.transform = transform
        self.split     = split

        id_to_paths: dict[str, list[str]] = {}
        for path in paths:
            key = _extract_key(path, "id")
            id_to_paths.setdefault(key, []).append(path)

        for id in id_to_paths:
            id_to_paths[id].sort()

        min_impressions = min(len(v) for v in id_to_paths.values())     
        assert 2 <= n_genuine_impressions <= min_impressions, (
            f"n_genuine_impressions must be between 2 and {min_impressions}, got {n_genuine_impressions}."
        )
        assert 1 <= n_impostor_impressions <= min_impressions, (
            f"n_impostor_samples must be between 1 and {min_impressions}, got {n_impostor_impressions}."
        )

        self.n_genuine_impressions  = n_genuine_impressions
        self.n_impostor_impressions = n_impostor_impressions
        self.n_ids                  = len(id_to_paths)

        genuine_pairs: list[tuple] = []
        for paths in id_to_paths.values():
            for path_a, path_b in combinations(paths[:n_genuine_impressions], 2):
                genuine_pairs.append((path_a, path_b, 0))

        impostor_pairs: list[tuple] = []
        id_list = list(id_to_paths.values())

        for impression_idx in range(n_impostor_impressions):
            impression_slice = [id_paths[impression_idx] for id_paths in id_list]
            for path_a, path_b in combinations(impression_slice, 2):
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
        n_genuine  = sum(1 for *_, lbl in self.pairs if lbl == 0)
        n_impostor = sum(1 for *_, lbl in self.pairs if lbl == 1)
        return (
            f"RecogEvaluationDataset(split='{self.split}', "
            f"n_ids={self.n_ids}, "
            f"n_genuine_impressions={self.n_genuine_impressions}, "
            f"n_impostor_impressions={self.n_impostor_impressions}, "
            f"n_pairs={len(self)}, "
            f"genuine={n_genuine}, "
            f"impostor={n_impostor})"
        )


# ──PADDatasets───────────────────────────────────────────
class PADDataset(Dataset):
    def __init__(
        self, 
        json_path="data/LivDet/livdet_pad_splits.json", 
        split="train", 
        transform=None
    ):
        self.transform = transform
        self.split     = split

        with open(json_path) as f:
            splits = json.load(f)

        assert split in splits, f"Split '{split}' not found. Available: {list(splits.keys())}"
        
        entries     = splits[split]
        self.paths  = [e["path"]  for e in entries]
        self.labels = [e["label"] for e in entries]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

    def __repr__(self):
        return (f"LivDetPADDataset(split='{self.split}', n_samples={len(self)}, "
                f"live={self.labels.count(0)}, spoof={self.labels.count(1)})")


def create_dataloaders(
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    test_dataset: Dataset = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> "DataLoader | tuple":
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # unify_splits()

    # train_dataset = RecogTrainingDataset(transform=transform)
    # val_dataset = RecogEvaluationDataset(split="val", transform=transform)
    # fvc_test = RecogEvaluationDataset(json_path="data/FVC/fvc_splits.json", split="test_fvc2000_db1", transform=transform)
    
    # print()
    # print(train_dataset)
    # print(val_dataset)
    # print(fvc_test)

    # loaders = create_dataloaders(train_dataset, val_dataset, fvc_test)

    # print()
    # for images, labels in loaders[0]:
    #     print(images.shape, labels.shape)
    #     break
    # for i in (1, 2):
    #     for imageas, imagebs, labels in loaders[i]:
    #         print(imageas.shape, imagebs.shape, labels.shape)
    #         break

    # create_LivDet_recog_splits()
    # create_LivDet_PAD_splits()
    # print()
    # recog_train_dataset = RecogTrainingDataset(json_path="data/LivDet/livdet_recog_splits.json", transform=transform)
    # recog_val_dataset = RecogEvaluationDataset(json_path="data/LivDet/livdet_recog_splits.json", split="val", transform=transform)
    # recog_test_dataset = RecogEvaluationDataset(json_path="data/LivDet/livdet_recog_splits.json", split="test_livdet2015_HiScan", transform=transform)
    # pad_train_dataset = PADDataset(json_path="data/LivDet/livdet_pad_splits.json", split="train", transform=transform)
    # pad_val_dataset = PADDataset(json_path="data/LivDet/livdet_pad_splits.json", split="val", transform=transform)
    # pad_test_dataset = PADDataset(json_path="data/LivDet/livdet_pad_splits.json", split="test_livdet2015_HiScan", transform=transform)
    # for dataset in [recog_train_dataset, recog_val_dataset, recog_test_dataset, 
    #                  pad_train_dataset, pad_val_dataset, pad_test_dataset]:
    #     print(dataset)
    #     print()