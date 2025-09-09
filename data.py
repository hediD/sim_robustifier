# data.py
import os
import glob
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from utils import ensure_dir
from constants import REAL_FOLDER

def collect_sim_images(data_root: str, sim_data_folder: str, sim_train_subfolders: List[str], total_images: int) -> List[str]:
    """Collect TOTAL_SIM_IMAGES across specified sim training subfolders with near-even split."""
    sim_train_paths: List[str] = []
    per_folder = total_images // len(sim_train_subfolders)
    rem = total_images % len(sim_train_subfolders)

    print(f"Collecting training images from {sim_data_folder} sim folders...")
    for i, subfolder in enumerate(sim_train_subfolders):
        folder_glob = os.path.join(data_root, sim_data_folder, subfolder, "*")
        images = sorted(glob.glob(folder_glob))
        target = per_folder + (1 if i < rem else 0)
        selected = images[:target] if len(images) >= target else images
        sim_train_paths.extend(selected)
        print(f"  {subfolder}: {len(selected)} images")
    return sim_train_paths

def collect_eval_paths(data_root: str, sim_data_folder: str, sim_eval_subfolder: str) -> Tuple[List[str], List[str]]:
    """Collect evaluation paths for both sim and real images."""
    sim_eval_paths = sorted(glob.glob(os.path.join(data_root, sim_data_folder, sim_eval_subfolder, "*")))
    real_image_paths = sorted(glob.glob(os.path.join(data_root, REAL_FOLDER, "*")))
    return sim_eval_paths, real_image_paths

def load_imagenet_datasets():
    """Return (train, validation) datasets from 'imagenet-1k'."""
    print("Loading ImageNet training & validation datasets...")
    train_dataset = load_dataset("imagenet-1k", split="train")
    val_dataset = load_dataset("imagenet-1k", split="validation")
    return train_dataset, val_dataset

def check_imagenet_availability_for_classes(
    train_dataset,
    val_dataset,
    class_ids: List[int],
) -> Dict[int, Dict[str, int]]:
    """
    Check how many samples are available in ImageNet for each class.
    Returns: {class_id: {"train": count, "val": count}}
    """
    print("Checking ImageNet availability for target classes...")
    train_labels = train_dataset["label"]
    val_labels = val_dataset["label"]
    class_id_set = set(class_ids)

    # Count availability per class
    availability = {cid: {"train": 0, "val": 0} for cid in class_ids}

    print("Counting TRAIN samples...")
    for label in tqdm(train_labels):
        if label in class_id_set:
            availability[label]["train"] += 1

    print("Counting VAL samples...")
    for label in tqdm(val_labels):
        if label in class_id_set:
            availability[label]["val"] += 1

    print("\nImageNet availability:")
    for cid in class_ids:
        train_count = availability[cid]["train"]
        val_count = availability[cid]["val"]
        print(f"  Class {cid}: {train_count} train, {val_count} val samples")

    return availability

def collect_imagenet_samples_for_classes(
    imagenet_save_dir: str,
    train_dataset,
    val_dataset,
    class_ids: List[int],
    targets_per_class: Dict[int, int],
    max_val_per_class: int,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Save selected images (train & val) per class_id as JPEGs to imagenet_save_dir
    and return dicts mapping class_id -> list[path].
    """
    ensure_dir(imagenet_save_dir)
    train_labels = train_dataset["label"]
    val_labels = val_dataset["label"]
    class_id_set = set(class_ids)

    print("Indexing ImageNet TRAIN for target classes...")
    train_indices = [i for i in tqdm(range(len(train_labels))) if train_labels[i] in class_id_set]
    ftrain = train_dataset.select(train_indices)

    print("Indexing ImageNet VAL for target classes...")
    val_indices = [i for i in tqdm(range(len(val_labels))) if val_labels[i] in class_id_set]
    fval = val_dataset.select(val_indices)

    class_train_paths: Dict[int, List[str]] = {cid: [] for cid in class_ids}
    class_val_paths: Dict[int, List[str]] = {cid: [] for cid in class_ids}

    # TRAIN
    print("Collecting ImageNet TRAIN samples...")
    # for speed, group by class
    train_by_class: Dict[int, List[int]] = {cid: [] for cid in class_ids}
    for i in range(len(ftrain)):
        cid = ftrain[i]["label"]
        if cid in train_by_class:
            train_by_class[cid].append(i)

    for cid in class_ids:
        target = targets_per_class[cid]
        os.makedirs(os.path.join(imagenet_save_dir, f"class_{cid:03d}_train"), exist_ok=True)
        count = 0
        for local_idx in train_by_class[cid][:target]:
            try:
                sample = ftrain[local_idx]
                img: Image.Image = sample["image"]
                path = os.path.join(imagenet_save_dir, f"class_{cid:03d}_train", f"sample_{count:05d}.jpg")
                img.save(path)
                class_train_paths[cid].append(path)
                count += 1
            except Exception:
                continue
        print(f"  TRAIN class {cid}: collected {len(class_train_paths[cid])}/{target}")

    # VAL
    print("Collecting ImageNet VAL samples...")
    val_by_class: Dict[int, List[int]] = {cid: [] for cid in class_ids}
    for i in range(len(fval)):
        cid = fval[i]["label"]
        if cid in val_by_class:
            val_by_class[cid].append(i)

    for cid in class_ids:
        os.makedirs(os.path.join(imagenet_save_dir, f"class_{cid:03d}_val"), exist_ok=True)
        count = 0
        for local_idx in val_by_class[cid][:max_val_per_class]:
            try:
                sample = fval[local_idx]
                img: Image.Image = sample["image"]
                path = os.path.join(imagenet_save_dir, f"class_{cid:03d}_val", f"val_{count:05d}.jpg")
                img.save(path)
                class_val_paths[cid].append(path)
                count += 1
            except Exception:
                continue
        print(f"  VAL class {cid}: collected {len(class_val_paths[cid])}/{max_val_per_class}")

    return class_train_paths, class_val_paths
