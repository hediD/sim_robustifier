# datasets_local.py
from typing import List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import torch

class FinetuneDataset(Dataset):
    """
    Expects a list of tuples: (image_path, class_id).
    Uses a huggingface image processor to produce pixel_values.
    """
    def __init__(self, items: List[Tuple[str, int]], processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        # pixel_values: [1, 3, H, W] -> squeeze to [3, H, W] so DataLoader can stack
        pv = inputs["pixel_values"].squeeze(0)
        return pv, torch.tensor(label, dtype=torch.long)
