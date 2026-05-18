"""
Mixed-input dataloader for the photometric robustness benchmark.

The dataset returns two versions of each clean/adverse pair:

    pair_raw:
        RGB tensors in [0, 1].
        Use this for HuggingFace models such as SegFormer and Mask2Former,
        which should keep their processor-based preprocessing.

    pair_repo:
        RGB tensors normalized with ImageNet mean/std.
        Use this for the local models taken from the SpatialRobustnessBench-style
        repository, e.g. DeepLabV3+, PIDNet, and DDRNet.

This avoids forcing a single preprocessing convention on all models.
"""

import os
import random
from typing import List

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PairedImageDataset(Dataset):
    """
    Paired clean/adverse dataset for the GTA-based photometric benchmark.

    Expected structure:
        data_dir/
            Day/
            Sunset/
            Night/

    Clean/reference image:
        Day folder, filename containing "Day_EXTRASUNNY"

    Adverse image selection:
        mode="random" -> random adverse condition
        mode="df"     -> Day-FOGGY
        mode="dr"     -> Day-RAIN
        mode="ss"     -> Sunset-EXTRASUNNY
        mode="sf"     -> Sunset-FOGGY
        mode="sr"     -> Sunset-RAIN
        mode="ns"     -> Night-EXTRASUNNY
        mode="nf"     -> Night-FOGGY
        mode="nr"     -> Night-RAIN
    """

    def __init__(self, data_dir, mode: str = "random"):
        self.data_dir = str(data_dir)
        self.mode = str(mode).lower()

        self.raw_transform = transforms.ToTensor()

        self.repo_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.day_dir = os.path.join(self.data_dir, "Day")
        self.sunset_dir = os.path.join(self.data_dir, "Sunset")
        self.night_dir = os.path.join(self.data_dir, "Night")

        self.image_pairs = self._group_images()

    def _group_images(self):
        all_images = {"Day": {}, "Sunset": {}, "Night": {}}

        def collect(folder, category):
            if not os.path.exists(folder):
                return

            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    loc_key = fname.split("_")[0]
                    all_images[category].setdefault(loc_key, []).append(
                        os.path.join(folder, fname)
                    )

        collect(self.day_dir, "Day")
        collect(self.sunset_dir, "Sunset")
        collect(self.night_dir, "Night")

        pairs = []
        for loc_key, day_imgs in all_images["Day"].items():
            clean_imgs = [
                p for p in day_imgs
                if "Day_EXTRASUNNY" in os.path.basename(p)
            ]

            if not clean_imgs:
                continue

            all_loc_variants = (
                all_images["Day"].get(loc_key, [])
                + all_images["Sunset"].get(loc_key, [])
                + all_images["Night"].get(loc_key, [])
            )

            for clean_path in clean_imgs:
                pairs.append((loc_key, clean_path, all_loc_variants))

        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def _select_adverse_path(self, main_path: str, others: List[str]) -> str:
        if self.mode.startswith("d"):
            category = "Day"
        elif self.mode.startswith("s"):
            category = "Sunset"
        elif self.mode.startswith("n"):
            category = "Night"
        else:
            category = random.choice(["Day", "Sunset", "Night"])

        weather = None
        if len(self.mode) > 1:
            weather = {
                "f": "FOGGY",
                "r": "RAIN",
                "s": "EXTRASUNNY",
            }.get(self.mode[1])

        matches = [
            p for p in others
            if f"/{category}/" in p.replace("\\", "/")
        ]

        if weather is not None:
            matches = [
                p for p in matches
                if weather in os.path.basename(p)
            ]

        if not matches:
            matches = list(others)

        sec_path = random.choice(matches)
        while sec_path == main_path and len(matches) > 1:
            sec_path = random.choice(matches)

        return sec_path

    @staticmethod
    def _condition_from_path(path: str) -> str:
        basename = os.path.basename(path)
        parts = basename.split("_")
        sec_illum = parts[1] if len(parts) > 1 else "UNK"
        sec_weather = parts[2].split(".")[0] if len(parts) > 2 else "UNK"
        return f"{sec_illum}-{sec_weather}"

    def __getitem__(self, idx):
        loc_id, main_path, others = self.image_pairs[idx]

        sec_path = self._select_adverse_path(main_path, others)

        main_img = Image.open(main_path).convert("RGB")
        sec_img = Image.open(sec_path).convert("RGB")

        main_raw = self.raw_transform(main_img)
        sec_raw = self.raw_transform(sec_img)

        main_repo = self.repo_transform(main_img)
        sec_repo = self.repo_transform(sec_img)

        return {
            "pair_raw": torch.stack([main_raw, sec_raw], dim=0),
            "pair_repo": torch.stack([main_repo, sec_repo], dim=0),
            "loc_id": loc_id,
            "sec_condition": self._condition_from_path(sec_path),
            "clean_path": main_path,
            "sec_path": sec_path,
        }
