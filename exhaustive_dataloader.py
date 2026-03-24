import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

WEATHER_TOKENS = ("EXTRASUNNY", "RAIN", "FOGGY")
ILLUM_TOKENS = ("Day", "Sunset", "Night")


class ExhaustivePairedImageDataset(Dataset):
    """
    Generates ALL pairs (clean -> adverse) for each location key.
    Clean is Day_EXTRASUNNY.
    Adverse covers all other available conditions for that key (excluding the exact clean image).

    Returns dict:
      - pair: [2,C,H,W]
      - key
      - main_path, sec_path
      - main_meta, sec_meta (illum/weather/condition)
    """

    def __init__(self, data_dir: str, transform=None, include_day_sunny_as_adverse: bool = False):
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.ToTensor()

        self.day_dir = os.path.join(data_dir, "Day")
        self.sunset_dir = os.path.join(data_dir, "Sunset")
        self.night_dir = os.path.join(data_dir, "Night")

        self.index = self._build_index(include_day_sunny_as_adverse)

    @staticmethod
    def _parse_meta_from_path(p: str):
        fname = os.path.basename(p)
        key = fname.split("_")[0]

        illum = None
        parts = os.path.normpath(p).split(os.sep)
        for cand in ILLUM_TOKENS:
            if cand in parts:
                illum = cand
                break

        weather = None
        for w in WEATHER_TOKENS:
            if w in fname:
                weather = w
                break

        condition = f"{illum}-{weather}" if (illum and weather) else None
        return {
            "key": key,
            "illumination": illum,
            "weather": weather,
            "condition": condition,
            "filename": fname,
            "path": p,
        }

    def _collect(self, folder: str):
        if not os.path.exists(folder):
            return []
        out = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(os.path.join(folder, f))
        return out

    def _build_index(self, include_day_sunny_as_adverse: bool):
        all_paths = self._collect(self.day_dir) + self._collect(self.sunset_dir) + self._collect(self.night_dir)

        by_key = {}
        for p in all_paths:
            key = os.path.basename(p).split("_")[0]
            by_key.setdefault(key, []).append(p)

        index = []
        for key, paths in by_key.items():
            clean_candidates = [
                p for p in paths
                if ("Day" in os.path.normpath(p).split(os.sep)) and ("EXTRASUNNY" in os.path.basename(p))
            ]
            if not clean_candidates:
                continue

            for clean_path in clean_candidates:
                for sec_path in paths:
                    if sec_path == clean_path:
                        continue
                    if not include_day_sunny_as_adverse:
                        if ("Day" in os.path.normpath(sec_path).split(os.sep)) and ("EXTRASUNNY" in os.path.basename(sec_path)):
                            continue
                    index.append((clean_path, sec_path))

        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        main_path, sec_path = self.index[idx]

        main_img = Image.open(main_path).convert("RGB")
        sec_img = Image.open(sec_path).convert("RGB")

        pair = torch.stack([self.transform(main_img), self.transform(sec_img)])  # [2,C,H,W]

        main_meta = self._parse_meta_from_path(main_path)
        sec_meta = self._parse_meta_from_path(sec_path)

        return {
            "pair": pair,
            "key": main_meta["key"],
            "main_path": main_path,
            "sec_path": sec_path,
            "main_meta": main_meta,
            "sec_meta": sec_meta,
        }
