import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

# ==========================================================
#  DATASET (GTAV paired clean/adverse) + METADATA
# ==========================================================

class PairedImageDataset(Dataset):
    """
    Returns a dict with:
      - "pair": Tensor [2, C, H, W] where pair[0]=clean (Day-EXTRASUNNY) and pair[1]=selected adverse
      - "key": location key (prefix before first "_")
      - "main_path": path to clean image
      - "sec_path": path to adverse image
      - "main_meta": parsed metadata from main_path
      - "sec_meta": parsed metadata from sec_path
      - "mode": the sampling mode used
    """

    def __init__(self, data_dir, mode="random", transform=None):
        self.data_dir = data_dir
        self.mode = mode.lower()
        self.transform = transform if transform else transforms.ToTensor()

        self.day_dir = os.path.join(data_dir, "Day")
        self.sunset_dir = os.path.join(data_dir, "Sunset")
        self.night_dir = os.path.join(data_dir, "Night")

        self.image_pairs = self._group_images()

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _parse_meta_from_path(p: str):
        """
        Best-effort parser based on your naming convention and folder structure.

        Expected:
          folder contains Day/Sunset/Night
          filename contains weather token among: EXTRASUNNY, RAIN, FOGGY
          location key is prefix before first "_"
        """
        fname = os.path.basename(p)
        key = fname.split("_")[0]

        # illumination from folder name if possible
        illum = None
        parts = os.path.normpath(p).split(os.sep)
        for cand in ("Day", "Sunset", "Night"):
            if cand in parts:
                illum = cand
                break

        # weather from filename
        weather = None
        for w in ("EXTRASUNNY", "RAIN", "FOGGY"):
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

    def _group_images(self):
        all_images = {"Day": {}, "Sunset": {}, "Night": {}}

        def collect(folder, category):
            if not os.path.exists(folder):
                return
            for f in sorted(os.listdir(folder)):
                if f.endswith((".jpg", ".png", ".jpeg")):
                    key = f.split("_")[0]
                    all_images[category].setdefault(key, []).append(os.path.join(folder, f))

        collect(self.day_dir, "Day")
        collect(self.sunset_dir, "Sunset")
        collect(self.night_dir, "Night")

        pairs = []
        for key, imgs in all_images["Day"].items():
            # clean/base is Day_EXTRASUNNY
            base = [x for x in imgs if "Day_EXTRASUNNY" in os.path.basename(x)]
            if not base:
                continue

            others = (
                all_images["Day"].get(key, [])
                + all_images["Sunset"].get(key, [])
                + all_images["Night"].get(key, [])
            )

            for b in base:
                pairs.append((b, others, key))

        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        main_path, others, key = self.image_pairs[idx]

        # illumination selection
        if self.mode.startswith("d"):
            cat = "Day"
        elif self.mode.startswith("s"):
            cat = "Sunset"
        elif self.mode.startswith("n"):
            cat = "Night"
        else:
            cat = random.choice(["Day", "Sunset", "Night"])

        # weather selection (optional second char)
        # f=FOGGY, r=RAIN, s=EXTRASUNNY
        if len(self.mode) > 1:
            w = self.mode[1]
            weather = {"f": "FOGGY", "r": "RAIN", "s": "EXTRASUNNY"}.get(w)
        else:
            weather = None

        matches = [x for x in others if os.sep + cat + os.sep in x or f"{os.sep}{cat}{os.sep}" in x]
        if weather:
            matches = [x for x in matches if weather in os.path.basename(x)]

        if not matches:
            matches = others

        sel = random.choice(matches)
        while sel == main_path:
            sel = random.choice(matches)

        main_img = Image.open(main_path).convert("RGB")
        sec_img = Image.open(sel).convert("RGB")

        pair = torch.stack([self.transform(main_img), self.transform(sec_img)])  # [2,C,H,W]

        sample = {
            "pair": pair,
            "key": key,
            "main_path": main_path,
            "sec_path": sel,
            "main_meta": self._parse_meta_from_path(main_path),
            "sec_meta": self._parse_meta_from_path(sel),
            "mode": self.mode,
        }
        return sample
