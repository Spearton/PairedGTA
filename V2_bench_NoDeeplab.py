import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd

from huggingface_hub import hf_hub_download
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ==========================================================
# USER CONFIG
# ==========================================================
DATA_DIR = "/home/ace/Downloads/Dataset"  # <-- CHANGE
SAVE_DIR = "./out_cityscapes_final"         # <-- CHANGE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
NUM_WORKERS = 4

MASS_COVERAGE = 0.99   # keep classes that cover 99% of masked clean mass (scenario-level)
MASS_MIN_ABS_FALLBACK = 1.0  # safety (avoid zero-total)

CONF_THRESHOLD = 0.6     # masked agreement/retention computed only on pixels with clean conf >= this
MASS_MIN_PLOT = 1000     # minimum masked pixel mass to include class in plots (avoid noisy tiny classes)
N_QUAL_SAMPLES = 6       # qualitative samples per scenario
QUAL_SEGFORMER_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"  # for visualization

# Only 3 scenario plots requested:
PLOT_SCENARIOS = ["Day-RAIN", "Sunset-FOGGY", "Night-RAIN"]
PLOT_DROP_CLASSNAME = "train"  # remove only from plots (still in CSV)

ENABLE_MASK2FORMER = True  # set False if you don't want mask2former; if scipy missing we auto-skip

# ==========================================================
# Cityscapes palette (19 classes) — standard ordering
# ==========================================================
CITYSCAPES_PALETTE_19 = np.array([
    (128,  64, 128),  # road
    (244,  35, 232),  # sidewalk
    ( 70,  70,  70),  # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170,  30),  # traffic light
    (220, 220,   0),  # traffic sign
    (107, 142,  35),  # vegetation
    (152, 251, 152),  # terrain
    ( 70, 130, 180),  # sky
    (220,  20,  60),  # person
    (255,   0,   0),  # rider
    (  0,   0, 142),  # car
    (  0,   0,  70),  # truck
    (  0,  60, 100),  # bus
    (  0,  80, 100),  # train
    (  0,   0, 230),  # motorcycle
    (119,  11,  32),  # bicycle
], dtype=np.uint8)

# ==========================================================
# Dataset (your paired loader, with "mode" to select scenario)
# ==========================================================
class PairedImageDataset(Dataset):
    """
    mode examples:
      - "random": random adverse across all illum/weather
      - "dr": Day-RAIN
      - "sf": Sunset-FOGGY
      - "nr": Night-RAIN
    """
    def __init__(self, data_dir, mode="random", transform=None):
        self.data_dir = data_dir
        self.mode = mode.lower()
        self.transform = transform if transform else transforms.ToTensor()

        self.day_dir = os.path.join(data_dir, "Day")
        self.sunset_dir = os.path.join(data_dir, "Sunset")
        self.night_dir = os.path.join(data_dir, "Night")
        self.image_pairs = self._group_images()

    def _group_images(self):
        all_images = {"Day": {}, "Sunset": {}, "Night": {}}

        def collect(folder, category):
            if not os.path.exists(folder):
                return
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    key = f.split("_")[0]
                    all_images[category].setdefault(key, []).append(os.path.join(folder, f))

        collect(self.day_dir, "Day")
        collect(self.sunset_dir, "Sunset")
        collect(self.night_dir, "Night")

        pairs = []
        for key, imgs in all_images["Day"].items():
            base = [x for x in imgs if "Day_EXTRASUNNY" in os.path.basename(x)]
            if not base:
                continue
            others = all_images["Day"].get(key, []) + all_images["Sunset"].get(key, []) + all_images["Night"].get(key, [])
            for b in base:
                pairs.append((key, b, others))
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        loc_id, main_path, others = self.image_pairs[idx]

        # illumination
        if self.mode.startswith("d"):
            cat = "Day"
        elif self.mode.startswith("s"):
            cat = "Sunset"
        elif self.mode.startswith("n"):
            cat = "Night"
        else:
            cat = random.choice(["Day", "Sunset", "Night"])

        # weather
        weather = None
        if len(self.mode) > 1:
            w = self.mode[1]
            weather = {"f": "FOGGY", "r": "RAIN", "s": "EXTRASUNNY"}.get(w)

        matches = [x for x in others if f"/{cat}/" in x.replace("\\", "/")]
        if weather:
            matches = [x for x in matches if weather in os.path.basename(x)]
        if not matches:
            matches = others

        sec_path = random.choice(matches)
        while sec_path == main_path and len(matches) > 1:
            sec_path = random.choice(matches)

        main = Image.open(main_path).convert("RGB")
        sec = Image.open(sec_path).convert("RGB")

        main_t = self.transform(main)
        sec_t = self.transform(sec)

        # parse condition from filename
        b = os.path.basename(sec_path)
        parts = b.split("_")
        sec_illum = parts[1] if len(parts) > 1 else "UNK"
        sec_weather = parts[2].split(".")[0] if len(parts) > 2 else "UNK"
        condition = f"{sec_illum}-{sec_weather}"

        return {
            "pair": torch.stack([main_t, sec_t], dim=0),  # [2,3,H,W]
            "loc_id": loc_id,
            "sec_condition": condition,
            "clean_path": main_path,
            "sec_path": sec_path,
        }


# ==========================================================
# Metrics helpers
# ==========================================================
def pixel_agreement_masked(pred_a: torch.Tensor, pred_b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    same = (pred_a == pred_b) & mask
    denom = mask.flatten(1).sum(dim=1).clamp_min(1)
    num = same.flatten(1).sum(dim=1)
    return (num.double() / denom.double())

def batch_counts_by_condition(pred_clean, pred_adv, conf_mask, conditions, num_classes: int):
    """
    Vectorized per-condition counts in a batch.
    Returns dict cond -> (same_counts[K] CPU float64, mass_counts[K] CPU float64)
    """
    idx_by_cond = defaultdict(list)
    for i, c in enumerate(conditions):
        idx_by_cond[c].append(i)

    out = {}
    for c, idxs in idx_by_cond.items():
        idxs_t = torch.tensor(idxs, device=pred_clean.device, dtype=torch.long)

        a = pred_clean.index_select(0, idxs_t)  # [b',H,W]
        b = pred_adv.index_select(0, idxs_t)
        m = conf_mask.index_select(0, idxs_t)

        a_f = a[m]
        b_f = b[m]

        valid = (a_f >= 0) & (a_f < num_classes) & (b_f >= 0) & (b_f < num_classes)
        a_f = a_f[valid]
        b_f = b_f[valid]

        mass = torch.bincount(a_f, minlength=num_classes).to(torch.float64)
        same = torch.bincount(a_f[a_f == b_f], minlength=num_classes).to(torch.float64)

        out[c] = (same.detach().to("cpu"), mass.detach().to("cpu"))
    return out


# ==========================================================
# Model adapters
# ==========================================================
class BaseAdapter:
    def __init__(self, name: str):
        self.name = name
    def num_classes(self) -> int:
        raise NotImplementedError
    def id2label(self) -> Optional[Dict[int, str]]:
        return None
    @torch.no_grad()
    def predict(self, images_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images_01: [B,3,H,W] float in [0,1]
        returns pred [B,H,W] int64, conf [B,H,W] float in [0,1]
        """
        raise NotImplementedError

class SegFormerAdapter(BaseAdapter):
    def __init__(self, name: str, model_id: str, device: torch.device):
        super().__init__(name)
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
        self._num = int(self.model.config.num_labels)
        self._id2label = dict(self.model.config.id2label)

    def num_classes(self) -> int:
        return self._num

    def id2label(self):
        return self._id2label

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        # images already [0,1] -> disable rescale
        inputs = self.processor(images=images_01, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        logits = out.logits
        B, _, H, W = images_01.shape
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf

class DeepLabV3CityscapesAdapter(BaseAdapter):
    """
    Torchvision DeepLabV3-ResNet50, weights from HF repo (or local). Robust loading:
    - tries strict load
    - if fails, filters checkpoint keeping only matching keys+shapes
    """

    def __init__(self, device: torch.device, repo_id: str = "Koushim/deeplabv3-resnet50-cityscapes"):
        super().__init__("DeepLabV3-R50 (Cityscapes)")
        self.device = device

        # instantiate model architecture with 19 classes
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            num_classes=19,
        ).to(device).eval()

        # download / locate weights from HF hub (or local path)
        try:
            weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        except Exception as e:
            raise RuntimeError(f"Could not download weights from HF hub ({repo_id}): {e}")

        sd = torch.load(weights_path, map_location="cpu")

        # Try strict load first (will raise if mismatch)
        try:
            self.model.load_state_dict(sd, strict=True)
            print("[info] DeepLabV3: loaded checkpoint with strict=True")
        except Exception as e_strict:
            print("[warn] strict load failed for DeepLabV3 (expected on some checkpoints).")
            print("       Attempting filtered load (only matching keys+shapes).")
            # Model state dict (target) and checkpoint keys
            model_sd = self.model.state_dict()
            kept = {}
            skipped_ckpt_keys = []
            mismatched_shapes = []

            for k, v in sd.items():
                if k in model_sd:
                    if v.shape == model_sd[k].shape:
                        kept[k] = v
                    else:
                        mismatched_shapes.append((k, v.shape, model_sd[k].shape))
                else:
                    skipped_ckpt_keys.append(k)

            # Load the kept dict non-strictly
            missing, unexpected = self.model.load_state_dict(kept, strict=False)
            print("[info] DeepLabV3 missing keys count:", len(missing))

            print(f"[info] DeepLabV3: loaded {len(kept)} tensors from checkpoint.")
            if len(skipped_ckpt_keys) > 0:
                print(f"[info] Checkpoint keys skipped (not present in model): {len(skipped_ckpt_keys)} (examples: {skipped_ckpt_keys[:5]})")
            if len(mismatched_shapes) > 0:
                print(f"[info] Keys with mismatched shapes: {len(mismatched_shapes)} (examples: {mismatched_shapes[:5]})")
            if missing:
                print(f"[info] Model keys missing after filtered load: {list(missing)[:8]} ... (total {len(missing)})")
            if unexpected:
                print(f"[info] Unexpected keys when loading filtered dict: {list(unexpected)[:8]} ... (total {len(unexpected)})")

            print("[info] DeepLabV3: continuing with partially loaded weights. Remaining parameters are left at model defaults.")


        # bookkeeping
        self._num = 19
        self._id2label = None

        # normalization used at inference time (torchvision style)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def num_classes(self) -> int:
        return self._num

    def id2label(self):
        return self._id2label

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        x = images_01.to(self.device)  # checkpoint expects [0,1] RGB without ImageNet norm
        out = self.model(x)["out"]  # [B,19,H,W]
        probs = F.softmax(out, dim=1)
        conf = probs.max(dim=1).values
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf

def try_build_mask2former(device: torch.device) -> List[BaseAdapter]:
    adapters = []
    if not ENABLE_MASK2FORMER:
        return adapters
    try:
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    except Exception as e:
        print(f"[warn] Mask2Former not available: {e}")
        return adapters
    try:
        import scipy  # noqa: F401
    except Exception as e:
        print(f"[warn] Skipping Mask2Former (scipy missing): {e}")
        return adapters

    class Mask2FormerAdapter(BaseAdapter):
        def __init__(self, name: str, model_id: str, device: torch.device):
            super().__init__(name)
            self.device = device
            self.processor = Mask2FormerImageProcessor.from_pretrained(model_id, use_fast=True)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()
            self._num = int(self.model.config.num_labels)
            self._id2label = dict(self.model.config.id2label)

        def num_classes(self) -> int:
            return self._num

        def id2label(self):
            return self._id2label

        @torch.no_grad()
        def predict(self, images_01: torch.Tensor):
            """
            images_01: [B,3,H,W] in [0,1]
            returns:
              pred: [B,H,W] int64
              conf: [B,H,W] float in [0,1]
            """
            B, _, H, W = images_01.shape

            # Safer HF path: pass a list of CPU tensors to the processor
            images_cpu = [img.detach().cpu() for img in images_01]

            inputs = self.processor(
                images=images_cpu,
                return_tensors="pt",
                do_rescale=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            # Raw query outputs
            class_logits = outputs.class_queries_logits  # [B, Q, C+1]
            mask_logits = outputs.masks_queries_logits  # [B, Q, h, w]

            # Drop the "no-object" class
            class_probs = F.softmax(class_logits, dim=-1)[..., :-1]  # [B, Q, C]
            mask_probs = torch.sigmoid(mask_logits)  # [B, Q, h, w]

            # Upsample masks to the original image size
            mask_probs = F.interpolate(
                mask_probs,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # [B, Q, H, W]

            # Build semantic scores per pixel
            # sem_scores[b, c, h, w] = sum_q class_probs[b,q,c] * mask_probs[b,q,h,w]
            sem_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

            # Normalize across classes to obtain per-pixel probabilities
            probs = sem_scores / sem_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

            conf = probs.max(dim=1).values  # [B, H, W]
            pred = probs.argmax(dim=1).to(torch.int64)  # [B, H, W]

            return pred, conf

    adapters.append(Mask2FormerAdapter("Mask2Former-Swin-S", "facebook/mask2former-swin-small-cityscapes-semantic", device))
    adapters.append(Mask2FormerAdapter("Mask2Former-Swin-L", "facebook/mask2former-swin-large-cityscapes-semantic", device))
    return adapters


# ==========================================================
# Visualization helpers
# ==========================================================
def select_classes_by_mass_coverage(df_s: pd.DataFrame, coverage: float, drop_class: str = None) -> List[str]:
    """
    df_s: rows for a SINGLE scenario (all models), must include columns:
      - class_name
      - mass_clean_masked
    coverage: e.g., 0.99 to keep classes covering 99% of total mass
    drop_class: optional class name to drop (e.g., "train")

    Returns: list of class_name to keep (ordered by mass desc).
    """
    df = df_s.copy()

    if drop_class is not None:
        df = df[df["class_name"] != drop_class]

    # sum masses across models to get a scenario-level notion of "important classes"
    mass_by_class = df.groupby("class_name")["mass_clean_masked"].sum().sort_values(ascending=False)

    total = float(mass_by_class.sum())
    if total < 1e-9:
        return []

    cum = mass_by_class.cumsum() / total
    keep = mass_by_class.index[cum <= coverage].tolist()

    # ensure at least 1 class and always include the class that crosses the threshold
    if len(keep) == 0:
        keep = [mass_by_class.index[0]]
    else:
        # include first class where cum > coverage as well (to reach coverage)
        if cum.iloc[len(keep)-1] < coverage and len(keep) < len(mass_by_class):
            keep.append(mass_by_class.index[len(keep)])

    return keep

def decode_cityscapes_mask(mask_hw: np.ndarray) -> np.ndarray:
    """
    mask_hw: [H,W] values 0..18
    returns RGB image [H,W,3]
    """
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(19):
        out[mask_hw == k] = CITYSCAPES_PALETTE_19[k]
    return out

def tensor_to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W] float in [0,1]
    """
    x = (x.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return x


# ==========================================================
# MAIN EVAL
# ==========================================================
@torch.no_grad()
def evaluate_all_models():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    # Models
    adapters: List[BaseAdapter] = [
        SegFormerAdapter("SegFormer-B0", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", device),
        SegFormerAdapter("SegFormer-B2", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024", device),
        SegFormerAdapter("SegFormer-B5", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", device),
    ]
    adapters += try_build_mask2former(device)

    # Canonical labels (from first segformer)
    id2label = adapters[0].id2label() or {i: str(i) for i in range(adapters[0].num_classes())}
    K = len(id2label)
    class_names = [id2label[i] for i in range(K)]

    # Dataset for full evaluation (random mode; covers all conditions due to internal sampling)
    ds = PairedImageDataset(DATA_DIR, mode="random", transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Aggregators
    rows_img = []
    cond_sum = {a.name: defaultdict(float) for a in adapters}
    cond_n = {a.name: defaultdict(int) for a in adapters}

    cond_same = {a.name: defaultdict(lambda: torch.zeros(K, dtype=torch.float64)) for a in adapters}
    cond_mass = {a.name: defaultdict(lambda: torch.zeros(K, dtype=torch.float64)) for a in adapters}

    global_same = {a.name: torch.zeros(K, dtype=torch.float64) for a in adapters}
    global_mass = {a.name: torch.zeros(K, dtype=torch.float64) for a in adapters}

    for batch in tqdm(loader, desc="Evaluating"):
        pair = batch["pair"]                 # [B,2,3,H,W]
        clean = pair[:, 0].to(device, non_blocking=True)
        adv = pair[:, 1].to(device, non_blocking=True)

        loc_ids = batch["loc_id"]
        conds = [c if c is not None else "UNKNOWN" for c in batch["sec_condition"]]

        for adp in adapters:
            pred_clean, conf_clean = adp.predict(clean)
            pred_adv, _ = adp.predict(adv)

            conf_mask = conf_clean >= CONF_THRESHOLD
            agree = pixel_agreement_masked(pred_clean, pred_adv, conf_mask)  # [B]

            # per-condition class counts (micro-optimized)
            counts = batch_counts_by_condition(pred_clean, pred_adv, conf_mask, conds, num_classes=K)
            for c, (same_c, mass_c) in counts.items():
                cond_same[adp.name][c] += same_c
                cond_mass[adp.name][c] += mass_c
                global_same[adp.name] += same_c
                global_mass[adp.name] += mass_c

            for i in range(clean.shape[0]):
                c = conds[i]
                v = float(agree[i].item())
                cond_sum[adp.name][c] += v
                cond_n[adp.name][c] += 1
                coverage = float(conf_mask[i].float().mean().item())
                rows_img.append({
                    "model": adp.name,
                    "loc_id": loc_ids[i],
                    "sec_condition": c,
                    "pixel_agreement_masked": v,
                    "conf_threshold": CONF_THRESHOLD,
                    "coverage_masked": coverage,
                    "clean_path": batch["clean_path"][i],
                    "sec_path": batch["sec_path"][i],
                })

    # Save per-image
    df_img = pd.DataFrame(rows_img)
    df_img.to_csv(os.path.join(SAVE_DIR, "results_pixel_agreement_per_image.csv"), index=False)

    # Save per-condition agreement
    rows_cond = []
    for m in cond_sum:
        for c in cond_sum[m]:
            n = cond_n[m][c]
            rows_cond.append({
                "model": m,
                "sec_condition": c,
                "pixel_agreement_masked_mean": cond_sum[m][c] / max(n, 1),
                "n_pairs": int(n),
                "conf_threshold": CONF_THRESHOLD,
            })
    df_cond = pd.DataFrame(rows_cond)
    df_cond.to_csv(os.path.join(SAVE_DIR, "results_pixel_agreement_by_condition.csv"), index=False)

    # Save class retention by condition
    rows_cls_cond = []
    for m in cond_same:
        for c in cond_same[m]:
            mass = cond_mass[m][c].clamp_min(1.0)
            ret = (cond_same[m][c] / mass)
            for k in range(K):
                rows_cls_cond.append({
                    "model": m,
                    "sec_condition": c,
                    "class_id": k,
                    "class_name": class_names[k],
                    "retention_masked": float(ret[k].item()),
                    "mass_clean_masked": float(cond_mass[m][c][k].item()),
                    "conf_threshold": CONF_THRESHOLD,
                })
    df_cls_cond = pd.DataFrame(rows_cls_cond)
    df_cls_cond.to_csv(os.path.join(SAVE_DIR, "results_class_retention_by_condition.csv"), index=False)

    # Save global class retention
    rows_cls_global = []
    for m in global_same:
        mass = global_mass[m].clamp_min(1.0)
        ret = global_same[m] / mass
        for k in range(K):
            rows_cls_global.append({
                "model": m,
                "class_id": k,
                "class_name": class_names[k],
                "retention_masked_global": float(ret[k].item()),
                "mass_clean_masked_global": float(global_mass[m][k].item()),
                "conf_threshold": CONF_THRESHOLD,
            })
    df_cls_global = pd.DataFrame(rows_cls_global)
    df_cls_global.to_csv(os.path.join(SAVE_DIR, "results_class_retention_global.csv"), index=False)

    # ==========================================================
    # PLOTS: ONLY 3 function-like plots (per scenario), drop "train" only from plots
    # ==========================================================
    plot_dir = os.path.join(SAVE_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for scen in PLOT_SCENARIOS:
        df_s = df_cls_cond[df_cls_cond["sec_condition"] == scen].copy()
        if df_s.empty:
            print(f"[warn] Scenario {scen} not found in results.")
            continue

        # Select classes by percentile coverage of mass (scenario-level, summed across models)
        keep_classes = select_classes_by_mass_coverage(
            df_s,
            coverage=MASS_COVERAGE,
            drop_class=PLOT_DROP_CLASSNAME,
        )
        if len(keep_classes) == 0:
            print(f"[warn] Scenario {scen}: no classes selected (mass total ~ 0).")
            continue

        # Filter to selected classes only
        df_s = df_s[df_s["class_name"].isin(keep_classes)].copy()

        # Pivot: class x model
        piv = df_s.pivot_table(index="class_name", columns="model", values="retention_masked")

        # Reorder classes by the selected list (mass-desc order)
        piv = piv.reindex(keep_classes)

        # Grouped barplot
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        models = list(piv.columns)
        classes = list(piv.index)

        x = np.arange(len(classes))
        n_models = len(models)
        bar_width = 0.8 / max(n_models, 1)

        for j, m in enumerate(models):
            y = piv[m].values
            ax.bar(
                x + j * bar_width - (n_models - 1) * bar_width / 2,
                y,
                width=bar_width,
                label=m
            )

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylabel("Retention (masked)")
        ax.set_title(f"Class retention — {scen} (conf >= {CONF_THRESHOLD}, mass coverage >= {MASS_COVERAGE:.0%})")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"plot_bar_class_retention__{scen}.png"), dpi=200)
        plt.close(fig)

        # Optional debug: show how much mass coverage you actually got
        # (useful once, then you can comment it out)
        mass_by_class = df_cls_cond[df_cls_cond["sec_condition"] == scen].copy()
        mass_by_class = mass_by_class[mass_by_class["class_name"] != PLOT_DROP_CLASSNAME]
        mass_sum = mass_by_class.groupby("class_name")["mass_clean_masked"].sum().sort_values(ascending=False)
        achieved = float(mass_sum.loc[keep_classes].sum() / max(float(mass_sum.sum()), 1e-9))
        print(f"[info] {scen}: kept {len(keep_classes)} classes, achieved mass coverage ~ {achieved:.3f}")

    # ==========================================================
    # QUALITATIVE: SegFormer clean vs adverse for the 3 scenarios
    # ==========================================================
    qual_dir = os.path.join(SAVE_DIR, "qualitative_segformer")
    os.makedirs(qual_dir, exist_ok=True)

    seg_vis = SegFormerAdapter("SegFormer-QUAL", QUAL_SEGFORMER_ID, device)

    scen_to_mode = {
        "Day-RAIN": "dr",
        "Sunset-FOGGY": "sf",
        "Night-RAIN": "nr",
    }



    def select_worst_cases(csv_path: str, model_name: str, scenario: str, k: int):
        df = pd.read_csv(csv_path)

        df = df[
            (df["model"] == model_name) &
            (df["sec_condition"] == scenario)
            ].copy()

        if df.empty:
            return df

        # 🔥 NEW: filter low-coverage samples
        df = df[df["coverage_masked"] >= 0.2]

        if df.empty:
            return df

        df = df.sort_values("pixel_agreement_masked", ascending=True)
        df = df.drop_duplicates(subset=["loc_id"], keep="first")

        return df.head(k)

    CSV_IMG = os.path.join(SAVE_DIR, "results_pixel_agreement_per_image.csv")
    MODEL_FOR_QUAL = "SegFormer-B5"  # scegli quale modello usare per trovare i worst cases

    for scen in PLOT_SCENARIOS:
        worst_df = select_worst_cases(CSV_IMG, MODEL_FOR_QUAL, scen, k=N_QUAL_SAMPLES)
        if worst_df.empty:
            print(f"[warn] No worst cases found for {MODEL_FOR_QUAL} in {scen}")
            continue

        fig = plt.figure(figsize=(14, 3 * len(worst_df)))

        for r, row in enumerate(worst_df.itertuples(index=False)):
            clean_img_pil = Image.open(row.clean_path).convert("RGB")
            adv_img_pil = Image.open(row.sec_path).convert("RGB")

            to_t = transforms.ToTensor()
            clean_t = to_t(clean_img_pil).unsqueeze(0).to(device)
            adv_t = to_t(adv_img_pil).unsqueeze(0).to(device)

            pred_c, _ = seg_vis.predict(clean_t)
            pred_a, _ = seg_vis.predict(adv_t)

            clean_img = tensor_to_uint8_img(clean_t[0])
            adv_img = tensor_to_uint8_img(adv_t[0])
            clean_mask = decode_cityscapes_mask(pred_c[0].detach().cpu().numpy().astype(np.int32))
            adv_mask = decode_cityscapes_mask(pred_a[0].detach().cpu().numpy().astype(np.int32))

            ax0 = fig.add_subplot(len(worst_df), 4, r * 4 + 1)
            ax0.imshow(clean_img);
            ax0.set_title("Clean");
            ax0.axis("off")

            ax1 = fig.add_subplot(len(worst_df), 4, r * 4 + 2)
            ax1.imshow(clean_mask);
            ax1.set_title("Pred clean");
            ax1.axis("off")

            ax2 = fig.add_subplot(len(worst_df), 4, r * 4 + 3)
            ax2.imshow(adv_img);
            ax2.set_title(f"Adverse ({scen})");
            ax2.axis("off")

            ax3 = fig.add_subplot(len(worst_df), 4, r * 4 + 4)
            ax3.imshow(adv_mask);
            ax3.set_title(f"Pred adv\nagree={row.pixel_agreement_masked:.3f}");
            ax3.axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(qual_dir, f"qual_segformer_worst__{scen}.png"), dpi=200)
        plt.close(fig)



    # Save run config
    with open(os.path.join(SAVE_DIR, "run_config.json"), "w") as f:
        import json
        json.dump({
            "data_dir": DATA_DIR,
            "conf_threshold": CONF_THRESHOLD,
            "mass_min_plot": MASS_MIN_PLOT,
            "plot_scenarios": PLOT_SCENARIOS,
            "drop_class_from_plot": PLOT_DROP_CLASSNAME,
            "models": [a.name for a in adapters],
            "qual_segformer": QUAL_SEGFORMER_ID,
            "qual_samples_per_scenario": N_QUAL_SAMPLES,
        }, f, indent=2)

    print("\nDone. Saved in:", SAVE_DIR)
    print(" - results_class_retention_by_condition.csv")
    print(" - plots/plot_bar_class_retention__<scenario>.png  (3 plots)")
    print(" - qualitative_segformer/qual_segformer_clean_vs_<scenario>.png  (3 grids)")


if __name__ == "__main__":
    evaluate_all_models()