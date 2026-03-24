import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    Mask2FormerForUniversalSegmentation,
)


# -----------------------------
# Config (no CLI selection)
# -----------------------------
MODELS = [
    ("SegFormer-B0", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"),
    ("SegFormer-B2", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
    ("SegFormer-B5", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"),
    ("Mask2Former-Swin-S", "facebook/mask2former-swin-small-cityscapes-semantic"),
    ("Mask2Former-Swin-L", "facebook/mask2former-swin-large-cityscapes-semantic"),
]

CONF_THRESHOLD = 0.80  # <-- change here


# -----------------------------
# Helpers
# -----------------------------
ILLUM_ORDER = ["Day", "Sunset", "Night"]
WEATHER_ORDER = ["EXTRASUNNY", "RAIN", "FOGGY"]

def condition_sort_key(cond: str):
    # cond like "Night-FOGGY"
    try:
        illum, weat = cond.split("-")
        return (ILLUM_ORDER.index(illum), WEATHER_ORDER.index(weat))
    except Exception:
        return (999, 999)



def sanitize(s: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in s])[:150]


def pixel_agreement_masked(pred_clean, pred_adv, mask):
    """
    pred_clean/pred_adv: [B,H,W] int64
    mask: [B,H,W] bool
    returns: [B] float
    """
    out = []
    for i in range(pred_clean.shape[0]):
        m = mask[i]
        denom = m.float().sum().clamp_min(1.0)
        out.append(((pred_clean[i] == pred_adv[i]) & m).float().sum() / denom)
    return torch.stack(out, dim=0)


def class_retention_from_preds(pred_clean, pred_adv, num_classes, mask=None):
    """
    retention(k) = P(adv=k | clean=k) computed on (optional) subset of pixels.
    pred_*: [H,W]
    mask: [H,W] bool or None
    returns:
      retention: [K] float
      mass_clean: [K] float
      same: [K] float
    """
    a = pred_clean.view(-1)
    b = pred_adv.view(-1)

    if mask is not None:
        m = mask.view(-1)
        a = a[m]
        b = b[m]

    valid = (a >= 0) & (a < num_classes) & (b >= 0) & (b < num_classes)
    a = a[valid]
    b = b[valid]

    mass = torch.bincount(a, minlength=num_classes).float()
    same = torch.bincount(a[a == b], minlength=num_classes).float()
    retention = same / mass.clamp_min(1.0)
    return retention, mass, same

def batch_counts_by_condition(pred_clean, pred_adv, conf_mask, conditions, num_classes: int):
    """
    Computes per-condition counts in a vectorized way (within a batch).
    Returns dict:
      cond -> (same_counts[K] on CPU, mass_counts[K] on CPU)

    pred_clean/pred_adv: [B,H,W] int64 (on GPU ok)
    conf_mask: [B,H,W] bool (on GPU ok)
    conditions: list[str] length B
    """
    # group indices by condition
    idx_by_cond = defaultdict(list)
    for i, c in enumerate(conditions):
        idx_by_cond[c].append(i)

    out = {}
    for c, idxs in idx_by_cond.items():
        idxs_t = torch.tensor(idxs, device=pred_clean.device, dtype=torch.long)

        a = pred_clean.index_select(0, idxs_t)   # [b',H,W]
        b = pred_adv.index_select(0, idxs_t)
        m = conf_mask.index_select(0, idxs_t)

        # flatten masked pixels
        a_f = a[m]
        b_f = b[m]

        valid = (a_f >= 0) & (a_f < num_classes) & (b_f >= 0) & (b_f < num_classes)
        a_f = a_f[valid]
        b_f = b_f[valid]

        mass = torch.bincount(a_f, minlength=num_classes).to(torch.float64)
        same = torch.bincount(a_f[a_f == b_f], minlength=num_classes).to(torch.float64)

        out[c] = (same.detach().to("cpu"), mass.detach().to("cpu"))

    return out


# -----------------------------
# Base Adapter
# -----------------------------
class AdapterBase:
    def __init__(self, display_name: str, model_id: str, device: torch.device):
        self.display_name = display_name
        self.model_id = model_id
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)

    def num_classes(self) -> int:
        raise NotImplementedError

    def id2label(self):
        return None

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        """
        images_01: [B,3,H,W] float in [0,1]
        returns:
          pred: [B,H,W] int64
          conf: [B,H,W] float in [0,1] (max class prob)  (used for conf_threshold)
        """
        raise NotImplementedError


# -----------------------------
# SegFormer Adapter
# -----------------------------
class SegFormerAdapter(AdapterBase):
    def __init__(self, display_name, model_id, device):
        super().__init__(display_name, model_id, device)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()

    def num_classes(self) -> int:
        return int(self.model.config.num_labels)

    def id2label(self):
        return getattr(self.model.config, "id2label", None)

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        # Use HF processor (normalization/resize) and then upsample back to original H,W
        B, C, H, W = images_01.shape

        inputs = self.processor(images=list(images_01), return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)
        logits = out.logits  # [B,K,h,w]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1).values  # [B,H,W]
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf


# -----------------------------
# Mask2Former Adapter
# (We compute per-pixel class probs from query outputs)
# -----------------------------
class Mask2FormerAdapter(AdapterBase):
    def __init__(self, display_name, model_id, device):
        super().__init__(display_name, model_id, device)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()

    def num_classes(self) -> int:
        # includes "no-object" in queries, but semantic classes are config.num_labels
        return int(self.model.config.num_labels)

    def id2label(self):
        return getattr(self.model.config, "id2label", None)

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        """
        Mask2Former outputs:
          - class_queries_logits: [B, Q, num_labels+1] (last is "no-object")
          - masks_queries_logits: [B, Q, h, w]
        Convert to semantic per-pixel probs:
          P(c, p) = sum_q softmax(class_q)[q,c] * sigmoid(mask_q)[q,p]
        """
        B, C, H, W = images_01.shape
        K = self.num_classes()

        inputs = self.processor(images=list(images_01), return_tensors="pt",  do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)

        class_logits = out.class_queries_logits  # [B,Q,K+1]
        mask_logits = out.masks_queries_logits   # [B,Q,h,w]

        # drop "no-object"
        class_probs = F.softmax(class_logits, dim=-1)[..., :K]  # [B,Q,K]
        mask_probs = torch.sigmoid(mask_logits)                 # [B,Q,h,w]

        # Upsample masks to original size
        mask_probs = F.interpolate(mask_probs, size=(H, W), mode="bilinear", align_corners=False)  # [B,Q,H,W]

        # Compute semantic probs [B,K,H,W] with einsum
        # sum over Q: class_probs[b,q,k] * mask_probs[b,q,h,w]
        sem_probs = torch.einsum("bqk,bqhw->bkhw", class_probs, mask_probs).clamp(0.0, 1.0)

        conf = sem_probs.max(dim=1).values
        pred = sem_probs.argmax(dim=1).to(torch.int64)
        return pred, conf


# -----------------------------
# Main evaluation
# -----------------------------
@torch.no_grad()
def run_eval(data_dir: str, save_dir: str, batch_size: int = 2, num_workers: int = 2, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    from exhaustive_dataloader import ExhaustivePairedImageDataset
    ds = ExhaustivePairedImageDataset(data_dir)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Build adapters
    adapters = []
    for display_name, model_id in MODELS:
        if "segformer" in model_id.lower():
            adapters.append(SegFormerAdapter(display_name, model_id, device))
        else:
            adapters.append(Mask2FormerAdapter(display_name, model_id, device))

    # Assume same Cityscapes label space -> use first model's labels as canonical for plots/tables
    canonical_id2label = adapters[0].id2label() or {i: str(i) for i in range(adapters[0].num_classes())}
    class_names = [canonical_id2label[i] for i in range(len(canonical_id2label))]

    # Outputs
    rows_image = []
    rows_condition = []
    rows_class_global = []  # per-class retention aggregated across ALL conditions (for comparative plot)

    for adp in adapters:
        model_name = adp.display_name
        K = adp.num_classes()

        # condition aggregators
        cond_sum = defaultdict(float)
        cond_n = defaultdict(int)

        # per-condition class aggregators (weighted by mass)
        cond_same_sum = defaultdict(lambda: torch.zeros((K,), dtype=torch.float64))
        cond_mass_sum = defaultdict(lambda: torch.zeros((K,), dtype=torch.float64))

        # global class aggregators
        global_same = torch.zeros((K,), dtype=torch.float64)
        global_mass = torch.zeros((K,), dtype=torch.float64)

        for batch in tqdm(loader, desc=f"Eval {model_name}"):
            pair = batch["pair"].to(device)  # [B,2,3,H,W]
            clean = pair[:, 0]
            adv = pair[:, 1]
            B, C, H, W = clean.shape

            pred_clean, conf_clean = adp.predict(clean)
            pred_adv, _ = adp.predict(adv)

            # conf_threshold mask from CLEAN confidence
            conf_mask = conf_clean >= CONF_THRESHOLD  # [B,H,W]

            agree = pixel_agreement_masked(pred_clean, pred_adv, conf_mask)  # [B]

            sec_meta = batch["sec_meta"]
            conds = sec_meta["condition"]
            illums = sec_meta["illumination"]
            weathers = sec_meta["weather"]
            keys = batch["key"]

            # ----- NEW: batch per-condition class counts (micro-optimization)
            # normalize cond strings now
            conds_norm = [c if c is not None else "UNKNOWN" for c in conds]
            counts_by_cond = batch_counts_by_condition(
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                conf_mask=conf_mask,
                conditions=conds_norm,
                num_classes=K,
            )
            # accumulate counts (CPU tensors)
            for c, (same_c, mass_c) in counts_by_cond.items():
                cond_same_sum[c] += same_c
                cond_mass_sum[c] += mass_c
                global_same += same_c
                global_mass += mass_c


            for i in range(B):
                cond = conds_norm[i] if conds[i] is not None else "UNKNOWN"
                illum = illums[i] if illums[i] is not None else "UNKNOWN"
                weat = weathers[i] if weathers[i] is not None else "UNKNOWN"

                agree_i = float(agree[i].item())

                # image-level log
                rows_image.append({
                    "model": model_name,
                    "location_key": keys[i],
                    "sec_condition": cond,
                    "sec_illumination": illum,
                    "sec_weather": weat,
                    "pixel_agreement_masked": agree_i,
                    "conf_threshold": CONF_THRESHOLD,
                })

                # condition aggregation
                cond_sum[cond] += agree_i
                cond_n[cond] += 1



                """
                # per-class retention (masked)
                ret_k, mass_k, same_k = class_retention_from_preds(
                    pred_clean[i], pred_adv[i], K, mask=conf_mask[i]
                )

                # Move to CPU for accumulation (avoid CUDA/CPU mismatch and reduce VRAM use)
                mass_k = mass_k.detach().to("cpu")
                same_k = same_k.detach().to("cpu")

                cond_same_sum[cond] += same_k.double()
                cond_mass_sum[cond] += mass_k.double()

                global_same += same_k.double()
                global_mass += mass_k.double()
                """

        # Per-condition summary for this model
        for cond in sorted(cond_n.keys()):
            n = cond_n[cond]
            rows_condition.append({
                "model": model_name,
                "sec_condition": cond,
                "num_pairs": int(n),
                "pixel_agreement_masked_mean": float(cond_sum[cond] / max(n, 1)),
                "conf_threshold": CONF_THRESHOLD,
            })

        # Global per-class retention for this model (masked, across ALL conditions)
        ret_global = (global_same / global_mass.clamp_min(1.0)).cpu().numpy()
        mass_global = global_mass.cpu().numpy()

        for k in range(K):
            rows_class_global.append({
                "model": model_name,
                "class_id": int(k),
                "class_name": class_names[k] if k < len(class_names) else str(k),
                "retention_masked_global": float(ret_global[k]),
                "mass_clean_masked_global": float(mass_global[k]),
                "conf_threshold": CONF_THRESHOLD,
            })

        # Save per-condition per-class retention table for this model (nice for paper appendix)
        per_cond_class = []
        for cond in sorted(cond_mass_sum.keys()):
            mass = cond_mass_sum[cond].clamp_min(1.0)
            ret = (cond_same_sum[cond] / mass).cpu().numpy()
            m = mass.cpu().numpy()
            for k in range(K):
                per_cond_class.append({
                    "model": model_name,
                    "sec_condition": cond,
                    "class_id": int(k),
                    "class_name": class_names[k] if k < len(class_names) else str(k),
                    "retention_masked": float(ret[k]),
                    "mass_clean_masked": float(m[k]),
                    "conf_threshold": CONF_THRESHOLD,
                })
        pd.DataFrame(per_cond_class).to_csv(
            os.path.join(save_dir, f"class_retention_by_condition__{sanitize(model_name)}.csv"),
            index=False
        )

    # Save CSVs
    df_img = pd.DataFrame(rows_image)
    df_cond = pd.DataFrame(rows_condition)
    df_cls = pd.DataFrame(rows_class_global)

    df_img.to_csv(os.path.join(save_dir, "results_image_level_all_models.csv"), index=False)
    df_cond.to_csv(os.path.join(save_dir, "results_condition_summary_all_models.csv"), index=False)
    df_cls.to_csv(os.path.join(save_dir, "results_class_retention_global_all_models.csv"), index=False)

    # Table: conditions x models
    pivot = df_cond.pivot_table(index="sec_condition", columns="model", values="pixel_agreement_masked_mean")
    pivot = pivot.loc[sorted(pivot.index, key=condition_sort_key)]

    pivot.to_csv(os.path.join(save_dir, "table_pixel_agreement_by_condition.csv"))

    # -----------------------------
    # Plots
    # -----------------------------

    # 1) Heatmap conditions x models
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    data = pivot.values
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(f"Pixel agreement (masked by conf>= {CONF_THRESHOLD})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "plot_heatmap_pixel_agreement_by_condition.png"), dpi=200)
    plt.close(fig)

    # 2) Bar: average over all conditions per model
    avg_by_model = df_cond.groupby("model")["pixel_agreement_masked_mean"].mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar(avg_by_model.index.tolist(), avg_by_model.values)
    ax.set_xticklabels(avg_by_model.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel("Mean pixel agreement (masked)")
    ax.set_title("Average pixel agreement across all conditions")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "plot_bar_avg_pixel_agreement.png"), dpi=200)
    plt.close(fig)

    # 3) Per-class comparative plot: retention by class x model
    # Keep only classes with enough mass (otherwise noise); threshold can be tuned.
    MASS_MIN = 1e5
    df_plot = df_cls[df_cls["mass_clean_masked_global"] >= MASS_MIN].copy()


    # pivot classes x models
    cls_pivot = df_plot.pivot_table(index="class_name", columns="model", values="retention_masked_global")
    cls_pivot.to_csv(os.path.join(save_dir, "table_class_retention_global.csv"))

    # 4) Function-like plot: pixel agreement vs ordered conditions (one line per model)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    x = list(range(len(pivot.index)))
    for model_name in pivot.columns:
        y = pivot[model_name].values
        ax.plot(x, y, marker="o", label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(list(pivot.index), rotation=45, ha="right")
    ax.set_ylabel("Pixel agreement (masked)")
    ax.set_title(f"Pixel agreement across conditions (conf >= {CONF_THRESHOLD})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "plot_lines_pixel_agreement_by_condition.png"), dpi=200)
    plt.close(fig)

    # 5) Function-like plot: class retention (global) vs class (one line per model)
    # Use the same filtered set used for cls_pivot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    x = list(range(len(cls_pivot.index)))
    for model_name in cls_pivot.columns:
        y = cls_pivot[model_name].values
        ax.plot(x, y, marker="o", label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(list(cls_pivot.index), rotation=45, ha="right")
    ax.set_ylabel("Retention (masked, global)")
    ax.set_title(f"Global class retention (conf >= {CONF_THRESHOLD}, mass >= {MASS_MIN:.0f})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "plot_lines_class_retention.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cls_pivot.values, aspect="auto")
    ax.set_xticks(range(len(cls_pivot.columns)))
    ax.set_xticklabels(list(cls_pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(cls_pivot.index)))
    ax.set_yticklabels(list(cls_pivot.index))
    ax.set_title(f"Class retention (masked, mass>={MASS_MIN:.0f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "plot_heatmap_class_retention.png"), dpi=200)
    plt.close(fig)

    # Save run config
    run_cfg = {
        "models": MODELS,
        "conf_threshold": CONF_THRESHOLD,
        "mass_min_for_class_plot": MASS_MIN,
        "num_pairs": int(len(df_img)),
        "num_conditions": int(df_cond["sec_condition"].nunique()),
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print("Saved in:", save_dir)
    print(" - results_image_level_all_models.csv")
    print(" - results_condition_summary_all_models.csv")
    print(" - results_class_retention_global_all_models.csv")
    print(" - table_pixel_agreement_by_condition.csv")
    print(" - plot_heatmap_pixel_agreement_by_condition.png")
    print(" - plot_bar_avg_pixel_agreement.png")
    print(" - table_class_retention_global.csv")
    print(" - plot_heatmap_class_retention.png")
    print(" - class_retention_by_condition__<model>.csv (one per model)")


if __name__ == "__main__":
    # Minimal “config” at top, no CLI model selection
    DATA_DIR = "/home/ace/Downloads/Dataset"     # <-- CHANGE
    SAVE_DIR = "./out_cityscapes_multi"    # <-- CHANGE
    BATCH_SIZE = 2                         # increase if you have VRAM
    NUM_WORKERS = 2
    DEVICE = "cuda"

    run_eval(
        data_dir=DATA_DIR,
        save_dir=SAVE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device_str=DEVICE,
    )
