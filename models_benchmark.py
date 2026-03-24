import os
import json
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation


# -----------------------------
# Metrics helpers
# -----------------------------
def fast_confusion_from_pairs(a: torch.Tensor, b: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    a, b: [H, W] int64
    Returns cm [K, K] where rows=a (clean), cols=b (adv).
    """
    a = a.view(-1)
    b = b.view(-1)
    mask = (a >= 0) & (a < num_classes) & (b >= 0) & (b < num_classes)
    a = a[mask]
    b = b[mask]
    idx = a * num_classes + b
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def iou_per_class_from_confusion(cm: torch.Tensor, eps: float = 1e-9):
    """
    cm: [K, K]
    IoU_k = TP / (TP + FP + FN)
    """
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    denom = tp + fp + fn
    iou = tp / (denom + eps)
    present = denom > 0
    return iou, present, denom


def retention_per_class(cm: torch.Tensor, eps: float = 1e-9):
    """
    retention(k) = P(pred_adv=k | pred_clean=k) = diag / row_sum
    """
    row_sum = cm.sum(dim=1).float()
    ret = torch.diag(cm).float() / (row_sum + eps)
    return ret, row_sum


def mean_entropy_and_confidence(logits: torch.Tensor):
    """
    logits: [B, K, H, W]
    returns:
      conf_mean: [B]
      ent_mean:  [B]
    """
    probs = F.softmax(logits, dim=1)
    conf = probs.max(dim=1).values.mean(dim=(1, 2))
    ent = -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean(dim=(1, 2))
    return conf, ent


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(
    loader,
    model,
    device,
    num_classes: int,
    id2label: dict,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    # Image-level rows
    image_rows = []

    # Aggregations
    cm_global = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")
    cm_by_condition = defaultdict(lambda: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu"))
    cm_by_illum = defaultdict(lambda: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu"))
    cm_by_weather = defaultdict(lambda: torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu"))

    for batch in tqdm(loader, desc="Evaluating"):
        # batch is a dict of lists/tensors (default collate)
        pair = batch["pair"].to(device)  # [B,2,C,H,W]
        clean = pair[:, 0]  # [B,C,H,W]
        adv = pair[:, 1]  # [B,C,H,W]
        B, C, H, W = clean.shape

        # Run model
        out_clean = model(pixel_values=clean)
        out_adv = model(pixel_values=adv)

        # Upsample logits to input size
        logits_clean = F.interpolate(out_clean.logits, size=(H, W), mode="bilinear", align_corners=False)
        logits_adv = F.interpolate(out_adv.logits, size=(H, W), mode="bilinear", align_corners=False)

        pred_clean = logits_clean.argmax(dim=1)  # [B,H,W]
        pred_adv = logits_adv.argmax(dim=1)

        # Agreement
        agreement = (pred_clean == pred_adv).float().mean(dim=(1, 2))  # [B]

        # Confidence + entropy
        conf_clean, ent_clean = mean_entropy_and_confidence(logits_clean)
        conf_adv, ent_adv = mean_entropy_and_confidence(logits_adv)

        # Metadata (collated)
        keys = batch["key"]
        main_paths = batch["main_path"]
        sec_paths = batch["sec_path"]

        # metas are dict-of-lists after collate
        sec_meta = batch["sec_meta"]
        main_meta = batch["main_meta"]

        # The following should be lists of length B:
        sec_conditions = sec_meta["condition"]
        sec_illums = sec_meta["illumination"]
        sec_weathers = sec_meta["weather"]

        for i in range(B):
            cm = fast_confusion_from_pairs(pred_clean[i], pred_adv[i], num_classes).to("cpu")
            cm_global += cm

            cond = sec_conditions[i] if sec_conditions[i] is not None else "UNKNOWN"
            illum = sec_illums[i] if sec_illums[i] is not None else "UNKNOWN"
            weat = sec_weathers[i] if sec_weathers[i] is not None else "UNKNOWN"

            cm_by_condition[cond] += cm
            cm_by_illum[illum] += cm
            cm_by_weather[weat] += cm

            iou_k, present, _ = iou_per_class_from_confusion(cm)
            self_miou = iou_k[present].mean().item() if present.any() else float("nan")

            image_rows.append({
                "location_key": keys[i],
                "main_path": main_paths[i],
                "sec_path": sec_paths[i],
                "sec_condition": cond,
                "sec_illumination": illum,
                "sec_weather": weat,
                "pixel_agreement": float(agreement[i].item()),
                "self_mIoU": float(self_miou),
                "conf_clean": float(conf_clean[i].item()),
                "conf_adv": float(conf_adv[i].item()),
                "conf_drop": float((conf_clean[i] - conf_adv[i]).item()),
                "entropy_clean": float(ent_clean[i].item()),
                "entropy_adv": float(ent_adv[i].item()),
                "entropy_increase": float((ent_adv[i] - ent_clean[i]).item()),
            })

    # -----------------------------
    # Save image-level
    # -----------------------------
    df_img = pd.DataFrame(image_rows)
    df_img.to_csv(os.path.join(save_dir, "results_image_level.csv"), index=False)

    # -----------------------------
    # Summaries
    # -----------------------------
    def summarize_confusion(cm: torch.Tensor):
        iou_k, present, _ = iou_per_class_from_confusion(cm)
        miou = iou_k[present].mean().item() if present.any() else float("nan")
        ret_k, row_mass = retention_per_class(cm)
        return miou, iou_k, ret_k, row_mass

    # Global summary
    global_miou, global_iou_k, global_ret_k, global_mass = summarize_confusion(cm_global)

    summary = {
        "num_samples": int(len(df_img)),
        "pixel_agreement_mean": float(df_img["pixel_agreement"].mean()),
        "self_mIoU_mean": float(df_img["self_mIoU"].mean()),
        "conf_drop_mean": float(df_img["conf_drop"].mean()),
        "entropy_increase_mean": float(df_img["entropy_increase"].mean()),
        "global_self_mIoU_from_confusion": float(global_miou),
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------
    # Per-condition summary
    # -----------------------------
    cond_rows = []
    for cond, cm in sorted(cm_by_condition.items(), key=lambda x: x[0]):
        miou, _, _, _ = summarize_confusion(cm)
        sub = df_img[df_img["sec_condition"] == cond]
        cond_rows.append({
            "sec_condition": cond,
            "num_samples": int(len(sub)),
            "pixel_agreement_mean": float(sub["pixel_agreement"].mean()) if len(sub) else float("nan"),
            "self_mIoU_mean": float(sub["self_mIoU"].mean()) if len(sub) else float("nan"),
            "self_mIoU_from_confusion": float(miou),
            "conf_drop_mean": float(sub["conf_drop"].mean()) if len(sub) else float("nan"),
            "entropy_increase_mean": float(sub["entropy_increase"].mean()) if len(sub) else float("nan"),
        })
    pd.DataFrame(cond_rows).to_csv(os.path.join(save_dir, "results_condition_summary.csv"), index=False)

    # -----------------------------
    # Per-illumination & per-weather summary
    # -----------------------------
    illum_rows = []
    for illum, cm in sorted(cm_by_illum.items(), key=lambda x: x[0]):
        miou, _, _, _ = summarize_confusion(cm)
        sub = df_img[df_img["sec_illumination"] == illum]
        illum_rows.append({
            "sec_illumination": illum,
            "num_samples": int(len(sub)),
            "pixel_agreement_mean": float(sub["pixel_agreement"].mean()) if len(sub) else float("nan"),
            "self_mIoU_mean": float(sub["self_mIoU"].mean()) if len(sub) else float("nan"),
            "self_mIoU_from_confusion": float(miou),
            "conf_drop_mean": float(sub["conf_drop"].mean()) if len(sub) else float("nan"),
            "entropy_increase_mean": float(sub["entropy_increase"].mean()) if len(sub) else float("nan"),
        })
    pd.DataFrame(illum_rows).to_csv(os.path.join(save_dir, "results_illumination_summary.csv"), index=False)

    weather_rows = []
    for weat, cm in sorted(cm_by_weather.items(), key=lambda x: x[0]):
        miou, _, _, _ = summarize_confusion(cm)
        sub = df_img[df_img["sec_weather"] == weat]
        weather_rows.append({
            "sec_weather": weat,
            "num_samples": int(len(sub)),
            "pixel_agreement_mean": float(sub["pixel_agreement"].mean()) if len(sub) else float("nan"),
            "self_mIoU_mean": float(sub["self_mIoU"].mean()) if len(sub) else float("nan"),
            "self_mIoU_from_confusion": float(miou),
            "conf_drop_mean": float(sub["conf_drop"].mean()) if len(sub) else float("nan"),
            "entropy_increase_mean": float(sub["entropy_increase"].mean()) if len(sub) else float("nan"),
        })
    pd.DataFrame(weather_rows).to_csv(os.path.join(save_dir, "results_weather_summary.csv"), index=False)

    # -----------------------------
    # Global class stability table
    # -----------------------------
    class_rows = []
    for k in range(num_classes):
        class_rows.append({
            "class_id": int(k),
            "class_name": id2label.get(k, str(k)),
            "self_iou_global": float(global_iou_k[k].item()),
            "retention_global": float(global_ret_k[k].item()),
            "mass_clean_global": int(global_mass[k].item()),  # how many pixels were class k in clean
        })
    pd.DataFrame(class_rows).to_csv(os.path.join(save_dir, "results_class_stability_global.csv"), index=False)

    # -----------------------------
    # Per-condition class stability (optional but super useful)
    # -----------------------------
    per_cond_class_rows = []
    for cond, cm in sorted(cm_by_condition.items(), key=lambda x: x[0]):
        miou, iou_k, ret_k, mass = summarize_confusion(cm)
        for k in range(num_classes):
            per_cond_class_rows.append({
                "sec_condition": cond,
                "class_id": int(k),
                "class_name": id2label.get(k, str(k)),
                "self_iou": float(iou_k[k].item()),
                "retention": float(ret_k[k].item()),
                "mass_clean": int(mass[k].item()),
            })
    pd.DataFrame(per_cond_class_rows).to_csv(
        os.path.join(save_dir, "results_class_stability_by_condition.csv"),
        index=False
    )

    # Save switch matrices for deeper analysis/plots
    torch.save(cm_global, os.path.join(save_dir, "switch_matrix_global_clean_to_adv.pt"))
    torch.save(dict(cm_by_condition), os.path.join(save_dir, "switch_matrices_by_condition.pt"))

    print("Saved in:", save_dir)
    print(" - results_image_level.csv")
    print(" - results_condition_summary.csv")
    print(" - results_illumination_summary.csv")
    print(" - results_weather_summary.csv")
    print(" - results_class_stability_global.csv")
    print(" - results_class_stability_by_condition.csv")
    print(" - summary.json")
    print(" - switch_matrix_global_clean_to_adv.pt")
    print(" - switch_matrices_by_condition.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="random",
                        help="Dataset mode: illumination prefix d/s/n; optional weather: f/r/s. "
                             "Examples: 'df' (Day-Foggy), 'nr' (Night-Rain), 'random'.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="eval_out")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Import your patched dataset
    from datalaoder_2 import PairedImageDataset  # <-- replace with your actual module/file

    ds = PairedImageDataset(data_dir=args.data_dir, mode=args.mode)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    num_classes = int(model.config.num_labels)
    id2label = getattr(model.config, "id2label", {i: str(i) for i in range(num_classes)})

    # Save run metadata
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    evaluate(loader, model, device, num_classes, id2label, args.save_dir)


if __name__ == "__main__":
    main()
