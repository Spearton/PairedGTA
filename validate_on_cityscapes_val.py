import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from huggingface_hub import hf_hub_download
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ==========================================================
# USER CONFIG
# ==========================================================
CITYSCAPES_ROOT = "/home/ace/Downloads/Cityscapes_dataset"   # e.g. /data/cityscapes
SAVE_DIR = "./cityscapes_val_check"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
NUM_WORKERS = 4

ENABLE_MASK2FORMER = True   # auto-skip if scipy missing

# ==========================================================
# CITYSCAPES LABEL MAPPING: labelIds -> trainIds
# ignore index = 255
# ==========================================================
IGNORE_INDEX = 255
NUM_CLASSES = 19

LABELID_TO_TRAINID = np.full((256,), IGNORE_INDEX, dtype=np.uint8)

# Standard Cityscapes labelIds -> trainIds
_mapping = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}
for k, v in _mapping.items():
    LABELID_TO_TRAINID[k] = v

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# ==========================================================
# DATASET
# ==========================================================
class CityscapesValDataset(Dataset):
    """
    Expects standard Cityscapes structure:
      root/
        leftImg8bit/val/<city>/*_leftImg8bit.png
        gtFine/val/<city>/*_gtFine_labelIds.png
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = self._collect_samples()

    def _collect_samples(self):
        img_root = os.path.join(self.root, "leftImg8bit", "val")
        gt_root = os.path.join(self.root, "gtFine", "val")

        samples = []
        if not os.path.isdir(img_root):
            raise RuntimeError(f"Cityscapes image root not found: {img_root}")
        if not os.path.isdir(gt_root):
            raise RuntimeError(f"Cityscapes label root not found: {gt_root}")

        for city in sorted(os.listdir(img_root)):
            city_img_dir = os.path.join(img_root, city)
            city_gt_dir = os.path.join(gt_root, city)
            if not os.path.isdir(city_img_dir):
                continue

            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue

                img_path = os.path.join(city_img_dir, fname)
                gt_name = fname.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                gt_path = os.path.join(city_gt_dir, gt_name)

                if os.path.isfile(gt_path):
                    samples.append((img_path, gt_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        gt = np.array(Image.open(gt_path), dtype=np.uint8)
        gt_train = LABELID_TO_TRAINID[gt]  # [H,W], uint8 with 255 ignore

        img_t = self.transform(img)  # [3,H,W] in [0,1]
        gt_t = torch.from_numpy(gt_train.astype(np.int64))

        return {
            "image": img_t,
            "target": gt_t,
            "image_path": img_path,
            "target_path": gt_path,
        }

# ==========================================================
# METRICS
# ==========================================================
def confusion_matrix_from_preds(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    pred, target: [H,W] int64
    returns [K,K] where rows=target, cols=pred
    """
    pred = pred.view(-1)
    target = target.view(-1)

    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]

    valid_pred = (pred >= 0) & (pred < num_classes)
    pred = pred[valid_pred]
    target = target[valid_pred]

    idx = target * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm

def iou_from_confusion(cm: torch.Tensor, eps: float = 1e-9):
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    denom = tp + fp + fn
    iou = tp / (denom + eps)
    valid = denom > 0
    miou = iou[valid].mean().item() if valid.any() else float("nan")
    return iou, miou, valid

# ==========================================================
# MODEL ADAPTERS
# ==========================================================
class BaseAdapter:
    def __init__(self, name: str):
        self.name = name

    def num_classes(self) -> int:
        raise NotImplementedError

    def id2label(self) -> Optional[Dict[int, str]]:
        return None

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        images_01: [B,3,H,W] in [0,1]
        returns pred: [B,H,W] int64
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
        inputs = self.processor(images=images_01, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        logits = out.logits
        B, _, H, W = images_01.shape
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).to(torch.int64)
        return pred

class DeepLabV3CityscapesAdapter(BaseAdapter):
    """
    Same checkpoint you're currently using, with selectable preprocess mode.
    preprocess_mode: "none", "imagenet", "255"
    """
    def __init__(
        self,
        device: torch.device,
        preprocess_mode: str = "none",
        repo_id: str = "Koushim/deeplabv3-resnet50-cityscapes",
    ):
        name = f"DeepLabV3-R50 ({preprocess_mode})"
        super().__init__(name)
        self.device = device
        self.preprocess_mode = preprocess_mode

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            num_classes=19,
        ).to(device).eval()

        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        sd = torch.load(weights_path, map_location="cpu")

        try:
            self.model.load_state_dict(sd, strict=True)
        except Exception:
            model_sd = self.model.state_dict()
            kept = {}
            for k, v in sd.items():
                if k in model_sd and v.shape == model_sd[k].shape:
                    kept[k] = v
            self.model.load_state_dict(kept, strict=False)

        self._num = 19
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def num_classes(self) -> int:
        return self._num

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        x = images_01.to(self.device)

        if self.preprocess_mode == "none":
            pass
        elif self.preprocess_mode == "imagenet":
            x = (x - self.mean) / self.std
        elif self.preprocess_mode == "255":
            x = x * 255.0
        else:
            raise ValueError(f"Unknown preprocess_mode: {self.preprocess_mode}")

        logits = self.model(x)["out"]
        pred = logits.argmax(dim=1).to(torch.int64)
        return pred

def try_build_mask2former(device: torch.device) -> List[BaseAdapter]:
    adapters = []
    if not ENABLE_MASK2FORMER:
        return adapters

    try:
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        import scipy  # noqa
    except Exception as e:
        print(f"[warn] Skipping Mask2Former: {e}")
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
            returns pred: [B,H,W] int64
            """
            B, _, H, W = images_01.shape

            # Safer path for HF processors: list of CPU tensors
            images_cpu = [img.detach().cpu() for img in images_01]

            inputs = self.processor(
                images=images_cpu,
                return_tensors="pt",
                do_rescale=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            seg_list = self.processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(H, W)] * B,
            )

            pred = torch.stack(seg_list, dim=0).to(torch.int64)
            return pred

    adapters.append(Mask2FormerAdapter(
        "Mask2Former-Swin-S",
        "facebook/mask2former-swin-small-cityscapes-semantic",
        device
    ))
    adapters.append(Mask2FormerAdapter(
        "Mask2Former-Swin-L",
        "facebook/mask2former-swin-large-cityscapes-semantic",
        device
    ))
    return adapters

# ==========================================================
# EVAL
# ==========================================================
@torch.no_grad()
def validate_models():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    ds = CityscapesValDataset(CITYSCAPES_ROOT, transform=transforms.ToTensor())
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[info] Cityscapes val samples: {len(ds)}")

    adapters: List[BaseAdapter] = [
        SegFormerAdapter("SegFormer-B0", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", device),
        SegFormerAdapter("SegFormer-B2", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024", device),
        SegFormerAdapter("SegFormer-B5", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", device),

        # test all three DeepLab preprocessing variants
        DeepLabV3CityscapesAdapter(device, preprocess_mode="none"),
        DeepLabV3CityscapesAdapter(device, preprocess_mode="imagenet"),
        DeepLabV3CityscapesAdapter(device, preprocess_mode="255"),
    ]
    adapters += try_build_mask2former(device)

    summary_rows = []
    per_class_rows = []

    for adp in adapters:
        print(f"\n[eval] {adp.name}")
        cm_total = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

        for batch in tqdm(loader, desc=adp.name):
            images = batch["image"].to(device, non_blocking=True)    # [B,3,H,W]
            targets = batch["target"]                                # [B,H,W] on CPU

            preds = adp.predict(images).detach().cpu()               # [B,H,W]

            # Ensure same size as GT (safety)
            if preds.shape[-2:] != targets.shape[-2:]:
                preds = F.interpolate(
                    preds.unsqueeze(1).float(),
                    size=targets.shape[-2:],
                    mode="nearest"
                ).squeeze(1).to(torch.int64)

            for i in range(preds.shape[0]):
                cm = confusion_matrix_from_preds(
                    pred=preds[i],
                    target=targets[i],
                    num_classes=NUM_CLASSES,
                    ignore_index=IGNORE_INDEX,
                )
                cm_total += cm

        iou_k, miou, valid = iou_from_confusion(cm_total)

        summary_rows.append({
            "model": adp.name,
            "mIoU": float(miou),
        })

        for k in range(NUM_CLASSES):
            per_class_rows.append({
                "model": adp.name,
                "class_id": k,
                "class_name": CITYSCAPES_CLASS_NAMES[k],
                "IoU": float(iou_k[k].item()),
                "valid": bool(valid[k].item()),
            })

        print(f"[result] {adp.name}: mIoU = {miou:.4f}")

    df_summary = pd.DataFrame(summary_rows).sort_values("mIoU", ascending=False)
    df_per_class = pd.DataFrame(per_class_rows)

    df_summary.to_csv(os.path.join(SAVE_DIR, "cityscapes_val_summary.csv"), index=False)
    df_per_class.to_csv(os.path.join(SAVE_DIR, "cityscapes_val_per_class_iou.csv"), index=False)

    print("\nSaved:")
    print(" -", os.path.join(SAVE_DIR, "cityscapes_val_summary.csv"))
    print(" -", os.path.join(SAVE_DIR, "cityscapes_val_per_class_iou.csv"))

    print("\nTop results:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    validate_models()