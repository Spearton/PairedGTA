import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


class BaseAdapter:
    def __init__(self, name: str):
        self.name = name

    def num_classes(self) -> int:
        raise NotImplementedError

    def id2label(self) -> Optional[Dict[int, str]]:
        return None

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SegFormerAdapter(BaseAdapter):
    def __init__(self, name: str, model_id: str, device: torch.device):
        super().__init__(name)
        self.device = device
        self.model_id = model_id
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
        _, _, H, W = images_01.shape
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf


class Mask2FormerAdapter(BaseAdapter):
    def __init__(self, name: str, model_id: str, device: torch.device):
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        import scipy  # noqa: F401

        super().__init__(name)
        self.device = device
        self.model_id = model_id
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
        _, _, H, W = images_01.shape
        images_cpu = [img.detach().cpu() for img in images_01]
        inputs = self.processor(images=images_cpu, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        class_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits

        class_probs = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_probs = torch.sigmoid(mask_logits)
        mask_probs = F.interpolate(mask_probs, size=(H, W), mode="bilinear", align_corners=False)

        sem_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        probs = sem_scores / sem_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

        conf = probs.max(dim=1).values
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "segformer_b0": {
        "name": "SegFormer-B0",
        "type": "segformer",
        "model_id": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    },
    "segformer_b2": {
        "name": "SegFormer-B2",
        "type": "segformer",
        "model_id": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    },
    "segformer_b5": {
        "name": "SegFormer-B5",
        "type": "segformer",
        "model_id": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    },
    "mask2former_swin_s": {
        "name": "Mask2Former-Swin-S",
        "type": "mask2former",
        "model_id": "facebook/mask2former-swin-small-cityscapes-semantic",
    },
    "mask2former_swin_l": {
        "name": "Mask2Former-Swin-L",
        "type": "mask2former",
        "model_id": "facebook/mask2former-swin-large-cityscapes-semantic",
    },
}


def get_model_registry() -> Dict[str, Dict[str, str]]:
    return copy.deepcopy(MODEL_REGISTRY)


def build_adapter_from_spec(spec: Dict[str, str], device: torch.device) -> BaseAdapter:
    model_type = spec["type"].lower()
    name = spec["name"]
    model_id = spec["model_id"]

    if model_type == "segformer":
        return SegFormerAdapter(name=name, model_id=model_id, device=device)
    if model_type == "mask2former":
        return Mask2FormerAdapter(name=name, model_id=model_id, device=device)

    raise ValueError(f"Unsupported model type: {model_type}")


def _normalize_model_config(models_cfg: Union[None, List, Dict]) -> List[Dict[str, str]]:
    if models_cfg is None:
        # default: all models in registry
        return [copy.deepcopy(v) for v in MODEL_REGISTRY.values()]

    # New style:
    # models:
    #   enabled:
    #     - segformer_b0
    #     - mask2former_swin_l
    # or:
    # models:
    #   enabled:
    #     - {key: segformer_b0}
    #     - {name: Custom, type: segformer, model_id: ...}
    if isinstance(models_cfg, dict) and "enabled" in models_cfg:
        items = models_cfg["enabled"]
        out = []
        for item in items:
            if isinstance(item, str):
                if item not in MODEL_REGISTRY:
                    raise KeyError(f"Unknown model key in YAML: {item}")
                out.append(copy.deepcopy(MODEL_REGISTRY[item]))
            elif isinstance(item, dict):
                if "key" in item:
                    key = item["key"]
                    if key not in MODEL_REGISTRY:
                        raise KeyError(f"Unknown model key in YAML: {key}")
                    spec = copy.deepcopy(MODEL_REGISTRY[key])
                    spec.update({k: v for k, v in item.items() if k != "key"})
                    out.append(spec)
                else:
                    required = {"name", "type", "model_id"}
                    missing = required - set(item.keys())
                    if missing:
                        raise ValueError(f"Custom model spec missing keys: {missing}")
                    out.append(copy.deepcopy(item))
            else:
                raise TypeError("Each entry in models.enabled must be either a string or a dict.")
        return out

    # Backward-compatible style:
    # models:
    #   segformer_b0: true
    #   segformer_b2: true
    #   segformer_b5: true
    #   mask2former: true
    if isinstance(models_cfg, dict):
        out = []
        if models_cfg.get("segformer_b0", False):
            out.append(copy.deepcopy(MODEL_REGISTRY["segformer_b0"]))
        if models_cfg.get("segformer_b2", False):
            out.append(copy.deepcopy(MODEL_REGISTRY["segformer_b2"]))
        if models_cfg.get("segformer_b5", False):
            out.append(copy.deepcopy(MODEL_REGISTRY["segformer_b5"]))
        if models_cfg.get("mask2former", False):
            out.append(copy.deepcopy(MODEL_REGISTRY["mask2former_swin_s"]))
            out.append(copy.deepcopy(MODEL_REGISTRY["mask2former_swin_l"]))
        return out

    raise TypeError("Unsupported models configuration format.")


def build_adapters_from_config(models_cfg, device: torch.device) -> List[BaseAdapter]:
    specs = _normalize_model_config(models_cfg)
    if not specs:
        return []
    adapters = []
    for spec in specs:
        adapters.append(build_adapter_from_spec(spec, device))
    return adapters


def resolve_model_name_from_key(model_key: str) -> str:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")
    return MODEL_REGISTRY[model_key]["name"]


def resolve_model_spec_from_key(model_key: str) -> Dict[str, str]:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")
    return copy.deepcopy(MODEL_REGISTRY[model_key])
