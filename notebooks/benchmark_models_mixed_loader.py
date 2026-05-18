import copy
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

CITYSCAPES_ID2LABEL: Dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


def _ensure_module(module_name: str, install_hint: str = ""):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        msg = f"Required dependency '{module_name}' is not available."
        if install_hint:
            msg += f" {install_hint}"
        raise ImportError(msg) from e


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise TypeError("Checkpoint must be a dict-like object.")


def _safe_load_checkpoint(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return _extract_state_dict(ckpt)


def _normalize_images(images_01: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "none":
        return images_01
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], device=images_01.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images_01.device).view(1, 3, 1, 1)
        return (images_01 - mean) / std
    raise ValueError(f"Unsupported input_norm: {mode}")


def _resolve_local_checkpoint(spec: Dict[str, Any]) -> Optional[Path]:
    checkpoint_path = spec.get("checkpoint_path")
    if checkpoint_path is None:
        return None
    p = Path(checkpoint_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {p}. "
            "Set a valid local checkpoint_path in the YAML configuration."
        )
    return p


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
        _ensure_module("scipy", "Install it with `pip install scipy`.")
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

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


class DeepLabV3PlusResNet101Adapter(BaseAdapter):
    def __init__(
        self,
        name: str,
        checkpoint_path: Union[str, Path],
        device: torch.device,
        input_norm: str = "imagenet",
    ):
        _ensure_module(
            "DeepLabV3PlusPytorch",
            "Make sure the DeepLabV3PlusPytorch repository/module is available in PYTHONPATH.",
        )
        from DeepLabV3PlusPytorch import network

        super().__init__(name)
        self.device = device
        self.input_norm = input_norm

        model_name = "deeplabv3plus_resnet101"
        num_classes = 19
        output_stride = 16
        separable_conv = False

        self.model = network.modeling.__dict__[model_name](
            num_classes=num_classes,
            output_stride=output_stride,
        )

        if separable_conv and "plus" in model_name:
            network.convert_to_separable_conv(self.model.classifier)

        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            # some checkpoints may have wrapped keys
            state = _extract_state_dict(state)

        try:
            self.model.load_state_dict(state, strict=False)
        except Exception:
            model_sd = self.model.state_dict()
            kept = {
                k: v for k, v in state.items()
                if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
            }
            self.model.load_state_dict(kept, strict=False)

        self.model = self.model.to(device).eval()
        self._num = 19
        self._id2label = CITYSCAPES_ID2LABEL

    def num_classes(self) -> int:
        return self._num

    def id2label(self):
        return self._id2label

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        x = _normalize_images(images_01.to(self.device), self.input_norm)
        logits = self.model(x)

        if isinstance(logits, dict):
            logits = logits.get("out", next(iter(logits.values())))
        elif isinstance(logits, (list, tuple)):
            logits = logits[0]

        _, _, H, W = images_01.shape
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf


class PIDNetAdapter(BaseAdapter):
    def __init__(
        self,
        name: str,
        checkpoint_path: Union[str, Path],
        device: torch.device,
        arch: str = "pidnet_m",
        input_norm: str = "imagenet",
    ):
        _ensure_module("models", "Make sure the PIDNet repository root is in PYTHONPATH.")
        from models import pidnet

        super().__init__(name)
        self.device = device
        self.input_norm = input_norm
        self.arch = arch

        # Your local PIDNet repo uses get_pred_model, not get_seg_model(model_name=...)
        self.model = pidnet.get_pred_model(name=arch, num_classes=19).to(device).eval()

        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model_dict = self.model.state_dict()
        cleaned = {}
        for k, v in state.items():
            new_k = k[6:] if k.startswith("model.") else k
            if new_k in model_dict and getattr(v, "shape", None) == model_dict[new_k].shape:
                cleaned[new_k] = v

        model_dict.update(cleaned)
        self.model.load_state_dict(model_dict, strict=False)

        self._num = 19
        self._id2label = CITYSCAPES_ID2LABEL

    def num_classes(self) -> int:
        return self._num

    def id2label(self):
        return self._id2label

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        x = _normalize_images(images_01.to(self.device), self.input_norm)
    
        B, C, H, W = x.shape
    
        # PIDNet can produce shape mismatches for sizes not aligned with its internal stride.
        # We pad to the next multiple of 32, run inference, then crop back.
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
    
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    
        out = self.model(x)
    
        if isinstance(out, (list, tuple)):
            logits = out[1] if len(out) > 1 else out[0]
        else:
            logits = out
    
        # Crop back to original padded-resolution area if needed
        logits = logits[..., :x.shape[2], :x.shape[3]]
    
        # Resize back to the original image size
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
    
        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf


class DDRNetOfficialAdapter(BaseAdapter):
    def __init__(
        self,
        name: str,
        checkpoint_path: Union[str, Path],
        device: torch.device,
        input_norm: str = "imagenet",
    ):
        super().__init__(name)
        self.device = device
        self.input_norm = input_norm

        _ensure_module("DDRNet_23", "Make sure /path/to/DDRNet/segmentation is in PYTHONPATH.")
        import DDRNet_23

        # Official repo exposes get_seg_model(cfg, **kwargs)
        self.model = DDRNet_23.get_seg_model(None).to(device).eval()

        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif "model_state" in state and isinstance(state["model_state"], dict):
                state = state["model_state"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]

        model_sd = self.model.state_dict()
        cleaned = {}

        for k, v in state.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]

            if new_k in model_sd and getattr(v, "shape", None) == model_sd[new_k].shape:
                cleaned[new_k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        print(f"[info] DDRNet official loaded tensors: {len(cleaned)}")
        if missing:
            print(f"[info] DDRNet official missing keys: {len(missing)}")
        if unexpected:
            print(f"[info] DDRNet official unexpected keys: {len(unexpected)}")

        self._num = 19
        self._id2label = CITYSCAPES_ID2LABEL

    def num_classes(self) -> int:
        return self._num

    def id2label(self):
        return self._id2label

    @torch.no_grad()
    def predict(self, images_01: torch.Tensor):
        x = _normalize_images(images_01.to(self.device), self.input_norm)
    
        B, C, H, W = x.shape
    
        # DDRNet can produce internal shape mismatches for non-aligned sizes.
        # Pad to the next multiple of 32, run inference, then resize back.
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
    
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    
        logits = self.model(x)
    
        if isinstance(logits, dict):
            logits = logits.get("out", next(iter(logits.values())))
        elif isinstance(logits, (list, tuple)):
            logits = logits[0]
    
        # Crop logits back to padded spatial area if needed
        logits = logits[..., :x.shape[2], :x.shape[3]]
    
        # Resize back to original image size
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
    
        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        pred = probs.argmax(dim=1).to(torch.int64)
        return pred, conf


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
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
    "deeplabv3plus_r101_local": {
        "name": "DeepLabV3+-R101",
        "type": "deeplabv3plus_r101",
        "input_norm": "none",
        "checkpoint_path": "/home/ace/Downloads/models_weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth",
    },
    "pidnet_m_local": {
        "name": "PIDNet-M",
        "type": "pidnet",
        "arch": "pidnet_m",
        "input_norm": "none",
        "checkpoint_path": "/home/ace/Downloads/models_weights/PIDNet_M_Cityscapes_val.pt",
    },
    "ddrnet23_official": {
        "name": "DDRNet-23",
        "type": "ddrnet_official",
        "input_norm": "none",
        "checkpoint_path": "/home/ace/Downloads/models_weights/ddrnet23_cityscapes.pth",
    },
    
}


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(MODEL_REGISTRY)


def build_adapter_from_spec(spec: Dict[str, Any], device: torch.device) -> BaseAdapter:
    model_type = str(spec["type"]).lower()
    name = spec["name"]

    if model_type == "segformer":
        return SegFormerAdapter(name=name, model_id=spec["model_id"], device=device)

    if model_type == "mask2former":
        return Mask2FormerAdapter(name=name, model_id=spec["model_id"], device=device)

    if model_type == "deeplabv3plus_r101":
        ckpt = _resolve_local_checkpoint(spec)
        if ckpt is None:
            raise ValueError(f"{name} requires a local checkpoint_path in the YAML configuration.")
        return DeepLabV3PlusResNet101Adapter(
            name=name,
            checkpoint_path=ckpt,
            device=device,
            input_norm=spec.get("input_norm", "imagenet"),
        )

    if model_type == "pidnet":
        ckpt = _resolve_local_checkpoint(spec)
        if ckpt is None:
            raise ValueError(f"{name} requires a local checkpoint_path in the YAML configuration.")
        return PIDNetAdapter(
            name=name,
            checkpoint_path=ckpt,
            device=device,
            arch=spec.get("arch", "pidnet_m"),
            input_norm=spec.get("input_norm", "imagenet"),
        )

    if model_type == "ddrnet_official":
        ckpt = _resolve_local_checkpoint(spec)
        if ckpt is None:
            raise ValueError(f"{name} requires a local checkpoint_path in the YAML configuration.")
        return DDRNetOfficialAdapter(
            name=name,
            checkpoint_path=ckpt,
            device=device,
            input_norm=spec.get("input_norm", "imagenet"),
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def _normalize_model_config(models_cfg: Union[None, List, Dict]) -> List[Dict[str, Any]]:
    if models_cfg is None:
        return [copy.deepcopy(v) for v in MODEL_REGISTRY.values()]

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
                    required = {"name", "type"}
                    missing = required - set(item.keys())
                    if missing:
                        raise ValueError(f"Custom model spec missing keys: {missing}")
                    out.append(copy.deepcopy(item))
            else:
                raise TypeError("Each entry in models.enabled must be either a string or a dict.")
        return out

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
    return [build_adapter_from_spec(spec, device) for spec in specs]


def resolve_model_name_from_key(model_key: str) -> str:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")
    return MODEL_REGISTRY[model_key]["name"]


def resolve_model_spec_from_key(model_key: str) -> Dict[str, Any]:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")
    return copy.deepcopy(MODEL_REGISTRY[model_key])