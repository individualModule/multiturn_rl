import os
import json
import shutil
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _ensure_empty_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def lora_only_state_dict(model: nn.Module) -> dict:
    """
    Extract only LoRA tensors from a (PEFT-wrapped) model's state_dict.
    """
    base = _unwrap(model)
    return {k: v.detach().cpu() for k, v in base.state_dict().items() if "lora" in k}


def save_lora_state_dict(model: nn.Module, output_path: str) -> str:
    """
    Save only LoRA tensors to a tiny .pt file.
    If output_path is a directory, the file will be named 'adapter_lora_state.pt'.
    """
    sd = lora_only_state_dict(model)
    if not sd:
        raise RuntimeError("No LoRA parameters found to save.")
    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "adapter_lora_state.pt")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"lora_state_dict": sd}, output_path)
    return output_path


def save_lora_adapter(model: nn.Module, output_dir: str) -> str:
    """
    Save a minimal adapter folder:
      - adapter_lora_state.pt
      - adapter_meta.json (format indicator)
    """
    os.makedirs(output_dir, exist_ok=True)
    save_lora_state_dict(model, os.path.join(output_dir, "adapter_lora_state.pt"))
    with open(os.path.join(output_dir, "adapter_meta.json"), "w") as f:
        json.dump({"format": "state_dict"}, f)
    return output_dir


def save_latest_adapter(model: nn.Module, run_name: str, adapters_dir: str = "adapters") -> str:
    out_dir = os.path.join(adapters_dir, run_name, "latest")
    _ensure_empty_dir(out_dir)
    return save_lora_adapter(model, out_dir)


def save_periodic_adapter(model: nn.Module, iteration: int, run_name: str, adapters_dir: str = "adapters") -> str:
    out_dir = os.path.join(adapters_dir, run_name, f"iter_{iteration:06d}")
    _ensure_empty_dir(out_dir)
    return save_lora_adapter(model, out_dir)


def save_best_adapter(model: nn.Module, run_name: str, adapters_dir: str = "adapters") -> str:
    out_dir = os.path.join(adapters_dir, run_name, "best")
    _ensure_empty_dir(out_dir)
    return save_lora_adapter(model, out_dir)


def apply_lora_config_if_needed(model: nn.Module, lora_config) -> nn.Module:
    """
    Ensure model is PEFT-wrapped with the given LoraConfig. No-op if already wrapped.
    """
    base = _unwrap(model)
    if hasattr(base, "peft_config"):
        return model
    from peft import get_peft_model
    return get_peft_model(base, lora_config)


def load_lora_state_dict(model: nn.Module, path: str, strict: bool = False, map_location: str = "cpu") -> Tuple[list, list]:
    """
    Load LoRA tensors from a minimal .pt file into an already LoRA-wrapped model.
    Returns (missing_keys, unexpected_keys).
    """
    base = _unwrap(model)
    blob = torch.load(path, map_location=map_location)
    sd = blob.get("lora_state_dict", blob)
    missing, unexpected = base.load_state_dict(sd, strict=strict)
    return missing, unexpected


def load_adapter_into_model(
    model: nn.Module,
    adapter_path: str,
    adapter_name: str = "default",
    is_trainable: bool = False,
    lora_config: Optional[object] = None,
) -> nn.Module:
    """
    Load adapter into model from:
      - a PEFT adapter directory (contains adapter_config.json), or
      - a minimal state_dict file/folder (adapter_lora_state.pt).
    For the state_dict case, the model must already be LoRA-wrapped with the same LoraConfig,
    or provide lora_config to wrap before loading.
    """
    base = _unwrap(model)

    # Case 1: PEFT adapter directory
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isdir(adapter_path) and os.path.exists(adapter_config_path):
        from peft import PeftModel, get_peft_model, PeftConfig
        if hasattr(base, "peft_config"):
            base.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=is_trainable)
            base.set_adapter(adapter_name)
            return model
        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        wrapped = get_peft_model(base, peft_cfg)
        wrapped.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=is_trainable)
        wrapped.set_adapter(adapter_name)
        return wrapped

    # Case 2: minimal state_dict (.pt file or folder with adapter_lora_state.pt)
    candidate = adapter_path
    if os.path.isdir(adapter_path):
        candidate = os.path.join(adapter_path, "adapter_lora_state.pt")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    if not hasattr(base, "peft_config"):
        if lora_config is None:
            raise ValueError("Model is not LoRA-wrapped. Provide lora_config to wrap before loading.")
        model = apply_lora_config_if_needed(model, lora_config)
        base = _unwrap(model)

    load_lora_state_dict(base, candidate, strict=False)
    return model