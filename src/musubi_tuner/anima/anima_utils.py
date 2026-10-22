import torch
from typing import Union, Optional, List, Dict
import os
from . import anima_models
import logging
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

logger = logging.getLogger(__name__)


from musubi_tuner.anima.anima_encoder import (
    load_anima_conditioner
)

def load_anima_text_encoder(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    qwen3_max_length: int = 512,
    t5_max_length: int = 512,
    t5_tokenizer_path: Optional[str] = None,
):
    return load_anima_conditioner(
        model_path=model_path,
        dtype=dtype,
        device=device,
        qwen3_max_length=qwen3_max_length,
        t5_max_length=t5_max_length,
        t5_tokenizer_path=t5_tokenizer_path,
    )


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
) -> anima_models.Anima:
    """
    Load Anima model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (
        not fp8_scaled and dit_weight_dtype is not None
    ) or dit_weight_dtype is None, "dit_weight_dtype should be None when fp8_scaled is True"

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # We currently support fixed DiT config for Anima models
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": 28,
        "num_heads": 16,
        "extra_per_block_abs_pos_emb": False,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": True,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
    }
    with torch.device("meta"):
        model = anima_models.Anima(**dit_config)
        # if dit_weight_dtype is not None:
        #     model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    def rename_hook(key: str) -> str:
        # Rename keys to remove "net." prefix for anima-base-v1.0.safetensors and previous versions
        if key.startswith("net."):
            return key[len("net.") :]
        # Also remove "model.diffusion_model." prefix for anima-aesthetics-v1.0.safetensors and later versions
        if key.startswith("model.diffusion_model."):
            return key[len("model.diffusion_model.") :]
        return key

    FP8_OPTIMIZATION_TARGET_KEYS = ["blocks", ""]
    # ".embed." excludes Embedding in LLMAdapter
    FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_embedder", "norm", "adaln", "final_layer", ".embed."]

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    for key in list(sd.keys()):
        new_key = rename_hook(key)
        if new_key != key:
            sd[new_key] = sd.pop(key)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [
            k
            for k in missing
            if not any(buf_name in k for buf_name in ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"))
        ]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}"
            )
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    logger.info(f"Loaded DiT model from {dit_path}, unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    return model
