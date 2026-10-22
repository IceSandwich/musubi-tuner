from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.safetensors_utils import WeightTransformHooks
from typing import Union, Optional, List, Dict, Tuple
import torch
from . import moss_image_models, convrot_int8_utils
import logging
logger = logging.getLogger(__name__)

# Prefixes of the DiT state dict keys which are removed on load
DIT_KEY_PREFIXES = ["model.diffusion_model.", "diffusion_model."]

def _strip_dit_prefix(key: str) -> str:
    for prefix in DIT_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def split_fused_qkv(key: str, tensor: Optional[torch.Tensor], dim: int = moss_image_models.MOSS_IMAGE_CONFIG["dim"]):
    """Split hook for the ComfyUI format: `layers.N.proj.qkv.weight` [3*dim, dim] -> q / k / v weights.

    The ConvRot INT8 files of ComfyUI quantize the fused `qkv`: the int8 `.weight` [3*dim, dim] and the per-row
    `.weight_scale` [3*dim, 1] are split along the rows as well (the rotation acts on the input features, so the rows are
    independent), and the `.comfy_quant` spec is duplicated for q / k / v.

    Returns (new_keys, new_tensors). (None, None) if the key is not a fused qkv tensor. If tensor is None, only the keys
    are returned (the loader calls the hook without a tensor first).
    """
    stripped = _strip_dit_prefix(key)
    if not stripped.startswith("layers."):
        return None, None
    suffix = None
    for candidate in (".proj.qkv.weight", ".proj.qkv.weight_scale", ".proj.qkv.comfy_quant"):
        if stripped.endswith(candidate):
            suffix = candidate[len(".proj.qkv") :]  # ".weight", ".weight_scale" or ".comfy_quant"
            break
    if suffix is None:
        return None, None
    base = stripped[: -len("qkv" + suffix)]
    new_keys = [base + "q" + suffix, base + "k" + suffix, base + "v" + suffix]
    if tensor is None:
        return new_keys, None
    if suffix == ".comfy_quant":
        return new_keys, [tensor, tensor.clone(), tensor.clone()]
    assert tensor.shape[0] == 3 * dim, f"Unexpected fused qkv tensor shape {tuple(tensor.shape)} for {key}"
    q, k, v = torch.split(tensor, dim, dim=0)
    return new_keys, [q.contiguous(), k.contiguous(), v.contiguous()]

def pad_text_embeds(embeds_list: List[torch.Tensor], max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of [L_i, C] tensors to [B, L, C] and return the mask [B, L] (int32, 1 = valid)."""
    length = max(e.shape[0] for e in embeds_list)
    if max_length is not None:
        length = max(length, max_length)
    first = embeds_list[0]
    embeds = first.new_zeros(len(embeds_list), length, first.shape[-1])
    mask = torch.zeros(len(embeds_list), length, dtype=torch.int32, device=first.device)
    for i, e in enumerate(embeds_list):
        embeds[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = 1
    return embeds, mask


def _cast_floating_tensors(sd: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]) -> None:
    """Cast the floating point tensors of a (partially) quantized state dict in place, keeping the fp32 `.scale_weight`."""
    if dtype is None:
        return
    for key, value in sd.items():
        if value.is_floating_point() and not key.endswith(".scale_weight"):
            sd[key] = value.to(dtype)

def load_moss_image_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
    disable_mmap: bool = False,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
) -> moss_image_models.MossImage:
    """Load the MOSS-Image DiT.

    Both the official diffusers shards (specify the first file, `diffusion_pytorch_model-00001-of-00003.safetensors`)
    and the ComfyUI single file (fused qkv) are supported. LoRA weights (sd-scripts format, `lora_unet_*`) can be merged
    on load.

    ConvRot INT8 (`convrot_int8=True`) quantizes the Linear layers of the main blocks at load time (after the LoRA merge).
    ComfyUI pre-quantized ConvRot INT8 files (`*_int8_convrot.safetensors`) are detected automatically and loaded as they
    are (LoRA cannot be merged into them). The patched model is frozen (`requires_grad=False`).

    Args:
        device: device used for LoRA merging and quantization
        dit_path: path to the checkpoint (first shard or single file)
        attn_mode: "torch", "flash", "sageattn" or "xformers"
        split_attn: process each sample separately in attention (saves memory for variable length batches)
        loading_device: device to load the weights to
        dit_weight_dtype: dtype of the weights. None loads the weights as is (bf16 for the official checkpoints). With
            ConvRot INT8, the dtype of the layers which are not quantized
        convrot_int8: quantize the base weights to ConvRot INT8
        convrot_int8_bwd: backward mode of the ConvRot INT8 layers, "bf16" (transient dequantization) or "int8"
    """
    # imported here: lora_utils depends on the `networks` package, which is not importable from tools/ scripts

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with torch.device("meta"):
        model = moss_image_models.MossImage(**moss_image_models.MOSS_IMAGE_CONFIG, attn_mode=attn_mode, split_attn=split_attn)
    # if dit_weight_dtype is not None:
    #     model.to(dit_weight_dtype)

    logger.info(
        f"Loading MOSS-Image DiT from {dit_path}, device={loading_device}, dtype={dit_weight_dtype}"
    )

    use_convrot_int8 = False
    quantizer = None
    DIT_CONVROT_INT8_TARGET_KEYS = ["layers."]
    DIT_CONVROT_INT8_EXCLUDE_KEYS = ["norm"]
    if use_convrot_int8:
        # an empty target list only converts the pre-quantized layers of the file
        quantizer = convrot_int8_utils.ConvRotInt8Quantizer(
            DIT_CONVROT_INT8_TARGET_KEYS if convrot_int8 else [], DIT_CONVROT_INT8_EXCLUDE_KEYS
        )

    # hooks = WeightTransformHooks()
    # hooks.rename_hook = _strip_dit_prefix
    # hooks.split_hook = split_fused_qkv

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=False,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=None if use_convrot_int8 else dit_weight_dtype,
        disable_numpy_memmap=disable_mmap,
        # weight_transform_hooks=WeightTransformHooks(split_hook=split_fused_qkv),
        quantizer=None,
    )

    # rename_hook
    for key in list(sd.keys()):
        new_key = _strip_dit_prefix(key)
        if new_key != key:
            sd[new_key] = sd.pop(key)
        if new_key.endswith(".comfy_quant"):
            sd[new_key] = sd.pop(new_key).to(device="cpu")

    # `rope.bands` is a derived constant: discard the checkpoint value and keep the recomputed buffer
    sd.pop("rope.bands", None)

    if use_convrot_int8:
        _cast_floating_tensors(sd, dit_weight_dtype)
        convrot_int8_utils.apply_convrot_int8_monkey_patch(
            model, sd, bwd_mode=convrot_int8_bwd, groupsize_map=quantizer.module_groupsizes
        )
        # int8 tensors cannot require grad, and load_state_dict(assign=True) re-wraps the incoming tensors with the meta
        # parameters' requires_grad (default True). The base weights are frozen for LoRA training anyway.
        model.requires_grad_(False)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if k != "rope.bands"]
    unexpected = [k for k in unexpected if not k.endswith(".comfy_attention.config")]
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    if loading_device.type != "cpu":
        model.rope.bands = model.rope._default_bands(loading_device)

    logger.info(f"Loaded MOSS-Image DiT from {dit_path}")
    return model