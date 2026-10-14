from pathlib import Path
comfyui_folder = Path(__file__).parent.parent.parent.parent / "ComfyUI"
if not (comfyui_folder / "comfy").exists():
	raise ValueError(f"Failed to detect ComfyUI installed location `{comfyui_folder}`, please set it in {Path(__file__).absolute()}")

import sys
sys.path.append(str(comfyui_folder.absolute()))

import json
import torch

from comfy import ops  # Always import this to fix issue: 'torch._C._CudaDeviceProperties' object has no attribute 'shared_memory_per_block'
from comfy.utils import convert_old_quants
from comfy_kitchen.tensor import QuantizedTensor
# from comfy.text_encoders.qwen3vl import _make_qwen3vl_model
# from comfy.text_encoders.qwen3vl import tokenizer as _make_qwen3vl_tokenizer
import comfy.model_management

comfy.model_management.training_fp8_bwd = False # control backward of QuantLinearFunc
class TypeHintForMixedPrecisionOps:
	Linear: torch.nn.Linear
mixed_precision_ops: TypeHintForMixedPrecisionOps = ops.mixed_precision_ops({}, full_precision_mm=False)

@torch.compiler.disable
class PatchedMixedPrecisionOpsLinear(mixed_precision_ops.Linear):
    def _forward(self, input, weight, bias):
        scale = getattr(self, 'input_scale', None)
        if scale is not None:
            scale = comfy.model_management.cast_to_device(scale, input.device, None)

        compute_dtype = input.dtype
        output = ops.QuantLinearFunc.apply(
            input, weight, bias, self.layout_type, scale, compute_dtype
        )
        return output
mixed_precision_ops.Linear = PatchedMixedPrecisionOpsLinear

def postprocess_comfy_safetensor(metadata: dict[str, any], state_dict: dict[str, torch.Tensor]):
	if "_quantization_metadata" in metadata: # for quantized dit
		print("[+] Detected comfy metadata")
		state_dict, _ = convert_old_quants(state_dict, metadata=metadata)

def test_convrot_w4a4_loads_into_params():
	"""ConvRot W4A4 checkpoints must load as the dedicated kitchen layout."""
	
	torch.manual_seed(456)
	layer_quant_config = {
		"layer": {
			"format": "convrot_w4a4",
			"convrot_groupsize": 256,
			"linear_dtype": "int8",
		}
	}
	weight = torch.randn(16, 256, dtype=torch.bfloat16)
	bias = torch.randn(16, dtype=torch.bfloat16)
	q_weight = QuantizedTensor.from_float(
		weight,
		"TensorCoreConvRotW4A4Layout",
		convrot_groupsize=256,
		quant_group_size=64,
	)
	q_weight = q_weight.to("cuda:0")
	bias = bias.to("cuda:0")
	state_dict = {
		"layer.weight": q_weight._qdata,
		"layer.bias": bias,
		"layer.weight_scale": q_weight._params.scale,
	}

	state_dict, _ = convert_old_quants(
		state_dict,
		metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
	)
	model = torch.nn.Module()
	model.layer = mixed_precision_ops.Linear(256, 16, device="cpu", dtype=torch.bfloat16)
	model.load_state_dict(state_dict, strict=False)

	assert isinstance(model.layer.weight, QuantizedTensor)
	assert model.layer.weight._layout_cls == "TensorCoreConvRotW4A4Layout"
	assert model.layer.weight._params.convrot_groupsize == 256
	assert model.layer.weight._params.quant_group_size == 64
	assert model.layer.weight._params.linear_dtype == "int8"
	print(model)
	print(model.layer.weight)
	print(model.layer.requires_grad_)

	input_tensor = torch.randn(4, 256, dtype=torch.bfloat16)
	input_tensor = input_tensor.to("cuda:0")
	model = model.to("cuda:0")
	loaded_out = model.layer(input_tensor)
	ref_out = torch.nn.functional.linear(input_tensor, q_weight, bias)
	assert torch.equal(loaded_out, ref_out)
	print(loaded_out, loaded_out.shape)

	saved = model.state_dict()
	saved_conf = json.loads(saved["layer.comfy_quant"].numpy().tobytes())
	assert saved_conf["format"] == "convrot_w4a4"
	assert saved_conf["convrot_groupsize"] == 256
	assert saved_conf["linear_dtype"] == "int8"
	assert "quant_group_size" not in saved_conf

if __name__ == "__main__":
	test_convrot_w4a4_loads_into_params()
