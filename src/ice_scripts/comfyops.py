from pathlib import Path
comfyui_folder = Path(__file__).parent.parent.parent.parent / "ComfyUI"
if not (comfyui_folder / "comfy").exists():
	raise ValueError(f"Failed to detect ComfyUI installed location `{comfyui_folder}`, please set it in {Path(__file__).absolute()}")

import sys
sys.path.append(str(comfyui_folder.absolute()))

import json
import torch

import comfy
from comfy import ops  # Always import this to fix issue: 'torch._C._CudaDeviceProperties' object has no attribute 'shared_memory_per_block'
from comfy.quant_ops import QUANT_ALGOS
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
	def forward(self, input, *args, **kwargs):
		_use_quantized = (
			getattr(self, 'layout_type', None) is not None and
			not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
			not getattr(self, 'comfy_force_cast_weights', False) and
			len(self.weight_function) == 0 and len(self.bias_function) == 0
		)
		quantize_input = QUANT_ALGOS.get(getattr(self, 'quant_format', None), {}).get("quantize_input", True)
		if quantize_input == False and getattr(self, 'quant_format', None) == "convrot_w4a4":
			quantize_input = True

		# Training path: quantized forward with compute_dtype backward via autograd function
		if (input.requires_grad and _use_quantized and quantize_input):
			# ModelOpt AWQ-style smoothing
			pre_quant_scale = getattr(self, 'pre_quant_scale', None)
			if pre_quant_scale is not None:
				input = input * comfy.model_management.cast_to_device(pre_quant_scale, input.device, input.dtype)

			scale = getattr(self, 'input_scale', None)
			if scale is not None:
				scale = comfy.model_management.cast_to_device(scale, input.device, None)

			return ops.QuantLinearFunc.apply(
				input, self.weight, self.bias, self.layout_type, scale, input.dtype
			)
	
		return super().forward(input, *args, **kwargs)

mixed_precision_ops.Linear = PatchedMixedPrecisionOpsLinear

def postprocess_comfy_safetensor(metadata: dict[str, any], state_dict: dict[str, torch.Tensor]):
	if "_quantization_metadata" in metadata: # for quantized dit
		print("[+] Detected comfy metadata")
		state_dict, _ = convert_old_quants(state_dict, metadata=metadata)


# https://arxiv.org/pdf/2501.17116
class INT8_DGE_Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, weight):
        # 1. 正常的 INT8 Fake Quantize 前向传播（硬件友好的硬量化）
        # 假设我们有 scale 缩放因子
        scale_w = 127.0 / weight.abs().max()
        weight_q = torch.round(weight * scale_w) / scale_w
        
        # 保存全精度的输入，供反向传播 DGE 算导数
        ctx.save_for_backward(activation, weight, torch.tensor(1.0 / scale_w))
        
        # 2. 跑前向矩阵乘法 (GEMM)
        return torch.matmul(activation, weight_q.t())

    @staticmethod
    def backward(ctx, grad_output):
        activation, weight, delta_tensor = ctx.saved_tensors
        delta = delta_tensor.item() # 获取当前的量化步长
        
        # 1. 标准求导（对应 STE 路径）
        grad_activation = torch.matmul(grad_output, weight)
        grad_weight_ste = torch.matmul(grad_output.t(), activation)
        
        # 2. 计算 DGE 梯度修正项（核心改动！）
        k = 5
        # 归一化缩放
        x_norm = (2.0 * weight / delta) - 1.0
        # 按照论文公式计算导数修正系数
        grad_correction = (1.0 / k) * torch.pow(torch.abs(x_norm) + 1e-7, (1.0 / k) - 1.0)
        # 工程截断，防止死区梯度爆炸
        grad_correction = torch.clamp(grad_correction, max=3.0) 
        
        # 3. 将修正项应用到权重的梯度上
        grad_weight = grad_weight_ste * grad_correction
        
        return grad_activation, grad_weight


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
