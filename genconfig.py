# -*- coding: utf-8 -*-

# 存放输出数据的文件夹，比如 /kaggle/working
root_dir: str = "working/training"
# 存放临时数据的文件夹，比如 /kaggle/tmp
tmp_dir: str = "working/tmp"
# lora的名称
lora_name: str = ""
# 加密密钥
encrypt_key: str = None

config_dict = {
	"network_arguments": {
		# "network_weights": None, # 预训练的lora
		# "dim_from_weights": False, # 从预训练的lora获取network_dim
		"network_dim": 8, # LORA的大小
		# "network_alpha": 2,
		# "network_module": "networks.lora_zimage",
		"network_module": "networks.lora_flux_2"
	},
	"optimizer_arguments": {
		# lr_scheduler: ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]
		# "lr_scheduler": "cosine_with_restarts",
		# "lr_scheduler_num_cycles": 1,
		# "lr_scheduler_power": None,
		# "lr_warmup_steps": 0,

		# optimizer: ["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor"]
		"optimizer_type": "AdamW8bit",
		# "optimizer_args": [
		# 	"weight_decay=0.1", 
		# 	"betas=[0.9,0.99]"
		# ],
		
        "learning_rate": 1e-4, # 2.0e-6,
	},
    "model_settings": {
        "sdpa": True,
        "flash_attn": False,
        "sage_attn": False,
        "xformers": False,
        "flash3": False,
        "split_attn": False,

        "compile": True, # require triton
		
        "blocks_to_swap": None, # int
		# "use_pinned_memory_for_block_swap": False,
		# "disable_numpy_memmap": False,
		# "img_in_txt_in_offloading": False,

		# "timestep_sampling": "shift",
		# "discrete_flow_shift": 2.0,
		"weighting_scheme": "none",
		"timestep_sampling": "flux2_shift",
    },
    "training_settings": {
        "seed": 42,
        # "gradient_checkpointing": False,
        "mixed_precision": "bf16", # "bf16", "fp16", "no"
    
        # "dit": "models/z_image_turbo_bf16.safetensors",
		# "base_weights": [ # 差分lora
        #     "models/zimage_turbo_training_adapter_v2.safetensors",
		# ],
		# "base_weights_multiplier": [ # 差分lora的应用倍数
        #     1,
		# ],
		# "vae": "models/ae.safetensors",
		# "vae_dtype": "bfloat16",
		"model_version": "klein-base-4b",

		"dit": "models/flux-2-klein-base-4b-fp8.safetensors",
		"fp8_base": True,
		"fp8_scaled": True,

		"text_encoder": "models/qwen_3_4b.safetensors",
		"fp8_text_encoder": True,

		"vae": "models/flux2-vae.safetensors",

		"max_train_epochs": 16, # 训练多少epoch
		# "train_batch_size": 1, # 训练的批大小，强制为1，无法设置
		
		# 增加批大小但是分多个steps完成这一批，当需要的批大小很大时且显存不够放一批时使用该值
		"gradient_checkpointing": True,
		"gradient_accumulation_steps": 2,

		"max_data_loader_n_workers": 2,
		"persistent_data_loader_workers": True,
	},
	"saving_arguments": {
		"log_with": "tensorboard",
		"log_prefix": lora_name,
		"output_name": lora_name,
		
		# 保存的频率
		"save_every_n_epochs": 1,
		"save_last_n_epochs": 15,
	}
}

dataset_dict = {
	"general": {
		# 训练的分辨率
		"resolution": 1024,
		"caption_extension": ".txt",
		"enable_bucket": True,
		"bucket_no_upscale": True,
		"batch_size": 1,
	},
	# 数据集，可设置多个数据集
	"datasets": [
		{
			# 数据重复几次
			"num_repeats": 8,
			# 数据集路径，最终会自动复制到tmpdir目录下使用
			"image_directory": R"",
		},
	]
}

import os
import shutil
import toml

current_dir = os.path.dirname(os.path.abspath(__file__))
workingdir = os.path.join(root_dir, lora_name)
configdir = os.path.join(workingdir, "configs")
logdir = os.path.join(workingdir, "logs")
outputdir = os.path.join(workingdir, "outputs")
datasetdir = os.path.join(tmp_dir, "dataset")

for x in [workingdir, configdir, logdir, outputdir, datasetdir]:
	os.makedirs(x, exist_ok = True)

config_dict["saving_arguments"]["output_dir"] = os.path.abspath(outputdir)
config_dict["saving_arguments"]["logging_dir"] = os.path.abspath(logdir)
config_dict["training_settings"]["dit"] = os.path.abspath(config_dict["training_settings"]["dit"])
if "vae" in config_dict["training_settings"]:
	config_dict["training_settings"]["vae"] = os.path.abspath(config_dict["training_settings"]["vae"])

for i in range(len(dataset_dict["datasets"])):
	targetfolder = os.path.join(datasetdir, str(i))
	if os.path.exists(targetfolder):
		shutil.rmtree(targetfolder)
	shutil.copytree(dataset_dict["datasets"][i]["image_directory"], targetfolder)
	dataset_dict["datasets"][i]["image_directory"] = os.path.abspath(targetfolder)
	dataset_dict["datasets"][i]["cache_directory"] = dataset_dict["datasets"][i]["image_directory"]

def CleanConfigAndSave(name: str, filename: str, config_dict: dict):
	for key in config_dict:
		if isinstance(config_dict[key], dict):
			config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}
	
	with open(filename, "w") as f:
		f.write(toml.dumps(config_dict))

	print(f"📄 {name} config saved to {filename}")

config_file = os.path.join(configdir, "training_config.toml")
CleanConfigAndSave("Train", config_file, config_dict)

dataset_file = os.path.join(configdir, "dataset_config.toml")
CleanConfigAndSave("Dataset", dataset_file, dataset_dict)

# accelerate_file = os.path.join(configdir, "accelerate_config", "config.yaml")
# from accelerate.utils import write_basic_config
# write_basic_config(save_location=accelerate_file)
# print(f"📄 Accelerate config saved to {accelerate_file}")