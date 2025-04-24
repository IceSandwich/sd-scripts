import os
import re
import toml
from time import time
from pathlib import Path
import typing

# 项目设置
project_name = "xxxx" # 保存到 working/{project_name}

# 数据集
dataset_path = R"D:\MyGithub\LCM_distillation\xxxx"
resolution = 1024
caption_extension = ".txt"
shuffle_tags = True
activation_tags = 1

# 训练
LOWRAM = True
num_repeats = 20
preferred_unit = "Epochs"
how_many = 15

save_every_n_epochs = 1
keep_only_last_n_epochs = 15

train_batch_size = 1
cross_attention = "sdpa"
# mixed_precision = "fp16"
mixed_precision = "bf16"
cache_latents = True
cache_latents_to_drive = True
cache_text_encoder_outputs = False

# "AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor"
optimizer = "AdamW8bit"
optimizer_args = "weight_decay=0.1 betas=[0.9,0.99]"
# optimizer = "AdaFactor"
# optimizer_args = "scale_parameter=False relative_step=False warmup_init=False"

recommended_values_for_prodigy = True

# 模型
model_file = R"D:\GITHUB\stable-diffusion-webui\models\Stable-diffusion\FluentlyXL-v3.safetensors"
vae_file = R"D:\GITHUB\stable-diffusion-webui\models\VAE\sdxl.vae.safetensors"

unet_lr = 3e-4*4
text_encoder_lr = 6e-5*4

# "constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"
lr_scheduler = "cosine_with_restarts"
lr_scheduler_number = 3

lr_warmup_ratio = 0.05
lr_warmup_steps = 0

min_snr_gamma = 7.0

lora_type = "LoRA"

# lora
network_dim = 8
network_alpha = 4

# locon
conv_dim = 4
conv_alpha = 1





WorkingDir = Path(__file__).parent.absolute() / "working" / project_name
ConfigDir = WorkingDir / "config"
LogDir = WorkingDir / "logs"
OutputDir = WorkingDir / "output"

os.makedirs(WorkingDir, exist_ok=True)
os.makedirs(ConfigDir, exist_ok=True)
os.makedirs(LogDir, exist_ok=True)
os.makedirs(OutputDir, exist_ok=True)

# 数据集检查
shuffle_caption = shuffle_tags
keep_tokens = activation_tags

if resolution < 768:
	resolution = 768
	print("⚠️ resolution is out of range, adjusted to: ", resolution)
elif resolution > 1536:
	resolution = 1536
	print("⚠️ resolution is out of range, adjusted to: ", resolution)
else:
	temp_resolution = round(resolution / 128) * 128
	if (resolution != temp_resolution):
		print("⚠️ resolution is rouned to nearest step: ", temp_resolution)
		resolution = temp_resolution

# 训练检查
if not preferred_unit in ["Epochs", "Steps"]:
	raise ValueError("💥 Error: invalid value for preferred_unit")

max_train_epochs = how_many if preferred_unit == "Epochs" else None
max_train_steps = how_many if preferred_unit == "Steps" else None

if not save_every_n_epochs:
	save_every_n_epochs = max_train_epochs
if not keep_only_last_n_epochs:
	keep_only_last_n_epochs = max_train_epochs

if train_batch_size < 1:
	train_batch_size = 1
	print("⚠️ train_batch_size is out of range, adjusted to: ", train_batch_size)
elif train_batch_size > 16:
	train_batch_size = 16
	print("⚠️ train_batch_size is out of range, adjusted to: ", train_batch_size)
else:
	train_batch_size = int(train_batch_size)

if not cross_attention in ["sdpa", "xformers"]:
	raise ValueError("💥 Error: invalid value for cross_attention")
	
if not mixed_precision in ["bf16", "fp16"]:
	raise ValueError("💥 Error: invalid value for mixed_precision")

if not optimizer in ["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor"]:
	raise ValueError("💥 Error: invalid value for optimizer")

optimizer_args = [a.strip() for a in optimizer_args.split(' ') if a]

if any(opt in optimizer.lower() for opt in ["dadapt", "prodigy"]):
	if recommended_values_for_prodigy:
		unet_lr = 0.75
		text_encoder_lr = 0.75
		network_alpha = network_dim

# 模型检查
if not lr_scheduler in ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]:
	raise ValueError("💥 Error: invalid value for lr_scheduler")

lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0
lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0

if min_snr_gamma == 0:
	pass
else:
	if min_snr_gamma < 4.0:
		min_snr_gamma = 4.0
		print("⚠️ min_snr_gamma is out of range, adjusted to: ", min_snr_gamma)
	elif min_snr_gamma > 16.0:
		min_snr_gamma = 16.0
		print("⚠️ min_snr_gamma is out of range, adjusted to: ", min_snr_gamma)

if not lora_type in ["LoRA", "LoCon"]:
	raise ValueError("💥 Error: invalid value for lora_type")
	
network_module = "networks.lora"
network_args = None
if lora_type.lower() == "locon":
	network_args = [f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}"]
	

# 生成配置
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
def SaveToml(config: typing.Dict[str, typing.Any], filename: str):
	for key in config:
		if isinstance(config[key], dict):
			config[key] = {k: v for k, v in config[key].items() if v is not None}
	with open(filename, "w") as f:
		f.write(toml.dumps(config))
	print(f"📄 Config saved to {filename}")

# AccelerateConfig = ConfigDir / "accelerate_config.yaml"
# from accelerate.utils import write_basic_config
# if not os.path.exists(str(AccelerateConfig)):
# 	write_basic_config(save_location=str(AccelerateConfig))
# 	print(f"📄 Config saved to {str(AccelerateConfig)}")

train_config = {
	"network_arguments": {
		"unet_lr": unet_lr,
		"text_encoder_lr": text_encoder_lr if not cache_text_encoder_outputs else 0,
		"network_dim": network_dim,
		"network_alpha": network_alpha,
		"network_module": network_module,
		"network_args": network_args,
		"network_train_unet_only": text_encoder_lr == 0 or cache_text_encoder_outputs,
	},
	"optimizer_arguments": {
		"learning_rate": unet_lr,
		"lr_scheduler": lr_scheduler,
		"lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
		"lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
		"lr_warmup_steps": lr_warmup_steps if lr_scheduler != "constant" else None,
		"optimizer_type": optimizer,
		"optimizer_args": optimizer_args if optimizer_args else None,
	},
	"training_arguments": {
		"pretrained_model_name_or_path": model_file,
		"vae": vae_file,
		"max_train_steps": max_train_steps,
		"max_train_epochs": max_train_epochs,
		"train_batch_size": train_batch_size,
		"seed": 42,
		"max_token_length": 225,
		"xformers": cross_attention == "xformers",
		"sdpa": cross_attention == "sdpa",
		"min_snr_gamma": min_snr_gamma if min_snr_gamma > 0 else None,
		"lowram": LOWRAM,
		"no_half_vae": True,
		"gradient_checkpointing": True,
		"gradient_accumulation_steps": 1,
		"max_data_loader_n_workers": 8,
		"persistent_data_loader_workers": True,
		"mixed_precision": mixed_precision,
		"full_bf16": mixed_precision == "bf16",
		"cache_latents": cache_latents,
		"cache_latents_to_disk": cache_latents_to_drive,
		"cache_text_encoder_outputs": cache_text_encoder_outputs,
		"min_timestep": 0,
		"max_timestep": 1000,
		"prior_loss_weight": 1.0,
	},
	"saving_arguments": {
		"save_precision": "fp16",
		"save_model_as": "safetensors",
		"save_every_n_epochs": save_every_n_epochs,
		"save_last_n_epochs": keep_only_last_n_epochs,
		"output_name": project_name,
		"output_dir": str(OutputDir),
		"log_prefix": project_name,
		"logging_dir": str(LogDir),
	}
}
TrainConfig = ConfigDir / "training_config.toml"
SaveToml(train_config, str(TrainConfig))

dataset_config_dict = {
	"general": {
		"resolution": resolution,
		"shuffle_caption": shuffle_caption and not cache_text_encoder_outputs,
		"keep_tokens": keep_tokens,
		"flip_aug": False,
		"caption_extension": caption_extension,
		"enable_bucket": True,
		"bucket_no_upscale": True,
		"bucket_reso_steps": 64,
		"min_bucket_reso": 256,
		"max_bucket_reso": 4096,
	},
	"datasets": [
		{
			"subsets": [
				{
					"num_repeats": num_repeats,
					"image_dir": dataset_path,
					"class_tokens": None if caption_extension else project_name
				}
			]
		}
	]
}
DatasetConfig = ConfigDir / "dataset_config.toml"
SaveToml(dataset_config_dict, str(DatasetConfig))

# print("\n⭐ Starting trainer...\n")

	# !accelerate launch --quiet --config_file={accelerate_config_file} 
	# --num_cpu_threads_per_process=1 
	# train_network_xl_wrapper.py
	# --dataset_config={dataset_config_file} 
	# --config_file={config_file}

print("Done. Run the following command by yourself.")
print(f'accelerate launch --num_cpu_threads_per_process=1 train_network_xl_wrapper.py --dataset_config="{str(DatasetConfig)}" --config_file="{str(TrainConfig)}"')
