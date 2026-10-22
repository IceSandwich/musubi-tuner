import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_krea2
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.moss_image import flux2_vae

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def encode_and_save_batch(vae: flux2_vae.AutoEncoder, batch: List[ItemInfo]):
	# Stack batch into (B, C, 1, H, W) in RGB order, normalized to [-1, 1].
	contents = []
	for item in batch:
		content = item.content
		content = content[0] if isinstance(content, list) else content  # (H, W, C)
		contents.append(torch.from_numpy(content))
	contents = torch.stack(contents, dim=0)  # (B, H, W, C)
	contents = contents.permute(0, 3, 1, 2)  # (B, C, H, W)
	# But flux small vae don't need F axis
	# contents = contents.unsqueeze(2)  # (B, C, 1, H, W), Qwen-Image VAE needs F axis
	contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

	with torch.no_grad():
		latents = vae.encode(contents.to(vae.device, dtype=vae.dtype))  # (B, C, 1, H, W)
		latents = vae.patchify_and_normalize(latents)

	for b, item in enumerate(batch):
		target_latent = latents[b].unsqueeze(dim=0)  # (1, C, H, W)
		print(f"Saving cache for item {item.item_key} at {item.latent_cache_path}, latents shape: {target_latent.shape}")
		save_latent_cache_krea2(item_info=item, latent=target_latent)

def main():
	parser = cache_latents.setup_parser_common()
	parser = cache_latents.hv_setup_parser(parser)  # VAE

	args = parser.parse_args()

	if args.vae_dtype is not None:
		raise ValueError("VAE dtype is not supported in Moss Image (uses the Flux small VAE default).")

	device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device)

	# Load dataset config
	blueprint_generator = BlueprintGenerator(ConfigSanitizer())
	logger.info(f"Load dataset config from {args.dataset_config}")
	user_config = config_utils.load_user_config(args.dataset_config)
	blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MOSS)
	train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

	datasets = train_dataset_group.datasets

	if args.debug_mode is not None:
		cache_latents.show_datasets(
			datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
		)
		return

	assert args.vae is not None, "VAE checkpoint is required"

	logger.info(f"Loading VAE model from {args.vae}")
	vae = flux2_vae.load_flux2_vae(args.vae, dtype=None, device=device, disable_mmap=True)
	vae.to(device)

	def encode(batch: List[ItemInfo]):
		encode_and_save_batch(vae, batch)

	cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
	main()
