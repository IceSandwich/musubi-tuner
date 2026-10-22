import argparse
import logging

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_qwen_image
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

from ice_scripts.comfyops import comfy
import comfy.sd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(clip, batch: list[ItemInfo]):
    prompts = [item.caption for item in batch]
    for i, item in enumerate(batch):
        print(f"Item {i}: {item.item_key}, prompt: {item.caption}")

        tokens = clip.tokenize(item.caption)
        tokens["qwen35_9b"][0] = [ (tid, 1.0) for (tid, tw) in tokens["qwen35_9b"][0] ] # drop token weights
        cond = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=False)
        save_text_encoder_output_cache_qwen_image(item, cond)

def anima_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3.5 9B text encoder safetensors path (official or ComfyUI key layout)",
    )
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for the text encoder, default is bfloat16")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = anima_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    te_dtype = torch.bfloat16
    if args.text_encoder_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype

        te_dtype = str_to_dtype(args.text_encoder_dtype)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MOSS)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # encoder = anima_utils.load_anima_text_encoder(args.text_encoder, dtype=te_dtype, device=device)
    encoder = comfy.sd.load_clip(
        ckpt_paths=[args.text_encoder],
        clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
    )
    clip_type = encoder.cond_stage_model.clip
    assert clip_type == "qwen35_9b", f"Wrong text encoder. Got {clip_type}. Expect Qwen3.5 9B."

    logger.info("Encoding with Qwen3.5 9B")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal encoder
        encode_and_save_batch(encoder, batch)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del encoder

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
