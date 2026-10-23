import abc
import typing
from accelerate import Accelerator
import torch
import logging
import argparse
import os
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.hv_train_network import DiTOutput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def shrink_hidden_state(hidden_state: torch.Tensor, mask: torch.Tensor):
    """
    hidden_state: (seq_len, *)
    mask: (seq_len)
    """
    valid = mask.bool()
    embed = hidden_state[valid]
    return embed

def pad_text_embeds(
    embeds_list: typing.List[torch.Tensor],
    max_length: typing.Optional[int] = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Right-pad a list of variable-length tensors to a fixed-length batch tensor.

    Each tensor in embeds_list has shape (L_i, *), where L_i is the sequence length
    and * represents any number of trailing dimensions (e.g., (), (D,), (layers, D)).
    All tensors must share the same trailing dimensions.

    The function pads all sequences to the same length (either the maximum L_i in the
    batch or a user-specified max_length, whichever is larger), stacks them into a
    single batch tensor, and returns a corresponding boolean-style mask indicating
    which positions are valid (1 = original token, 0 = padding).

    Args:
        embeds_list: A list of torch.Tensors, each of shape (L_i, *).
                     All tensors must have identical trailing dimensions (*).
        max_length: Optional target length. If provided, all sequences will be padded
                    to at least this length. If None, the longest sequence in the
                    current batch determines the padding length.

    Returns:
        A tuple of (padded_embeds, mask):
            - padded_embeds: torch.Tensor of shape (B, L, *), where B = len(embeds_list),
              L = max(actual lengths, max_length). Original embeddings are placed at the
              beginning of each sequence; remaining positions are zero-padded.
            - mask: torch.Tensor of shape (B, L) with dtype torch.int32. Entry (i, j) = 1
              if position j in sample i contains an original embedding, and 0 if it is
              padding.

    Raises:
        ValueError: If embeds_list is empty.
        RuntimeError: If the tensors in embeds_list have incompatible trailing dimensions.

    Example:
        >>> e1 = torch.randn(3, 4)   # (L=3, D=4)
        >>> e2 = torch.randn(5, 4)   # (L=5, D=4)
        >>> embeds, mask = pad_text_embeds([e1, e2])
        >>> embeds.shape
        torch.Size([2, 5, 4])
        >>> mask
        tensor([[1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1]], dtype=torch.int32)

        >>> e3 = torch.randn(2, 6, 8)  # (L=2, layers=6, hidden=8)
        >>> e4 = torch.randn(4, 6, 8)  # (L=4, layers=6, hidden=8)
        >>> embeds, mask = pad_text_embeds([e3, e4], max_length=8)
        >>> embeds.shape
        torch.Size([2, 8, 6, 8])
        >>> mask.shape
        torch.Size([2, 8])
    """
    if not embeds_list:
        raise ValueError("embeds_list cannot be empty.")

    # Determine target sequence length
    actual_max_len = max(e.shape[0] for e in embeds_list)
    target_len = actual_max_len
    if max_length is not None:
        target_len = max(actual_max_len, max_length)

    # Infer output shape: (B, L, *) where * preserves all trailing dims
    first = embeds_list[0]
    batch_size = len(embeds_list)
    output_shape = (batch_size, target_len) + first.shape[1:]

    # Pre-allocate padded tensor and mask
    padded = first.new_zeros(output_shape)
    mask = torch.zeros(batch_size, target_len, dtype=torch.int32, device=first.device)

    # Copy original embeddings and set mask bits
    for i, emb in enumerate(embeds_list):
        seq_len = emb.shape[0]
        padded[i, :seq_len] = emb
        mask[i, :seq_len] = 1

    return padded, mask

class Conditioner(abc.ABC):
    @abc.abstractmethod
    def encode_text(self, text: typing.List[str]) -> typing.Dict[str, torch.Tensor]:
        """
        output with prefix varlen_ to avoid stacking in training batch.
        output with subfix _mask will be used for trimming with same key tensor in cache_cond.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def load_model(self, checkpoints: typing.List[str]):
        raise NotImplementedError()

class DiTNetwork(abc.ABC):
    @abc.abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()
    
    def load_model(self, 
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        device: torch.device | str,
        loading_device: torch.device | str,
        # blocks_to_swap: int = 0,
        # swap_config: BlockSwapConfig | None = None,
    ):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_latent_shape(self, height: int, width: int) -> typing.Tuple[int, int, int, int]:
        """
        return (B, C, H, W)
        """
        raise NotImplementedError()
    
    def get_prefer_dtype(self) -> torch.dtype:
        return torch.bfloat16
    
    @abc.abstractmethod
    def get_contexts_from_batch(self, batch: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        """
        Conditioner传递的数据会去掉 varlen_ 前缀
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def call_dit(self, noisy_model_input: torch.Tensor, timesteps: torch.Tensor, contexts: typing.Dict[str, torch.Tensor], accelerator: Accelerator, args: typing.Dict[str, typing.Any] = {}) -> torch.Tensor:
        """
        请调用 get_contexts_from_batch 获取数据。
        传递的 noisy_model_input 维度是 (B, frame_len, channel, height, width)
        """
        raise NotImplementedError()

    @torch.inference_mode()
    def generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        negative_embeds: typing.Optional[torch.Tensor],
        negative_mask: typing.Optional[torch.Tensor],
        height: int,
        width: int,
        steps: int,
        guidance_scale: float,
        flow_shift: float,
        uncond_skip_layers: tuple[int, ...],
        seed: typing.Optional[int],
        device: torch.device | str,
        show_progress: bool = True,
        num_train_timesteps: int = 1000,
    ) -> torch.Tensor:
        """Sample normalized, patchified latents [1, 128, H/16, W/16] (float32).

        The initial noise is drawn on a CPU generator in float32 and CFG / Euler updates are done in float32 like the official
        pipeline. CFG is applied when `guidance_scale != 1` and the negative embeddings are given.
        """
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        shape = self.get_latent_shape(height, width)
        latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

        prefer_dtype = self.get_prefer_dtype()
        positive = prompt_embeds.to(device, dtype=prefer_dtype)
        positive_mask = prompt_mask.to(device)
        do_cfg = guidance_scale != 1.0 and negative_embeds is not None
        if do_cfg:
            negative = negative_embeds.to(device, dtype=prefer_dtype)
            negative_mask = negative_mask.to(device)

        def get_sampling_sigmas(
            num_inference_steps: int, shift: float = 5.0, training_shift: float = 1.0
        ) -> torch.Tensor:
            """Return the sampling sigmas [steps + 1] (descending, the last value is 0) on the CPU in float32.

            The number of effective steps can be smaller than `num_inference_steps` because the shifted grid is snapped to the
            training grid and duplicates are removed (same as the official scheduler).
            """
            if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
                raise ValueError("Steps and shifts must be positive.")
            total = num_train_timesteps

            def _sigma_grid(count: int, shift: float, total: int) -> torch.Tensor:
                lower = shift / (total + shift - 1)
                values = torch.linspace(1, lower, count, dtype=torch.float32)
                return shift * values / (1 + (shift - 1) * values)
            sigmas = _sigma_grid(num_inference_steps, shift, total)
            training = _sigma_grid(total, training_shift, total).flip(0)  # ascending
            upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
            lower = (upper - 1).clamp(min=0)
            nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
            sigmas = torch.unique_consecutive(training[nearest])
            return torch.cat([sigmas, sigmas.new_zeros(1)])

        sigmas = get_sampling_sigmas(steps, flow_shift).to(device)
        num_steps = len(sigmas) - 1

        def euler_step(latents: torch.Tensor, model_pred: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
            """One Euler step in float32: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v"""
            dt = (sigmas[step_index + 1] - sigmas[step_index]).to(torch.float32)
            return latents.float() + dt * model_pred.float()

        from tqdm import tqdm
        with torch.no_grad(), tqdm(total=num_steps, desc="Denoising steps", disable=not show_progress) as pbar:
            for i in range(num_steps):
                timestep = (sigmas[i] * num_train_timesteps).expand(latents.shape[0])
                x = latents.to(prefer_dtype)
                prediction = self.get_model()(x, timestep, positive, positive_mask).float()
                if do_cfg:
                    uncond = self.get_model()(x, timestep, negative, negative_mask, skip_layers=uncond_skip_layers).float()
                    prediction = uncond + guidance_scale * (prediction - uncond)
                latents = euler_step(latents, prediction, sigmas, i)
                pbar.update()

        return latents
    
    @abc.abstractmethod
    def get_dit_blocks_module(self) -> torch.nn.ModuleList:
        raise NotImplementedError()
    
class AutoEncoder(abc.ABC):
    @abc.abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def load_model(self, checkpoint: str, device: torch.device | str = "cpu", dtype: torch.dtype | None = None):
        """
        ae都是走eval且requires_grad_(false)的方式
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abc.abstractmethod
    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def decode_from_dit(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def encode_for_dit(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    
class Module(abc.ABC):
    @abc.abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_architecture(self) -> typing.Tuple[str, str]:
        """
        返回缩写和全名，比如 "kr2", "krea2"。\n
        根据这个值构建Blueprint和图像桶，定义在musubi_tuner.dataset.architectures。
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def create_conditioner(self) -> Conditioner:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def create_ditnetwork(self) -> DiTNetwork:
        raise NotImplementedError()

    @abc.abstractmethod
    def create_autoencoder(self) -> AutoEncoder:
        raise NotImplementedError()

    

    @classmethod
    def __load_dataset(cls, args, short_arch: str):
        from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
        from musubi_tuner.dataset import config_utils
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=short_arch)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

        datasets = train_dataset_group.datasets

        import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
        all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
        return datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset

    def setup_cache_text_encoder_output_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--text_encoder",
            type=str,
            required=True,
            help=f"{self.create_conditioner().get_name()} text encoder safetensors path (official or ComfyUI key layout)",
        )
        return parser

    def main_cache_text_encoder_output(self):
        import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
        parser = cache_text_encoder_outputs.setup_parser_common()
        parser = self.setup_cache_text_encoder_output_parser(parser)
        args = parser.parse_args()

        # Load dataset config
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset = self.__load_dataset(args, self.get_model_architecture()[0])

        encoder = self.create_conditioner()
        encoder.load_model([args.text_encoder])
        logger.info(f"Encoding with {encoder.get_name()}")

        def encode_for_text_encoder(batch: list[ItemInfo]):
            nonlocal encoder
            prompts: typing.List[str] = [item.caption for item in batch]
            results = encoder.encode_text(prompts)

            for i, item in enumerate(batch):
                print(f"Item {i}: {item.item_key}, prompt: {item.caption}")

                sd = {}
                for key in results.keys():
                    if key.endswith("_mask"): continue
                    value = results[key][i]
                    mask_key = key + "_mask"
                    if mask_key in results:
                        value = shrink_hidden_state(value, results[mask_key][i])
                    sd[key] = value
                from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_common
                save_text_encoder_output_cache_common(item, sd, self.get_model_architecture()[1])

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

    def main_cache_latents(self):
        import musubi_tuner.cache_latents as cache_latents
        parser = cache_latents.setup_parser_common()
        parser = cache_latents.hv_setup_parser(parser)  # VAE

        args = parser.parse_args()

        # if args.vae_dtype is not None:
        #     raise ValueError("VAE dtype is not supported in Moss Image (uses the Flux small VAE default).")

        device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)

        # Load dataset config
        datasets, _, _ = self.__load_dataset(args, self.get_model_architecture()[0])

        if args.debug_mode is not None:
            cache_latents.show_datasets(
                datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
            )
            return

        assert args.vae is not None, "VAE checkpoint is required"

        logger.info(f"Loading VAE model from {args.vae}")
        vae = self.create_autoencoder()
        vae.load_model(args.vae, device=device)

        def encode(batch: typing.List[ItemInfo]):
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
                latents = vae.encode_for_dit(contents)

            for b, item in enumerate(batch):
                target_latent = latents[b].unsqueeze(dim=0)  # (1, C, H, W)
                print(f"Saving cache for item {item.item_key} at {item.latent_cache_path}, latents shape: {target_latent.shape}")

                sd = {f"latents": target_latent.detach().cpu().contiguous()}
                from musubi_tuner.dataset.cache_io import save_latent_cache_common
                save_latent_cache_common(item, sd, self.get_model_architecture()[1])



        cache_latents.encode_datasets(datasets, encode, args)



    def setup_generate_image_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser
    
    def main_generate_image(self):
        parser = argparse.ArgumentParser(description=f"Generate images with {self.get_name()}.")
        parser.add_argument("prompt", type=str, nargs="?", default=None, help="Prompt for image generation (single-prompt mode)")
        parser.add_argument("--dit", type=str, required=True, help="Path to the MMDiT checkpoint (.safetensors)")
        parser.add_argument("--vae", type=str, required=True, help="Path to the VAE checkpoint (.safetensors)")
        parser.add_argument(
            "--text_encoder",
            type=str,
            required=True,
            help="Text encoder safetensors path (official or ComfyUI key layout)",
        )
        parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (used only when --guidance_scale > 1)")
        parser.add_argument("--steps", type=int, default=28, help="Number of denoising steps")
        parser.add_argument("--guidance_scale", type=float, default=5.5, help="Classifier-free guidance scale (<= 1 disables CFG)")
        parser.add_argument("--y1", type=float, default=0.5, help="Timestep-shift mu at min resolution")
        parser.add_argument("--y2", type=float, default=1.15, help="Timestep-shift mu at max resolution")
        parser.add_argument("--mu", type=float, default=None, help="Pin a constant timestep-shift mu (overrides y1/y2)")
        parser.add_argument("--width", type=int, default=1024, help="Width of the generated image")
        parser.add_argument("--height", type=int, default=1024, help="Height of the generated image")
        parser.add_argument("--num-images", dest="num_images", type=int, default=1, help="Number of images to generate")
        parser.add_argument(
            "--seed", type=int, default=None, help="Base seed; image i uses seed + i. If omitted, a random seed is used."
        )
        parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
        parser.add_argument(
            "--attn_mode",
            type=str,
            default="torch",
            choices=["torch", "flash", "sageattn", "xformers"],
            help="attention backend ('torch'=SDPA, 'flash', 'sageattn', 'xformers')",
        )
        parser.add_argument(
            "--split_attn",
            action="store_true",
            help="split attention per sample (recommended for non-sdpa backends; required for xformers + GQA)",
        )
        parser.add_argument(
            "--blocks_to_swap",
            type=int,
            default=0,
            help="number of main SingleStreamBlocks to offload to CPU (block swap). Max 26 (= 28 - 2). "
            "Reduces VRAM at the cost of speed.",
        )
        parser.add_argument(
            "--save_path",
            type=str,
            required=True,
            help="Directory to save generated images. File names are auto-generated from a timestamp and the seed.",
        )
        parser.add_argument(
            "--bell",
            action="store_true",
            help="Ring the terminal bell after each prompt (interactive) or at the end (other modes)",
        )
        parser = self.setup_generate_image_parser(parser)
        args = parser.parse_args()

        # Load all three models up front and keep them: the encoder and VAE live on CPU, the DiT
        # lives on the GPU (with block swap as needed). Encoder/VAE shuttle to the GPU per prompt.
        encoder = self.create_conditioner()
        encoder.load_model([args.text_encoder])

        def build_pipeline():
            ae = self.create_autoencoder()
            ae.load_model(args.vae)

            loading_device = "cpu" if args.blocks_to_swap > 0 else args.device
            mmdit = self.create_ditnetwork()
            mmdit.load_model(
                dit_path=args.dit,
                attn_mode=args.attn_mode,
                split_attn=args.split_attn,
                device=args.device,
                loading_device=loading_device,
            )
            mmdit.get_model().eval().requires_grad_(False)

            if args.blocks_to_swap > 0:
                raise NotImplementedError()
            else:
                mmdit.to(device=args.device)

            return mmdit, ae

        dit, ae = build_pipeline()

        def encode(prompts, negative_prompts, cfg: bool):
            """Encode prompts with the Qwen3-VL conditioner, shuttling it to ``te_device`` for the call.

            The encoder is moved to ``te_device`` (the GPU, or CPU with ``--text_encoder_cpu``) just for
            the encode, then moved back to CPU so the denoise/decode that follow have the VRAM to
            themselves. The returned embeddings are tiny and ``sample`` moves them to the compute device.
            """
            logger.info(f"Encoding prompts with {encoder.get_name()}")
            def decompose_result(result: typing.Dict[str, torch.Tensor]):
                return result["embeds"], result["embeds_mask"]
            txt, txtmask = decompose_result(encoder.encode_text(prompts))
            untxt, untxtmask = None, None
            if cfg:
                untxt, untxtmask = decompose_result(encoder.encode_text(negative_prompts))
            embeds = (txt, txtmask, untxt, untxtmask)
            return embeds  # (txt, txtmask, untxt, untxtmask)

        @torch.no_grad()
        def sample(
            txt,
            txtmask,
            *,
            untxt=None,
            untxtmask=None,
            device="cuda",
            dtype=torch.bfloat16,
            width=1024,
            height=1024,
            steps=28,
            cfg_scale=5.5,
            seed=0,
            minres=256,
            maxres=1280,
            y1=0.5,
            y2=1.15,
            mu=None,
        ):
            scale = cfg_scale
            # flow_shift = float(prompt_dict.get("flow_shift", DEFAULT_SAMPLE_FLOW_SHIFT))
            flow_shift = 5.0
            # uncond_skip_layers = parse_skip_layers(args.sample_uncond_skip_layers)
            uncond_skip_layers = ()

            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

            height = max(16, height - height % 16)
            width = max(16, width - width % 16)

            # clean_memory_on_device(accelerator.device)
            latents = dit.generate_latents(
                txt,
                txtmask,
                untxt,
                untxtmask,
                height,
                width,
                steps,
                cfg_scale,
                flow_shift,
                uncond_skip_layers,
                seed,
                device=device,
            )

            # Decode latents in float32 outside autocast (the VAE is in float32)
            org_vae_device = ae.get_model().device
            ae.to(device)
            with torch.autocast(device_type=device, enabled=False):
                pixels = ae.decode_from_dit(latents.to(device, dtype=ae.get_model().dtype))
            ae.to(org_vae_device)

            pixels = ((pixels[0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
            from PIL import Image
            image = Image.fromarray(pixels)

            return [image]


        def generate():
            """Run one prompt end to end: encode -> denoise -> decode, returning a list of PIL images.

            The DiT (``dit``) and VAE (``ae``) are reused across calls. Per call the encoder is shuttled
            to ``te_device`` for the encode and the VAE is shuttled to the GPU for the decode (both inside
            the helpers), while the DiT stays resident.

            The base seed is resolved here: if ``args.seed`` is None a random one is drawn (and written
            back to ``args.seed`` so the caller can use it for file names); image *i* uses ``base + i``.
            """
            if args.seed is None:
                import random
                args.seed = random.randint(0, 2**32 - 1)
                logger.info(f"No seed specified; using random seed {args.seed}")

            prompts = [args.prompt] * args.num_images
            negative_prompts = [args.negative_prompt] * args.num_images
            cfg = args.guidance_scale > 1.0

            txt, txtmask, untxt, untxtmask = encode(prompts, negative_prompts, cfg)

            # Re-arm the block-swap offloader before each forward (no-op when block swap is off).
            if args.blocks_to_swap > 0:
                raise NotImplementedError()
                dit.prepare_block_swap_before_forward()

            images = sample(
                txt,
                txtmask,
                untxt=untxt,
                untxtmask=untxtmask,
                device=args.device,
                dtype=dit.get_prefer_dtype(),
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.guidance_scale,
                seed=args.seed,
                y1=args.y1,
                y2=args.y2,
                mu=args.mu,
            )
            return images
        
        images = generate()

        def save_images(images, save_path: str, base_seed: int):
            """Save a list of PIL images into ``save_path`` (created if needed).

            File names are auto-generated as ``<timestamp>_<seed>.png``; image *i* uses ``base_seed + i``,
            so a single timestamp + per-image seed keeps the names unique within and across prompts.
            """
            def get_time_flag() -> str:
                """Millisecond timestamp string used to make output file names unique (e.g. 20260623-143012-123)."""
                from datetime import datetime
                import time
                return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]

            os.makedirs(save_path, exist_ok=True)
            time_flag = get_time_flag()
            for i, image in enumerate(images):
                out = os.path.join(save_path, f"{time_flag}_{base_seed + i}.png")
                image.save(out)
                logger.info(f"saved {out}")

        save_images(images, args.save_path, args.seed)
        if args.bell:
            print("\a")

    def setup_train_network_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def main_train_network(self):
        from musubi_tuner.hv_train_network import (
            DiTOutput,
            NetworkTrainer,
            clean_memory_on_device,
            load_prompts,
            setup_parser_common,
            read_config_from_file,
        )
        that = self
        class SubNetworkTrainer(NetworkTrainer):
            def __init__(self):
                super().__init__()
                self.snt_dit = that.create_ditnetwork()
                self.snt_vae = that.create_autoencoder()

            def handle_model_specific_args(self, args: argparse.Namespace):
                self._i2v_training = False
                self._control_training = False
                self.default_guidance_scale = 1.0  # K2 t2i, not used at train time

            @property
            def architecture(self) -> str:
                return that.get_model_architecture()[0]

            @property
            def architecture_full_name(self) -> str:
                return that.get_model_architecture()[1]

            def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
                logger.info(f"Loading VAE model from {args.vae}")
                self.snt_vae.load_model(args.vae)
                return self.snt_vae.get_model()
            
            def load_transformer(self, accelerator: Accelerator, args: argparse.Namespace, dit_path: str, attn_mode: str, split_attn: bool, loading_device: str, dit_weight_dtype: torch.dtype | None):
                self.snt_dit.load_model(
                    dit_path=args.dit,
                    attn_mode=attn_mode,
                    split_attn=args.split_attn,
                    device=accelerator.device,
                    loading_device=loading_device
                )
                return self.snt_dit.get_model()
            
            def compile_transformer(self, args, transformer):
                model = transformer  # SingleStreamDiT
                # Compile the per-block SingleStreamBlocks (the heavy, repeated compute). The forward
                # already pads the combined sequence to a multiple of 256 to keep kernel shapes stable.
                # When block swap is on, exclude the swap blocks' Linears from compile (cf. zimage/qwen_image).
                # ConvRot int8 Linears are also excluded: the custom autograd.Function + autotuned Triton
                # kernels are not dynamo-traceable.
                disable_linear = self.blocks_to_swap > 0
                from musubi_tuner.utils import model_utils
                return model_utils.compile_transformer(args, model, [self.snt_dit.get_dit_blocks_module()], disable_linear=disable_linear)
            
            def scale_shift_latents(self, latents):
                return latents
            
            def call_dit(
                self,
                args: argparse.Namespace,
                accelerator: Accelerator,
                transformer,
                latents: torch.Tensor,
                batch: dict[str, torch.Tensor],
                noise: torch.Tensor,
                noisy_model_input: torch.Tensor,
                timesteps: torch.Tensor,
                network_dtype: torch.dtype,
                **kwargs,
            ) -> DiTOutput:
                contexts = self.snt_dit.get_contexts_from_batch(batch)
                model_pred = self.snt_dit.call_dit(noisy_model_input, timesteps, contexts, accelerator, {
                    "gradient_checkpointing": args.gradient_checkpointing,
                })
                # Rectified flow target: v = noise - x0
                target = noise - latents
                return DiTOutput(pred=model_pred, target=target)
        
        from musubi_tuner.training.parser_common import setup_parser_common
        from musubi_tuner.hv_train_network import read_config_from_file
        parser = setup_parser_common()
        parser = that.setup_train_network_parser(parser)
        args = parser.parse_args()
        args = read_config_from_file(args, parser)

        trainer = SubNetworkTrainer()
        args.dit_dtype = str(trainer.snt_dit.get_prefer_dtype()).strip("torch.")
        trainer.train(args)

    def convert_lora_statedict_to_comfy(self, sd: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        suffix = [ "q", "k", "v" ]
        lora_ud = ["lora_down", "lora_up"]
        for k in list(sd.keys()):
            if k not in sd: continue
            sk = k.split('.')
            if sk[0].endswith("_qkv") and sk[1] in lora_ud and sk[2].isnumeric() and sk[3] == "weight":
                basename = sk[0][:-len('_qkv')]
                renames = {}
                for ind, sf in enumerate(suffix):
                    for ln in lora_ud:
                        old_name = f"{basename}_qkv.{ln}.{ind}.weight"
                        new_name = f"{basename}_{sf}.{ln}.weight"
                        assert old_name in sd
                        renames[old_name] = new_name
                for old_name, new_name in renames.items():
                    v = sd.pop(old_name)
                    sd[new_name] = v
                alpha_name = f"{basename}_qkv.alpha"
                assert alpha_name in sd
                alpha = sd.pop(alpha_name)
                for sf in suffix:
                    new_name = f"{basename}_{sf}.alpha"
                    sd[new_name] = alpha.clone()
        return sd

    def main_convert_lora_to_comfy(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, required=True)
        parser.add_argument("--output", type=str, required=True)
        args = parser.parse_args()

        import safetensors
        with safetensors.safe_open(args.input, framework="pt") as f:
            sd = { k: f.get_tensor(k) for k in f.keys() }
            md = f.metadata()

        sd = self.convert_lora_statedict_to_comfy(sd)

        import safetensors.torch
        safetensors.torch.save_file(sd, args.output, md)
        print(f"Save to {args.output} successfully.")


