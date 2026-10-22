import typing
import torch
import torch.nn
from ice_scripts import general_adapter
from ice_scripts.comfyops import comfy
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS, ARCHITECTURE_MOSS_FULL

class Qwen35_9BConditioner(general_adapter.Conditioner):
    def get_name(self):
        return "Qwen3.5 9B"
    
    def load_model(self, checkpoints: typing.List[str]):
        assert len(checkpoints) == 1, f"Expect one checkpoint but got {len(checkpoints)} checkpoints"
        import comfy.sd
        self.clip = comfy.sd.load_clip(
            ckpt_paths=[checkpoints[0]],
            clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
        )
        clip_type = self.clip.cond_stage_model.clip
        assert clip_type == "qwen35_9b", f"Wrong text encoder. Got {clip_type}. Expect Qwen3.5 9B."

    def encode_text(self, text: typing.List[str]) -> typing.Dict[str, torch.Tensor]:
        """
        Args:
            text: list of caption strings, length = batch_size

        Returns:
            hidden_states: Tensor[batch_size, max_seq_len, hidden_dim]
            attention_mask: Tensor[batch_size, max_seq_len]
        """

        all_hidden_states = []

        for caption in text:
            # 1. tokenize
            tokens = self.clip.tokenize(caption)

            # 2. drop token weights (keep only token ids)
            token_ids = [tid for (tid, _) in tokens["qwen35_9b"][0]]
            tokens["qwen35_9b"][0] = [(tid, 1.0) for tid in token_ids]

            # 3. encode
            cond = self.clip.encode_from_tokens(
                tokens,
                return_pooled=False,
                return_dict=False
            )
            # cond: Tensor[1, seq_len, hidden_dim]

            all_hidden_states.append(cond.squeeze(dim=0))

        # 4. pad to max length
        embeds, mask = general_adapter.pad_text_embeds(all_hidden_states)

        return {
            "embeds": embeds,
            "embeds_mask": mask,
        }

class MossDiT(general_adapter.DiTNetwork):
    def get_name(self) -> str:
        return "MossImage 5B DiT"
    
    def load_model(self, dit_path: str, attn_mode: str, split_attn: bool, device: torch.device | str, loading_device: torch.device | str):
        from . import moss_image_utils
        self.model = moss_image_utils.load_moss_image_model(
            device=device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=None
        )
    
    def get_model(self) -> torch.nn.Module:
        return self.model
    
    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        self.model = self.model.to(device=device, dtype=dtype)

    def get_latent_shape(self, height: int, width: int) -> typing.Tuple[int, int, int, int]:
        return 1, 128, height // 16, width // 16
    


class Flux2AE(general_adapter.AutoEncoder):
    def get_name(self) -> str:
        return "Flux 2 AE"
    
    def load_model(self, checkpoint: str, device: str = "cpu", dtype: torch.dtype | None = None):
        from . import flux2_vae
        self.ae = flux2_vae.load_flux2_vae(checkpoint, dtype=dtype, device=device, disable_mmap=True)
        self.ae = self.ae.to(dtype=dtype).eval().requires_grad_(False)

    def get_model(self) -> torch.nn.Module:
        return self.ae

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        self.ae = self.ae.to(device=device, dtype=dtype)

    def decode_from_dit(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.ae.decode_from_dit(tensor)
    
    def encode_for_dit(self, tensor: torch.Tensor) -> torch.Tensor:
        latents = self.ae.encode(tensor.to(self.ae.device, dtype=self.ae.dtype))  # (B, C, 1, H, W)
        latents = self.ae.patchify_and_normalize(latents)
        return latents

class MossImageAdapter(general_adapter.Module):
    def get_name(self) -> str:
        return "Moss Image"
    
    def get_model_architecture(self) -> typing.Tuple[str, str]:
        return ARCHITECTURE_MOSS, ARCHITECTURE_MOSS_FULL
    
    def create_conditioner(self) -> general_adapter.Conditioner:
        return Qwen35_9BConditioner()
    
    def create_ditnetwork(self) -> general_adapter.DiTNetwork:
        return MossDiT()
    
    def create_autoencoder(self) -> general_adapter.AutoEncoder:
        return Flux2AE()
    
    
    
