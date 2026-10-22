import logging
import torch
from torch import Tensor
from typing import Tuple

logger = logging.getLogger(__name__)

from ice_scripts.comfyops import comfy
import comfy.sd

def load_qwen35_9b_conditioner(
    model_path: str,
):
    return Qwen35_9BConditioner(model_path)

class Qwen35_9BConditioner(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.clip = comfy.sd.load_clip(
            ckpt_paths=[path],
            clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
        )
        clip_type = self.clip.cond_stage_model.clip
        assert clip_type == "qwen35_9b", f"Wrong text encoder. Got {clip_type}. Expect Qwen3.5 9B."

    @torch.inference_mode()
    def forward(self, text: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text: list of caption strings, length = batch_size

        Returns:
            hidden_states: Tensor[batch_size, max_seq_len, hidden_dim]
            attention_mask: Tensor[batch_size, max_seq_len]
        """

        batch_size = len(text)
        all_hidden_states = []
        all_masks = []

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
            all_masks.append(
                torch.ones(cond.shape[1], dtype=torch.long)
            )

        # 4. pad to max length
        from . import moss_image_utils
        return moss_image_utils.pad_text_embeds(all_hidden_states)
        max_len = max(h.shape[1] for h in all_hidden_states)
        hidden_dim = all_hidden_states[0].shape[-1]

        padded_hidden = torch.zeros(
            batch_size, max_len, hidden_dim,
            dtype=all_hidden_states[0].dtype,
            device=all_hidden_states[0].device
        )

        padded_mask = torch.zeros(
            batch_size, max_len,
            dtype=torch.long,
            device=all_hidden_states[0].device
        )

        for i in range(batch_size):
            seq_len = all_hidden_states[i].shape[1]
            padded_hidden[i, :seq_len] = all_hidden_states[i]
            padded_mask[i, :seq_len] = all_masks[i]

        return padded_hidden, padded_mask