import logging

import torch
from transformers import (
    Qwen3Model
)

from musubi_tuner.utils.safetensors_utils import load_split_weights

from . import anima_utils
from typing import Optional, Union, List, Dict

import os
import safetensors

logger = logging.getLogger(__name__)


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(qwen3_path, torch_dtype=dtype, local_files_only=True).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(__file__), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        with safetensors.safe_open(qwen3_path, framework="pt", device="cpu") as f:
            state_dict = { k: f.get_tensor(k) for k in f.keys() }

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model.") :]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_anima_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    qwen3_max_length: int = 512,
    t5_max_length: int = 512,
    t5_tokenizer_path: Optional[str] = None,
):
    tokenizer = AnimaTokenizer(
        qwen3_path=model_path, 
        t5_tokenizer_path=t5_tokenizer_path,
        qwen3_max_length=qwen3_max_length,
        t5_max_length=t5_max_length,
    )
    # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
    qwen3_text_encoder, _ = load_qwen3_text_encoder(
        qwen3_path=model_path, 
        dtype=dtype,
        device=device,
    )
    return AnimaConditioner(
        qwen=qwen3_text_encoder,
        tokenizer=tokenizer
    )


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(__file__), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(__file__), "configs", "t5_old")
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


class AnimaTokenizer():
    """Tokenize strategy for Anima: dual tokenization with Qwen3 + T5.

    Qwen3 tokens are used for the text encoder.
    T5 tokens are used as target input IDs for the LLM Adapter (NOT encoded by T5).

    Can be initialized with either pre-loaded tokenizer objects or paths to load from.
    """

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen3_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
    ) -> None:
        # Load tokenizers from paths if not provided directly
        if qwen3_tokenizer is None:
            if qwen3_path is None:
                raise ValueError("Either qwen3_tokenizer or qwen3_path must be provided")
            qwen3_tokenizer = load_qwen3_tokenizer(qwen3_path)
        if t5_tokenizer is None:
            t5_tokenizer = load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

    def tokenize(self, text: Union[str, List[str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        # Tokenize with Qwen3
        qwen3_encoding = self.qwen3_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.qwen3_max_length
        )
        qwen3_input_ids: torch.Tensor = qwen3_encoding["input_ids"]
        qwen3_attn_mask: torch.Tensor = qwen3_encoding["attention_mask"]

        # Tokenize with T5 (for LLM Adapter target tokens)
        t5_encoding = self.t5_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.t5_max_length
        )
        t5_input_ids: torch.Tensor = t5_encoding["input_ids"]
        t5_attn_mask: torch.Tensor = t5_encoding["attention_mask"]
        return qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask

class AnimaConditioner(torch.nn.Module):
    def __init__(self, qwen: Qwen3Model, tokenizer: AnimaTokenizer):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        
    def forward(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # AnimaTokenizeStrategy
        qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = self.tokenizer.tokenize(text)

        # AnimaTextEncodingStrategy
        qwen3_text_encoder = self.qwen
        encoder_device = qwen3_text_encoder.device

        qwen3_input_ids = qwen3_input_ids.to(encoder_device)
        qwen3_attn_mask = qwen3_attn_mask.to(encoder_device)
        outputs = qwen3_text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
        prompt_embeds = outputs.last_hidden_state
        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        return prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask

