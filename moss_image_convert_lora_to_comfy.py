import sys
from pathlib import Path
module_src = Path(__file__).parent / "src"
sys.path.append(str(module_src.absolute()))

from musubi_tuner.moss_image.moss_image_adapter import MossImageAdapter
import torch

if __name__ == "__main__":
    adapter = MossImageAdapter()
    adapter.main_convert_lora_to_comfy()
    # ae = adapter.create_autoencoder()
    # ae.load_model("..\\ComfyUI\\models\\vae\\full_encoder_small_decoder.safetensors", device="cuda")
    # import safetensors
    # with safetensors.safe_open(R"1_1500x1261_mossi.safetensors", framework="pt") as f:
    #     sd = { k: f.get_tensor(k) for k in f.keys() }
    # with torch.autocast(device_type="cuda", enabled=False):
    #     latents = sd["latents"]
    #     pixels = ae.decode_from_dit(latents.to("cuda", dtype=ae.get_model().dtype))

    # pixels = ((pixels[0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    # from PIL import Image
    # image = Image.fromarray(pixels)
    # image.save("references/test.png")
