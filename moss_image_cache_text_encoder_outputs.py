# from musubi_tuner.moss_image_cache_text_encoder_outputs import main

from musubi_tuner.moss_image.moss_image_adapter import MossImageAdapter

if __name__ == "__main__":
    adapter = MossImageAdapter()
    adapter.main_cache_text_encoder_output()
