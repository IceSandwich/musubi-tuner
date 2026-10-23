import sys
from pathlib import Path
module_src = Path(__file__).parent / "src"
sys.path.append(str(module_src.absolute()))

from musubi_tuner.moss_image.moss_image_adapter import MossImageAdapter

if __name__ == "__main__":
    adapter = MossImageAdapter()
    adapter.main_train_network()
