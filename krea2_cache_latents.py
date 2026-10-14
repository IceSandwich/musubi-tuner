import sys
from pathlib import Path
module_src = Path(__file__).parent / "src"
sys.path.append(str(module_src.absolute()))

from musubi_tuner.krea2_cache_latents import main

if __name__ == "__main__":
    main()
