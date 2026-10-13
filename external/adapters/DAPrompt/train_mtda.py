#!/usr/bin/env python3
"""Single-model, three-target RN50 benchmark entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spl"))
from train_vlm_competitors import main
if __name__ == "__main__":
    main(default_method="DAPrompt")
