import os
import json

from src.models.predict import load_model

model = load_model()


from pathlib import Path
base_path = Path(__file__).parent
img_file = str((base_path / "cell.png").resolve())
print(img_file)
inp = {"file": img_file}

model.predict(inp)