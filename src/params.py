import os
from pathlib import Path

_app_path = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_app_path)
DATA_PATH: Path = Path(os.path.join(ROOT, 'data'))
