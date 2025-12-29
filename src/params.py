import os
from pathlib import Path

_app_path = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_app_path)
DATA_PATH: Path = Path(os.path.join(ROOT, 'data'))
BATHYMETRY_PATH: Path = Path(os.path.join(_app_path, 'data/BathymetryData.dat'))

FEATURES = ['x', 'y', 'z', 'sin t', 'cos t', 'sin h', 'cos h', 'bathymetry', ' dt', 'v']
