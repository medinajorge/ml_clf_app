import os
from pathlib import Path

# PATH
_app_path = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_app_path)
BATHYMETRY_PATH: Path = Path(os.path.join(_app_path, 'data/BathymetryData.dat'))

DATA_DIR: Path = Path(os.path.join(ROOT, 'data'))
SCALER_DIR: Path = Path(os.path.join(_app_path, 'scaler'))
CLF_MODEL_DIR: Path = Path(os.path.join(_app_path, 'clf_model'))
CONF_MODEL_DIR: Path = Path(os.path.join(_app_path, 'conf_model'))
ASSETS_DIR: Path = Path(os.path.join(ROOT, 'assets'))

# MODEL
FEATURES = ['x', 'y', 'z', 'sin t', 'cos t', 'sin h', 'cos h', 'bathymetry', ' dt', 'v']
FEATURES_SCALE = ['bathymetry', ' dt', 'v']
NUM_FEATURES_SCALE = 3

MAX_SEQUENCE_LENGTH = 512

CATEGORY_TO_SPECIES = {0: 'Adelie penguin',
                       1: 'Arctic Herring gull',
                       2: 'Ascension frigatebird',
                       3: 'Atlantic puffin',
                       4: 'Australian sea lion',
                       5: 'Baraus petrel',
                       6: 'Beluga whale',
                       7: 'Black-browed albatross',
                       8: 'Black-footed albatross',
                       9: 'Blue shark',
                       10: 'Blue whale',
                       11: 'Bowhead whale',
                       12: 'Bullers albatross',
                       13: 'California sea lion',
                       14: 'Chinstrap penguin',
                       15: 'Common eider',
                       16: 'Corys shearwater',
                       17: 'Dugong',
                       18: 'Emperor penguin',
                       19: 'Fin whale',
                       20: 'Galapagos sea lion',
                       21: 'Great shearwater',
                       22: 'Green turtle',
                       23: 'Grey seal',
                       24: 'Grey-headed albatross',
                       25: 'Harbour seal',
                       26: 'Hawksbill turtle',
                       27: 'Humpback whale',
                       28: 'Ivory gull',
                       29: 'Kemps Ridley turtle',
                       30: 'Killer whale',
                       31: 'King eider',
                       32: 'Laysan albatross',
                       33: 'Leatherback turtle',
                       34: 'Leopard seal',
                       35: 'Little penguin',
                       36: 'Loggerhead turtle',
                       37: 'Long-nosed fur seal',
                       38: 'Macaroni penguin',
                       39: 'Manx shearwater',
                       40: 'Masked booby',
                       41: 'Murphys petrel',
                       42: 'Narwhal',
                       43: 'New Zealand sea lion',
                       44: 'Northern elephant seal',
                       45: 'Northern fulmar',
                       46: 'Northern fur seal',
                       47: 'Northern gannet',
                       48: 'Oceanic whitetip shark',
                       49: 'Olive Ridley turtle',
                       50: 'Polar bear',
                       51: 'Red-tailed tropic bird',
                       52: 'Reef manta ray',
                       53: 'Ringed seal',
                       54: 'Sabines gull',
                       55: 'Salmon shark',
                       56: 'Scopolis shearwater',
                       57: 'Short-finned pilot whale',
                       58: 'Short-tailed shearwater',
                       59: 'Shortfin mako shark',
                       60: 'Sooty tern',
                       61: 'Southern elephant seal',
                       62: 'Sperm whale',
                       63: 'Streaked shearwater',
                       64: 'Thick-billed murre',
                       65: 'Tiger shark',
                       66: 'Trindade petrel',
                       67: 'Wandering albatross',
                       68: 'Weddell seal',
                       69: 'Wedge-tailed shearwater',
                       70: 'Western gull',
                       71: 'Whale shark',
                       72: 'White shark',
                       73: 'White-tailed tropic bird',
                       }


# GUI APP
FONTSIZE_BUTTON = 11
FONTSIZE_INFO = 10
FONTSIZE_TITLE = 15
FONTSIZE_STATUS = 9
