from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
from pathlib import Path
import xgboost

# Collect XGBoost shared libraries
binaries = collect_dynamic_libs('xgboost')

# Also manually add the lib directory
xgb_path = Path(xgboost.__file__).parent
lib_dir = xgb_path / 'lib'

if lib_dir.exists():
    import glob
    for lib_file in lib_dir.glob('libxgboost.*'):
        binaries.append((str(lib_file), 'xgboost/lib'))

# Collect any data files
datas = collect_data_files('xgboost')

# Hidden imports
hiddenimports = [
    'xgboost.core',
    'xgboost.libpath',
    'xgboost.collective',
]
