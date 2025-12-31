# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('assets', 'assets'),
        ('config.yaml', '.'),
    ],
    hiddenimports=[
        # Core ML dependencies
        'h5py',
        'tensorflow',
        'tensorflow.python',
        'tensorflow.python.ops',
        'tensorflow.python.keras',
        'tensorflow.python.keras.engine',
        'tensorflow.python.keras.layers',
        'tensorflow.python.keras.models',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.preprocessing',
        'xgboost',
        'numba',

        # GUI dependencies
        'ttkbootstrap',
        'PIL._tkinter_finder',
        'PIL.Image',
        'PIL.ImageTk',

        # Data processing
        'pandas',
        'numpy',
        'scipy.stats',
        'pvlib.solarposition',

        # Other utilities
        'yaml',
        'requests',
    ],
    hookspath=['.'],
    hooksconfig={},
    excludes=[
        # Exclude large unused packages
        'matplotlib',
        'matplotlib.pyplot',
        'IPython',
        'jupyter',
        'jupyter_client',
        'jupyter_core',
        'notebook',
        'nbconvert',
        'nbformat',
        'jedi',
        'parso',

        # Exclude Qt (you're using tkinter)
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',

        # Exclude test frameworks
        'pytest',
        'nose',
        '_pytest',

        # Exclude development tools
        'setuptools',
        'pip',
        'wheel',
        'distutils',

        # Exclude documentation tools
        'sphinx',
        'docutils',

        # Exclude plotly and related
        'plotly',
        'kaleido',

        # Exclude other data science tools you don't use
        'statsmodels',
        'patsy',
        'seaborn',
        'bokeh',

        # Exclude database drivers you don't use
        'sqlalchemy',
        'MySQLdb',
        'psycopg2',
        'pysqlite2',

        # Exclude web frameworks
        'flask',
        'django',
        'tornado',

        # Exclude xarray and netCDF (if you don't use them)
        'xarray',
        'netCDF4',
        'cftime',

        # Exclude lxml if not needed
        'lxml',
        'lxml.etree',

        # Exclude problematic phdu modules
        'phdu',

        # Unused tensorflow dependencies
        'tensorboard',
        'tensorflow.contrib',

        # Exclude CUDA/GPU support
        'nvidia.cublas',
        'nvidia.cuda_cupti',
        'nvidia.cuda_nvcc',
        'nvidia.cuda_nvrtc',
        'nvidia.cuda_runtime',
        'nvidia.cudnn',
        'nvidia.cufft',
        'nvidia.curand',
        'nvidia.cusolver',
        'nvidia.cusparse',
        'nvidia.nccl',
        'nvidia.nvjitlink',

        ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DMSC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False, # change to True for smaller file size
    upx=True,
    upx_exclude=[],
    name='DMSC',
)
