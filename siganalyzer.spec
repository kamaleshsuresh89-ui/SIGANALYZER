import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Hidden imports required for dynamic loading
hidden_imports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'pyqtgraph',
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.fft',
    'numba',
    'onnxruntime',
    'jinja2',
    'sqlite3',
    'pydantic',
    'pydantic_core',
    'sklearn',
]
hidden_imports += collect_submodules('siganalyzer')

# Assets and resources
datas = []
if os.path.isdir('assets'):
    datas.append(('assets', 'assets'))

a = Analysis(
    ['src/siganalyzer/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'torch', 'tensorflow', 'IPython'],
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
    name='siganalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=(sys.platform == 'darwin'),
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='siganalyzer',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='SIGANALYZER.app',
        icon=None,
        bundle_identifier='com.siganalyzer.desktop',
    )
