# -*- mode: python ; coding: utf-8 -*-

import sys
import os

# --- INICIO DE SECCIÓN MODIFICABLE ---
file_description = 'OptiPro - Sistema de Optimización Logística'
product_name = 'OptiPro' # Nombre del producto (se usará para product_version)
author_name = 'Equipo Tec: Leo, Eduardo, Adid, Mario'
copyright_info = '© 2025 Equipo Tec. Todos los derechos reservados.'
# --- FIN DE SECCIÓN MODIFICABLE ---

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('img', 'img')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='optipro', # Nombre del archivo ejecutable de salida
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['img\\optipro.ico'],
    description=file_description,
    company=author_name,
    product_version=product_name,
    copyright=copyright_info,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='optipro',
)
