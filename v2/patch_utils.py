# -*- coding: utf-8 -*-
# patch_utils.py
import os
import re
import numpy as np
from typing import Tuple, List, Optional

ORIENTATIONS = {
    "axial":    lambda vol: vol,                      # (Y, X, Z)
    "coronal":  lambda vol: np.transpose(vol, (1, 0, 2)),
    "sagittal": lambda vol: np.transpose(vol, (2, 0, 1)),
}

def scandir_paths(root: str, pattern: str) -> List[str]:
    rx = re.compile(pattern)
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            p = os.path.join(r, f)
            if rx.match(p):
                out.append(p)
    out.sort()
    return out

def robust_percentile_normalize(vol: np.ndarray, pmin: float, pmax: float) -> Tuple[np.ndarray, float, float]:
    """Normaliza volume para 0..1 por percentis (consistente por volume)."""
    vol = np.asarray(vol, dtype=np.float32)
    vmin = float(np.nanpercentile(vol, pmin))
    vmax = float(np.nanpercentile(vol, pmax))
    if vmax <= vmin + 1e-8:
        vmax = vmin + 1e-8
    out = (vol - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False), vmin, vmax

def make_crop_spaces(h: int, w: int, crop: int, step: int, thresh_size: int):
    """
    Replica a lógica do seu extract_subimages.worker (ordem h->w, com borda).
    """
    h_space = np.arange(0, max(1, h - crop + 1), step)
    if h - (h_space[-1] + crop) > thresh_size:
        h_space = np.append(h_space, h - crop)
    w_space = np.arange(0, max(1, w - crop + 1), step)
    if w - (w_space[-1] + crop) > thresh_size:
        w_space = np.append(w_space, w - crop)
    return h_space.astype(int), w_space.astype(int)

def normalize_key(full_name: str) -> str:
    """Remove __HR / __LRxN, mantém __sXXX (igual ao seu)."""
    return re.sub(r'__(HR|LRx\d+)_(s\d+)$', r'__\2', full_name)

# --- filtro escuro: usa seu filters.is_predominantly_dark se existir ---
try:
    from filters import is_predominantly_dark as _dark_filter
except Exception:
    _dark_filter = None

def patch_is_dark(
    patch_01: np.ndarray,
    thr01: float = 20.0/255.0,   # ~0.0784  (equivalente ao threshold=20 do seu filtro)
    percentage: float = 0.7,     # 70% dos pixels abaixo do limiar => patch "escuro"
    ignore_nan: bool = True,
    border: int = 0              # opcional: descarta uma borda antes de medir
) -> bool:
    """
    Determina se o patch (em escala 0..1) é 'predominantemente escuro'
    pela fração de pixels < thr01 ser maior que `percentage`.

    - patch_01: ndarray 2D (float, 0..1)
    - thr01: limiar na escala 0..1 (20/255 ≈ 0.0784 reproduz seu filtro original)
    - percentage: fração mínima de pixels abaixo do limiar para descartar
    - ignore_nan: ignora NaNs no cômputo
    - border: pixels da borda a descartar (0 = usa tudo)
    """
    x = np.asarray(patch_01, dtype=np.float32)
    if x.ndim != 2:
        # Se vier com canal, projeta pra 2D
        x = x[..., 0] if x.ndim == 3 else x.reshape(x.shape[0], x.shape[1])

    # Recorta borda se solicitado
    if border > 0:
        x = x[border:-border, border:-border]
        if x.size == 0:
            return True  # se ficou vazio, trate como escuro

    # Clipa para 0..1 por segurança
    x = np.clip(x, 0.0, 1.0)

    # Máscara de válidos
    if ignore_nan:
        mask = np.isfinite(x)
        total = mask.sum()
        if total == 0:
            return True
        frac_dark = np.mean((x[mask] < thr01))
    else:
        frac_dark = np.mean(x < thr01)

    return bool(frac_dark > float(percentage))

def split_dataset_by_volume(basenames: List[str], division=(0.95, 0.03, 0.02)):
    rng = np.random.default_rng(12345)
    uniq = sorted(set(basenames))
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(n * division[0])
    n_va = int(n * division[1])
    train = uniq[:n_tr]
    val   = uniq[n_tr:n_tr+n_va]
    test  = uniq[n_tr+n_va:]
    return train, val, test
