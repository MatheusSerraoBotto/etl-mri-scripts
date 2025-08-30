# -*- coding: utf-8 -*-
# Compatível com Python 3.8+

import os
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ===============================
# Utilidades de I/O e normalização
# ===============================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _normalize_uint8(x: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    """Normaliza para 0..255 por percentis (visualização, não para quantificação)."""
    x = np.asarray(x, dtype=np.float32)
    vmin = float(np.nanpercentile(x, pmin))
    vmax = float(np.nanpercentile(x, pmax))
    if vmax <= vmin + 1e-8:
        vmax = vmin + 1e-8
    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)

# ===============================
# Janelas / K-space
# ===============================

def _window2d(shape: Tuple[int, int], window_type: str = "hamming") -> Optional[np.ndarray]:
    """Janela 2D separável para suavizar bordas do k-space (reduz ringing)."""
    h, w = shape
    wt = (window_type or "none").lower()
    if wt == "none":
        return None
    if wt == "hamming":
        wy = np.hamming(h).astype(np.float32)
        wx = np.hamming(w).astype(np.float32)
    elif wt == "hann":
        wy = np.hanning(h).astype(np.float32)
        wx = np.hanning(w).astype(np.float32)
    else:
        raise ValueError("window_type deve ser 'hamming', 'hann' ou 'none'")
    return (wy[:, None] * wx[None, :]).astype(np.float32)

def _kspace_lowpass(
    img: np.ndarray,
    crop_factors: Tuple[float, float] = (0.7, 0.7),
    window_type: str = "hamming",
    keep_size: bool = False
) -> np.ndarray:
    """
    FFT2 -> janela -> crop central -> (opção) re-embed para tamanho original -> IFFT2 (magnitude).
    - keep_size=False: retorna imagem MENOR (downsample físico).
    - keep_size=True: mantém tamanho original (perde alta frequência sem mudar shape).
    """
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape
    K = fftshift(fft2(img, norm="ortho")).astype(np.complex64, copy=False)

    fy, fx = crop_factors
    kh = max(2, int(H * fy))
    kw = max(2, int(W * fx))

    cy, cx = H // 2, W // 2
    y0, y1 = cy - kh // 2, cy + (kh - kh // 2)
    x0, x1 = cx - kw // 2, cx + (kw - kw // 2)
    K_crop = K[y0:y1, x0:x1]

    win = _window2d((kh, kw), window_type=window_type)
    if win is not None:
        K_crop = (K_crop * win).astype(np.complex64, copy=False)

    if keep_size:
        # Re-embed no k-space original (zero-fill fora do bloco central)
        K_emb = np.zeros_like(K, dtype=np.complex64)
        K_emb[y0:y1, x0:x1] = K_crop
        img_c = ifft2(ifftshift(K_emb), norm="ortho")
        img_lp = np.abs(img_c).astype(np.float32, copy=False)
        return img_lp
    else:
        img_c = ifft2(ifftshift(K_crop), norm="ortho")
        img_small = np.abs(img_c).astype(np.float32, copy=False)
        return img_small

# ===============================
# Artefatos: Motion e Bias
# ===============================

def _fourier_shift_subpixel(img: np.ndarray, shift_yx: Tuple[float, float]) -> np.ndarray:
    """Desloca imagem por fase em Fourier (sub-pixel, sem reamostragem explícita)."""
    H, W = img.shape
    # Frequências normalizadas (ciclos/pixel)
    ky = fftshift(np.fft.fftfreq(H)).astype(np.float32)
    kx = fftshift(np.fft.fftfreq(W)).astype(np.float32)
    SY, SX = np.meshgrid(ky, kx, indexing="ij")
    dy, dx = shift_yx

    phase = np.exp(-2j * np.pi * (SY * dy + SX * dx)).astype(np.complex64)
    K = fftshift(fft2(img, norm="ortho")).astype(np.complex64, copy=False)
    K_shift = K * phase
    out = ifft2(ifftshift(K_shift), norm="ortho")
    return np.abs(out).astype(np.float32, copy=False)

def _apply_motion(
    img: np.ndarray,
    max_shift: float = 0.5,
    line_jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    - 'max_shift': deslocamento subpixel (rigid) aplicado uma vez (Fourier shift).
    - 'line_jitter': ruído de fase por linha de k-space (em radianos, desvio padrão).
      Simula movimento/instabilidade durante a aquisição (direção de phase-encode).
    """
    if rng is None:
        rng = np.random.default_rng()

    out = img.astype(np.float32, copy=False)

    # Rigid shift
    if max_shift and max_shift > 0:
        dy = float(rng.uniform(-max_shift, max_shift))
        dx = float(rng.uniform(-max_shift, max_shift))
        out = _fourier_shift_subpixel(out, (dy, dx))

    # Line jitter em k-space (fase aleatória por linha)
    if line_jitter and line_jitter > 0:
        H, W = out.shape
        K = fftshift(fft2(out, norm="ortho")).astype(np.complex64, copy=False)
        phases = rng.normal(0.0, line_jitter, size=(H, 1)).astype(np.float32)
        K = K * np.exp(1j * phases)
        out = ifft2(ifftshift(K), norm="ortho")
        out = np.abs(out).astype(np.float32, copy=False)

    return out

def _apply_bias_field(
    img: np.ndarray,
    strength: float = 0.1,
    scale_rel: float = 0.15,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Campo multiplicativo suave (B1+/recepção). 'scale_rel' é fração do menor lado usada para sigma do Gaussian.
    """
    if strength <= 0:
        return img.astype(np.float32, copy=False)
    if rng is None:
        rng = np.random.default_rng()
    H, W = img.shape
    sigma = max(4, int(min(H, W) * float(scale_rel)))
    noise = rng.normal(0, 1, (H, W)).astype(np.float32)
    field = gaussian_filter(noise, sigma=sigma).astype(np.float32)
    field -= field.min()
    den = field.max() - field.min()
    if den <= 0:
        den = 1.0
    field = field / den  # 0..1
    field = (1.0 - strength/2.0) + strength * field  # ~ [1-strength/2, 1+strength/2]
    return (img * field).astype(np.float32, copy=False)

# ===============================
# Multi-coil (sensibilidades) e ruído
# ===============================

def _synth_coil_sensitivities(
    shape: Tuple[int, int],
    Nc: int = 8,
    sigma_rel: float = 0.6
) -> np.ndarray:
    """
    Gera mapas de sensibilidade de Nc bobinas dispostas ao redor do FOV (magnitudes suaves).
    Retorna array (Nc, H, W) em float32, normalizado para que mean(sqrt(sum |c|^2)) ≈ 1.
    """
    H, W = shape
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    X, Y = np.meshgrid(x, y)  # (H, W)

    coils = []
    radius = 1.2  # bobinas ligeiramente fora do FOV
    for k in range(Nc):
        theta = 2 * np.pi * k / Nc
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)
        dist2 = (X - cx)**2 + (Y - cy)**2
        mag = np.exp(-dist2 / (2 * (sigma_rel**2))).astype(np.float32)
        coils.append(mag)
    C = np.stack(coils, axis=0)  # (Nc, H, W)

    norm = np.sqrt(np.sum(C**2, axis=0) + 1e-8)
    C = C / (np.mean(norm) + 1e-8)  # normaliza para média ~1 no mapa RSS
    return C.astype(np.float32, copy=False)

def _add_noise_and_combine(
    img_mag: np.ndarray,
    noise_model: str = "chi",
    Nc: int = 8,
    noise_sigma: Optional[float] = None,
    snr_target: Optional[float] = None,
    snr_7t_estimate: float = 40.0,
    B0_target_T: float = 3.0,
    alpha_snr: float = 1.0,
    account_voxel_size_factor: float = 1.0,
    coil_sens: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Adiciona ruído físico e combina:
      - 'rician' (single-coil)
      - 'chi' (multi-coil + RSS)
    Estratégia de sigma:
      - Se 'noise_sigma' dado: usa direto.
      - Senão, define SNR alvo: SNR_target = snr_target (se dado) OU
        snr_7t_estimate * (B0_target/7)^alpha * account_voxel_size_factor.
        Depois sigma baseado em média de sinal e modelo.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_model = (noise_model or "chi").lower()
    img_mag = np.asarray(img_mag, dtype=np.float32)
    m = float(np.mean(img_mag)) if img_mag.size else 0.0

    # Define SNR alvo se necessário
    if noise_sigma is None:
        if snr_target is None:
            snr_target = float(snr_7t_estimate) * (float(B0_target_T) / 7.0) ** float(alpha_snr)
            snr_target *= float(account_voxel_size_factor)
        snr_target = max(1.0, float(snr_target))

    if noise_model == "rician":
        # Single-coil: aproxima SNR ~ mean / sigma
        if noise_sigma is None:
            sigma = max(1e-6, m / snr_target)
        else:
            sigma = float(noise_sigma)
        n_r = rng.normal(0.0, sigma, img_mag.shape).astype(np.float32)
        n_i = rng.normal(0.0, sigma, img_mag.shape).astype(np.float32)
        out = np.sqrt((img_mag + n_r)**2 + n_i**2, dtype=np.float32)
        return out

    elif noise_model == "chi":
        # Multi-coil + RSS com sensibilidades
        Nc = max(1, int(Nc))
        if coil_sens is None:
            coil_sens = _synth_coil_sensitivities(img_mag.shape, Nc=Nc, sigma_rel=0.6)  # (Nc,H,W)
        else:
            assert coil_sens.shape[0] == Nc and coil_sens.shape[1:] == img_mag.shape

        S = img_mag[None, ...] * coil_sens  # (Nc,H,W), magnitude (fase ~0)
        # Aproximação: ruído efetivo RSS cresce ~ sqrt(Nc)
        if noise_sigma is None:
            k_eff = np.sqrt(Nc)
            sigma = max(1e-6, m / (snr_target * k_eff))
        else:
            sigma = float(noise_sigma)

        n_r = rng.normal(0.0, sigma, S.shape).astype(np.float32)
        n_i = rng.normal(0.0, sigma, S.shape).astype(np.float32)
        S_noisy = (S + n_r) ** 2 + (n_i ** 2)
        rss = np.sqrt(np.sum(S_noisy, axis=0), dtype=np.float32)  # (H,W)
        return rss

    else:
        raise ValueError("noise_model deve ser 'rician' ou 'chi'")

# ===============================
# Contraste de sequência (SPGR/FLASH, opcional)
# ===============================

def _spgr_signal(
    PD: np.ndarray,
    T1: np.ndarray,
    T2s: np.ndarray,
    TR: float,
    TE: float,
    FA_deg: float,
    B1_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Sinal SPGR (FLASH) simplificado (magnitude):
      S ∝ PD * sin(FA_eff) * (1 - E1) / (1 - E1*cos(FA_eff)) * exp(-TE/T2*)
      E1 = exp(-TR/T1)
    Exige mapas PD, T1, T2*. Se não tiver, não chame (use fallback).
    """
    PD = np.asarray(PD, dtype=np.float32)
    T1 = np.asarray(T1, dtype=np.float32)
    T2s = np.asarray(T2s, dtype=np.float32)
    assert PD.shape == T1.shape == T2s.shape
    FA = float(FA_deg) * np.pi / 180.0
    if B1_map is not None:
        FA_eff = (B1_map.astype(np.float32) * FA).astype(np.float32)
    else:
        FA_eff = FA
    E1 = np.exp(-float(TR) / (T1 + 1e-8)).astype(np.float32)
    num = (1.0 - E1)
    den = (1.0 - E1 * np.cos(FA_eff) + 1e-8)
    s = PD * np.sin(FA_eff) * (num / den) * np.exp(-float(TE) / (T2s + 1e-8))
    return s.astype(np.float32, copy=False)

def _scale_T_maps_for_field(
    T1_7T: np.ndarray,
    T2s_7T: np.ndarray,
    B0_target_T: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heurísticas globais para transformar T1/T2* de 7T para 3T ou 1.5T.
    Sem mapas reais por tecido, usamos escala global (ajustável conforme literatura).
    Defaults aproximados (ordem de grandeza):
      7T -> 3T:   T1 * 0.8,   T2* * 1.4
      7T -> 1.5T: T1 * 0.7,   T2* * 1.8
    """
    b = float(B0_target_T)
    if abs(b - 3.0) < 0.25:
        sT1, sT2s = 0.80, 1.40
    elif abs(b - 1.5) < 0.25:
        sT1, sT2s = 0.70, 1.80
    else:
        frac = (7.0 - b) / (7.0 - 1.5)
        sT1 = 0.70 + 0.10 * (1 - frac)  # 0.7..0.8
        sT2s = 1.80 - 0.40 * (1 - frac) # 1.8..1.4
    return (T1_7T * sT1).astype(np.float32), (T2s_7T * sT2s).astype(np.float32)

# ===============================
# Função principal (física + parâmetros)
# ===============================

def funcao_degradacao(
    imagem: np.ndarray,
    save: bool = True,
    fator_reducao: int = 4,
    # ---- parâmetros de "aquisição" ----
    keep_size: bool = False,                # True: mantém shape; False: reduz dimensão
    crop_factors: Optional[Tuple[float, float]] = None,  # se None e keep_size=False -> 1/fator
    window_type: str = "hamming",
    # ---- alvo de campo (B0) e SNR ----
    alvo_campo: str = "3T",                 # '3T' ou '1.5T' (ou '3'/'1.5')
    alpha_snr: float = 1.0,                 # SNR ∝ B0^alpha (≈1 na prática)
    snr_7t_estimate: float = 40.0,          # chute de SNR médio da 7T para calibração
    snr_target: Optional[float] = None,     # se quiser fixar SNR alvo manualmente
    account_voxel_size: bool = True,        # se reduz dimensão, SNR ↑ ~ fator_reducao (área)
    noise_model: str = "chi",               # 'chi' (multi-coil RSS) ou 'rician'
    noise_sigma: Optional[float] = None,    # se definido, ignora SNR alvo
    Nc: int = 8,                             # nº de bobinas (se 'chi')
    coil_sens: Optional[np.ndarray] = None, # (Nc,H,W) opcional
    # ---- artefatos ----
    motion_max_shift: float = 0.5,
    motion_line_jitter: float = 0.0,        # radianos (desvio padrão por linha)
    bias_strength: float = 0.10,
    bias_scale_rel: float = 0.15,
    # ---- contraste de sequência (opcional) ----
    usar_spgr: bool = False,                # True se quiser re-sintetizar contraste
    TR: float = 20e-3,                      # segundos (ex.: 20 ms)
    TE: float = 4e-3,                       # segundos
    FA_deg: float = 15.0,                   # graus
    PD_map: Optional[np.ndarray] = None,
    T1_map_7T: Optional[np.ndarray] = None,
    T2s_map_7T: Optional[np.ndarray] = None,
    B1_map: Optional[np.ndarray] = None,    # escala do FA efetivo
    # ---- salvamento ----
    out_dir: str = "saida_sim",
    save_format: str = "png",               # 'png' (8-bit) ou 'tiff' (float32)
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simula uma versão "3T" ou "1.5T" de um slice 7T, o mais fiel possível ao pipeline físico:
    k-space low-pass (resolução), artefatos (motion, bias), ruído físico (Rician/Chi), e (opcional) contraste SPGR.

    Retorna:
      dict {
        'imagem_7t': float32 (slice original, ou recontruído SPGR se usar_spgr=True),
        'imagem_3t': float32 (slice degradado simulando 3T/1.5T),
        'meta': { ... parâmetros usados ... }
      }
    """
    rng = np.random.default_rng(seed)
    slc = np.asarray(imagem, dtype=np.float32)
    if slc.ndim != 2:
        raise ValueError("Forneça um slice 2D (use data[:, :, idx]).")
    H, W = slc.shape
    if fator_reducao not in (2, 3, 4):
        raise ValueError("fator_reducao deve ser 2, 3 ou 4")

    # ----------- alvo de campo numérico -----------
    alvo = str(alvo_campo).lower().replace("t", "")
    try:
        B0_target_T = float(alvo.replace(",", "."))
    except Exception:
        B0_target_T = 3.0  # default 3T

    # ----------- contraste (opcional) -----------
    if usar_spgr and (PD_map is not None and T1_map_7T is not None and T2s_map_7T is not None):
        # redimensiona mapas se necessário (crop/pad simples ao centro)
        def _fit(arr):
            A = np.asarray(arr, dtype=np.float32)
            if A.shape == (H, W):
                return A
            Ay, Ax = A.shape
            sy0 = max(0, (Ay - H) // 2); sx0 = max(0, (Ax - W) // 2)
            A = A[sy0:sy0+H, sx0:sx0+W]
            py = max(0, H - A.shape[0]); px = max(0, W - A.shape[1])
            if py > 0 or px > 0:
                A = np.pad(A, ((py//2, py - py//2), (px//2, px - px//2)), mode='edge')
            return A.astype(np.float32, copy=False)

        PD = _fit(PD_map)
        T1_7 = _fit(T1_map_7T)
        T2s_7 = _fit(T2s_map_7T)
        B1 = _fit(B1_map) if B1_map is not None else None

        # escala de relaxometria para alvo (heurística global)
        T1_tgt, T2s_tgt = _scale_T_maps_for_field(T1_7, T2s_7, B0_target_T=B0_target_T)

        # re-sintetiza "imagem 7T" (normalizada) e "imagem target" (contraste físico)
        img_spgr_7t = _spgr_signal(PD, T1_7, T2s_7, TR=TR, TE=TE, FA_deg=FA_deg, B1_map=B1)
        img_spgr_tgt = _spgr_signal(PD, T1_tgt, T2s_tgt, TR=TR, TE=TE, FA_deg=FA_deg, B1_map=B1)

        # normaliza para mesma escala média do input (mantém nível global comparável)
        scale = (np.mean(slc) + 1e-8) / (np.mean(img_spgr_7t) + 1e-8)
        img_7t = (img_spgr_7t * scale).astype(np.float32, copy=False)
        base_for_acq = (img_spgr_tgt * scale).astype(np.float32, copy=False)
    else:
        # sem mapas: mantemos contraste do input no "alvo", focando em resolução + SNR
        img_7t = slc.copy()
        base_for_acq = slc.copy()

    # ----------- "Aquisição" (resolução via k-space) -----------
    if keep_size:
        cf = crop_factors if crop_factors is not None else (0.7, 0.7)
        img_res = _kspace_lowpass(base_for_acq, crop_factors=cf, window_type=window_type, keep_size=True)
        voxel_factor = 1.0
    else:
        # se crop_factors não dado, usa 1/fator_reducao (reduz dimensão física)
        cf = crop_factors if crop_factors is not None else (1.0 / float(fator_reducao), 1.0 / float(fator_reducao))
        img_res = _kspace_lowpass(base_for_acq, crop_factors=cf, window_type=window_type, keep_size=False)
        voxel_factor = float(fator_reducao) if account_voxel_size else 1.0  # SNR ↑ ~ fator_reducao

    # ----------- Artefatos: motion + bias -----------
    img_art = _apply_motion(img_res, max_shift=motion_max_shift, line_jitter=motion_line_jitter, rng=np.random.default_rng(seed))
    img_art = _apply_bias_field(img_art, strength=bias_strength, scale_rel=bias_scale_rel, rng=np.random.default_rng(seed+1 if seed is not None else None))

    # ----------- Ruído físico -----------
    img_noisy = _add_noise_and_combine(
        img_mag=img_art,
        noise_model=noise_model,
        Nc=Nc,
        noise_sigma=noise_sigma,
        snr_target=snr_target,
        snr_7t_estimate=snr_7t_estimate,
        B0_target_T=B0_target_T,
        alpha_snr=alpha_snr,
        account_voxel_size_factor=voxel_factor,
        coil_sens=coil_sens,
        rng=np.random.default_rng(seed+2 if seed is not None else None)
    )

    # ----------- Salvamento ----------- 
    # _ensure_dir(out_dir)
    # stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # alvo_tag = str(alvo_campo).replace(" ", "").replace(".", "_")
    # if save_format.lower() == "png":
    #     iio.imwrite(os.path.join(out_dir, f"degradada_{alvo_tag}_f{fator_reducao}_{stamp}.png"),
    #                 _normalize_uint8(img_noisy))
    #     if save:
    #         iio.imwrite(os.path.join(out_dir, f"original_7t_{stamp}.png"), _normalize_uint8(img_7t))
    # elif save_format.lower() == "tiff":
    #     # salva float32
    #     iio.imwrite(os.path.join(out_dir, f"degradada_{alvo_tag}_f{fator_reducao}_{stamp}.tiff"),
    #                 img_noisy.astype(np.float32, copy=False))
    #     if save:
    #         iio.imwrite(os.path.join(out_dir, f"original_7t_{stamp}.tiff"), img_7t.astype(np.float32, copy=False))
    # else:
    #     raise ValueError("save_format deve ser 'png' ou 'tiff'")

    # ----------- Retorno -----------
    meta = dict(
        alvo_campo=alvo_campo,
        B0_target_T=B0_target_T,
        keep_size=keep_size,
        crop_factors=cf,
        window_type=window_type,
        fator_reducao=fator_reducao,
        noise_model=noise_model,
        Nc=Nc,
        snr_7t_estimate=snr_7t_estimate,
        snr_target=snr_target,
        noise_sigma=noise_sigma,
        alpha_snr=alpha_snr,
        account_voxel_size=account_voxel_size,
        voxel_factor=voxel_factor,
        motion_max_shift=motion_max_shift,
        motion_line_jitter=motion_line_jitter,
        bias_strength=bias_strength,
        bias_scale_rel=bias_scale_rel,
        usar_spgr=usar_spgr,
        TR=TR, TE=TE, FA_deg=FA_deg,
        tem_mapas=bool(PD_map is not None and T1_map_7T is not None and T2s_map_7T is not None),
        save_format=save_format,
        out_dir=out_dir
    )
    return {"imagem_7t": img_7t.astype(np.float32, copy=False),
            "imagem_3t": img_noisy.astype(np.float32, copy=False),
            "meta": meta}

# ===============================
# PRESETS específicos para CÉREBRO
# ===============================

PRESETS_BRAIN: Dict[str, Dict[str, Any]] = {
    # Anatômico T1w (MPRAGE/SPGR-like) em 3T
    "3T_T1W": dict(
        fator_reducao=2,
        keep_size=False,
        crop_factors=None,         # usa 1/fator
        window_type="hamming",
        alvo_campo="3T",
        noise_model="chi",
        Nc=32,
        snr_target=20.0,
        bias_strength=0.08,
        bias_scale_rel=0.15,
        motion_max_shift=0.5,
        motion_line_jitter=0.02,
        usar_spgr=False
    ),
    # Anatômico T1w em 1.5T
    "15T_T1W": dict(
        fator_reducao=3,
        keep_size=False,
        crop_factors=None,
        window_type="hamming",
        alvo_campo="1.5T",
        noise_model="chi",
        Nc=24,
        snr_target=12.0,
        bias_strength=0.12,
        bias_scale_rel=0.18,
        motion_max_shift=0.6,
        motion_line_jitter=0.03,
        usar_spgr=False
    ),
    # T2* / GRE-like (3T) – mais sensível a susceptibilidade
    "3T_T2STAR": dict(
        fator_reducao=2,
        keep_size=False,
        crop_factors=None,
        window_type="hamming",
        alvo_campo="3T",
        noise_model="chi",
        Nc=32,
        snr_target=15.0,
        bias_strength=0.10,
        bias_scale_rel=0.15,
        motion_max_shift=0.7,
        motion_line_jitter=0.05,
        usar_spgr=False,  # se tiver mapas e quiser, ligue e ajuste TE↑
        TE=20e-3          # sugestão para destacar T2*
    ),
    # fMRI-like (EPI) – “look” clínico (sem distorções B0)
    "3T_fMRI": dict(
        fator_reducao=4,
        keep_size=False,
        crop_factors=None,
        window_type="hamming",
        alvo_campo="3T",
        noise_model="chi",
        Nc=32,
        snr_target=10.0,
        bias_strength=0.10,
        bias_scale_rel=0.18,
        motion_max_shift=0.8,
        motion_line_jitter=0.08,
        usar_spgr=False
    ),
}

def funcao_degradacao_brain(
    imagem: np.ndarray,
    preset: str = "3T_T1W",
    save: bool = True,
    out_dir: str = "saida_sim",
    save_format: str = "png",
    seed: Optional[int] = None,
    **overrides: Any
) -> Dict[str, Any]:
    """
    Wrapper conveniente para CÉREBRO: aplica um preset e permite override total.
    Ex.: funcao_degradacao_brain(slice, preset='3T_T1W', motion_max_shift=0.3)
    """
    if preset not in PRESETS_BRAIN:
        raise ValueError(f"Preset inválido: {preset}. Opções: {list(PRESETS_BRAIN.keys())}")
    params = dict(PRESETS_BRAIN[preset])
    params.update(overrides)  # usuário sobrepõe o preset

    return funcao_degradacao(
        imagem=imagem,
        save=save,
        out_dir=out_dir,
        save_format=save_format,
        seed=seed,
        **params
    )

# ===============================
# Como usar (exemplos)
# ===============================
if __name__ == "__main__":
    import nibabel as nib

    path = "ds006001/anat/sub-C1_acq_FLASH20_200um.nii.gz"
    img = nib.load(path)
    data = img.get_fdata()  # ou use dataobj para RAM baixa
    slice_idx = data.shape[2] // 2
    slice_2d = data[:, :, slice_idx]

    # 1) Preset 3T T1w (cérebro)
    res = funcao_degradacao_brain(slice_2d, preset="3T_T1W", save=True, seed=123)
    print("3T_T1W:", res["imagem_7t"].shape, "->", res["imagem_3t"].shape)

    # create image with both 7T and 3T images and labels
    import matplotlib.pyplot as plt
    # put images in vertical stacked
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    # first 3t
    axs[1].imshow(res["imagem_7t"], cmap='gray')
    axs[1].set_title("Image 7T")
    axs[1].axis('off')
    # then 3t
    axs[0].imshow(res["imagem_3t"], cmap='gray')
    axs[0].set_title("Image 3T")
    axs[0].axis('off')
    plt.tight_layout()
    plt.show()
    # # Uncomment to save the figure
    fig.savefig("output_3T_T1W.png")


    # # 2) Preset 1.5T T1w com override (manter shape)
    # res2 = funcao_degradacao_brain(slice_7t, preset="15T_T1W", keep_size=True, crop_factors=(0.33, 0.33), seed=123)
    # print("15T_T1W keep_size:", res2["imagem_7t"].shape, "->", res2["imagem_3t"].shape)

    # # 3) fMRI-like com overrides de motion
    # res3 = funcao_degradacao_brain(slice_7t, preset="3T_fMRI", motion_line_jitter=0.1, seed=123)
    # print("3T_fMRI:", res3["imagem_7t"].shape, "->", res3["imagem_3t"].shape)
