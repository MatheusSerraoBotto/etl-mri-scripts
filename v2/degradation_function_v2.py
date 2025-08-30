# -*- coding: utf-8 -*-
# Compatível com Python 3.8+

import os
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ===============================
# Constantes físicas / cenário
# ===============================

# SNR corrigido (empírico) medido em 7T (magnitude) — fornecido
SNR_7T_EMPIRICAL = 153.03

# Parâmetros FLASH do dataset 7T reportado (TR/TE/FA)
FLASH_TR_s = 40e-3    # 40 ms
FLASH_TE_s = 14.2e-3  # 14.2 ms
FLASH_FA_deg = 20.0   # 20°

# ===============================
# Utilidades
# ===============================

def _prescan_normalize(img_rss: np.ndarray, coil_sens: np.ndarray,
                       beta: float = 1.0, blur_sigma_rel: float = 0.08) -> np.ndarray:
    """
    'Prescan normalize' aproximado: divide o RSS pelo envelope de recepção sqrt(sum|Ck|^2).
    - beta=1.0 => achatamento total; valores menores (ex: 0.7) fazem achatamento parcial.
    - blur_sigma_rel suaviza o envelope para evitar imprimir o padrão das bobinas.
    """
    H, W = img_rss.shape
    s_rss = np.sqrt(np.sum(coil_sens**2, axis=0)).astype(np.float32)
    if blur_sigma_rel and blur_sigma_rel > 0:
        sigma = max(1, int(min(H, W) * float(blur_sigma_rel)))
        s_rss = gaussian_filter(s_rss, sigma=sigma).astype(np.float32)
    s_rss /= (np.mean(s_rss) + 1e-8)
    return (img_rss / np.maximum(s_rss**beta, 1e-6)).astype(np.float32)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _window2d(shape: Tuple[int, int], window_type: str = "hamming") -> Optional[np.ndarray]:
    """Janela 2D separável para suavizar bordas no k-space (reduz ringing)."""
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

# ===============================
# Operações em k-space
# ===============================

def _kspace_lowpass(
    img: np.ndarray,
    crop_factors: Tuple[float, float],
    window_type: str = "hamming"
) -> np.ndarray:
    """
    FFT2 -> janela -> crop central -> (opção) re-embed -> IFFT2 (magnitude).
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

    # SEM manter tamanho: retorna imagem MENOR (downsample físico)
    img_c = ifft2(ifftshift(K_crop), norm="ortho")
    return np.abs(img_c).astype(np.float32, copy=False)

# ===============================
# Motion e Bias
# ===============================

def _fourier_shift_subpixel(img: np.ndarray, shift_yx: Tuple[float, float]) -> np.ndarray:
    """Desloca imagem por fase em Fourier (sub-pixel)."""
    H, W = img.shape
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
    line_jitter: float = 0.02,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Rigid shift sub-pixel + jitter de fase por linha de k-space."""
    if rng is None:
        rng = np.random.default_rng()
    out = img.astype(np.float32, copy=False)
    if max_shift and max_shift > 0:
        dy = float(rng.uniform(-max_shift, max_shift))
        dx = float(rng.uniform(-max_shift, max_shift))
        out = _fourier_shift_subpixel(out, (dy, dx))
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
    strength: float = 0.02,
    scale_rel: float = 0.15,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Campo multiplicativo suave (B1+/recepção)."""
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
    field = (1.0 - strength/2.0) + strength * field
    return (img * field).astype(np.float32, copy=False)

# ===============================
# Multi-coil / ruído (Chi/RSS)
# ===============================

def _synth_coil_sensitivities(
    shape: Tuple[int, int],
    Nc: int = 32,
    sigma_rel: float = 0.75
) -> np.ndarray:
    """Mapas de sensibilidade sintéticos para Nc bobinas ao redor do FOV."""
    H, W = shape
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    coils = []
    radius = 1.12
    for k in range(Nc):
        theta = 2 * np.pi * k / Nc
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)
        dist2 = (X - cx)**2 + (Y - cy)**2
        mag = np.exp(-dist2 / (2 * (sigma_rel**2))).astype(np.float32)
        coils.append(mag)
    C = np.stack(coils, axis=0)  # (Nc,H,W)
    norm = np.sqrt(np.sum(C**2, axis=0) + 1e-8)
    C = C / (np.mean(norm) + 1e-8)
    return C.astype(np.float32, copy=False)

def _snr_target_from_field(
    snr_7t: float,
    B0_target_T: float,
    alpha_snr: float,
    voxel_factor: float
) -> float:
    """
    SNR_target = SNR_7T * (B0_target/7)^alpha * voxel_factor
    """
    B0 = float(B0_target_T)
    alpha = float(alpha_snr)
    vf = float(max(1e-6, voxel_factor))
    return float(snr_7t) * (B0 / 7.0) ** alpha * vf

def _add_noise_and_combine(
    img_mag: np.ndarray,
    noise_model: str = "chi",
    Nc: int = 32,
    noise_sigma: Optional[float] = None,
    snr_target: Optional[float] = None,
    snr_7t_estimate: float = SNR_7T_EMPIRICAL,
    B0_target_T: float = 3.0,
    alpha_snr: float = 1.0,
    account_voxel_size_factor: float = 1.0,
    coil_sens: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    prescan_beta: float = 1.0,
    prescan_blur_rel: float = 0.08
) -> np.ndarray:
    """
    Adiciona ruído físico e combina:
      - 'rician' (single-coil, magnitude)
      - 'chi' (multi-coil + RSS)
    sigma é calculado a partir do SNR alvo (salvo se noise_sigma for fornecido).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_model = (noise_model or "chi").lower()
    img_mag = np.asarray(img_mag, dtype=np.float32)
    m = float(np.mean(img_mag)) if img_mag.size else 0.0

    if noise_sigma is None:
        if snr_target is None:
            snr_target = _snr_target_from_field(
                snr_7t=snr_7t_estimate,
                B0_target_T=B0_target_T,
                alpha_snr=alpha_snr,
                voxel_factor=account_voxel_size_factor
            )
        snr_target = max(1.0, float(snr_target))

    if noise_model == "rician":
        sigma = float(noise_sigma) if noise_sigma is not None else max(1e-8, m / snr_target)
        n_r = rng.normal(0.0, sigma, img_mag.shape).astype(np.float32)
        n_i = rng.normal(0.0, sigma, img_mag.shape).astype(np.float32)
        return np.sqrt((img_mag + n_r)**2 + n_i**2).astype(np.float32)

    elif noise_model == "chi":
        Nc = max(1, int(Nc))
        if coil_sens is None:
            coil_sens = _synth_coil_sensitivities(img_mag.shape, Nc=Nc, sigma_rel=0.75)
        else:
            assert coil_sens.shape[0] == Nc and coil_sens.shape[1:] == img_mag.shape
        S = img_mag[None, ...] * coil_sens  # (Nc,H,W)

        if noise_sigma is None:
            sigma = max(1e-8, m / (snr_target * np.sqrt(Nc)))
        else:
            sigma = float(noise_sigma)

        n_r = rng.normal(0.0, sigma, S.shape).astype(np.float32)
        n_i = rng.normal(0.0, sigma, S.shape).astype(np.float32)
        S_noisy = (S + n_r) ** 2 + (n_i ** 2)
        out = np.sqrt(np.sum(S_noisy, axis=0)).astype(np.float32)
        # Prescan normalize: achata o envelope de recepção (uniformiza periferia)
        if prescan_beta and prescan_beta > 0:
            out = _prescan_normalize(out, coil_sens, beta=float(prescan_beta),
                                     blur_sigma_rel=float(prescan_blur_rel))
        # Preserva o ganho DC (não deixa "estourar" globalmente)
        m = float(np.mean(img_mag)) if img_mag.size else 0.0
        
        # Reescala robusta em 2 passos (mediana e p98) + clamp conservador
        ref = img_mag
        thr = np.percentile(ref, 60.0)
        mask = ref > max(thr, 1e-6)

        if np.any(mask):
            # Passo 1: alinhar nível geral (mediana) sem “puxar” demais
            med_ref = np.median(ref[mask]); med_out = np.median(out[mask])
            s1 = (med_ref + 1e-8) / (med_out + 1e-8)
            out *= np.clip(s1, 0.85, 1.15)

            # Passo 2: alinhar highlights (p98) e evitar “estouro” na periferia
            p_ref = np.percentile(ref[mask], 98.0)
            p_out = np.percentile(out[mask], 98.0)
            s2 = (p_ref + 1e-8) / (p_out + 1e-8)
            out *= np.clip(s2, 0.85, 1.10)
        return out
    else:
        raise ValueError("noise_model deve ser 'rician' ou 'chi'")

# ===============================
# (Opcional) Contraste SPGR/FLASH
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
    """Sinal FLASH/SPGR simplificado (magnitude)."""
    PD = np.asarray(PD, dtype=np.float32)
    T1 = np.asarray(T1, dtype=np.float32)
    T2s = np.asarray(T2s, dtype=np.float32)
    assert PD.shape == T1.shape == T2s.shape
    FA = float(FA_deg) * np.pi / 180.0
    FA_eff = (B1_map.astype(np.float32) * FA).astype(np.float32) if B1_map is not None else FA
    E1 = np.exp(-float(TR) / (T1 + 1e-8)).astype(np.float32)
    num = (1.0 - E1)
    den = (1.0 - E1 * np.cos(FA_eff) + 1e-8)
    s = PD * np.sin(FA_eff) * (num / den) * np.exp(-float(TE) / (T2s + 1e-8))
    return s.astype(np.float32, copy=False)

def _scale_T_maps_for_field(T1_7T: np.ndarray, T2s_7T: np.ndarray, B0_target_T: float) -> Tuple[np.ndarray, np.ndarray]:
    """Heurística global para relaxometria 7T -> {3T,1.5T}."""
    b = float(B0_target_T)
    if abs(b - 3.0) < 0.25:
        sT1, sT2s = 0.80, 1.40
    elif abs(b - 1.5) < 0.25:
        sT1, sT2s = 0.70, 1.80
    else:
        frac = (7.0 - b) / (7.0 - 1.5)
        sT1 = 0.70 + 0.10 * (1 - frac)
        sT2s = 1.80 - 0.40 * (1 - frac)
    return (T1_7T * sT1).astype(np.float32), (T2s_7T * sT2s).astype(np.float32)

# ===============================
# Função principal (interface mantida; downsample físico sempre)
# ===============================

def degradation_function(
    imagem: np.ndarray,
    fator_reducao: int = 2,
    # ---- aquisição ----
    crop_factors: Optional[Tuple[float, float]] = None,  # Se None -> (1/f,1/f)
    window_type: str = "hamming",
    # ---- campo + SNR ----
    alvo_campo: str = "3T",
    alpha_snr: float = 1.0,
    snr_7t_estimate: float = SNR_7T_EMPIRICAL,
    snr_target: Optional[float] = None,
    account_voxel_size: bool = False,
    noise_model: str = "chi",
    noise_sigma: Optional[float] = None,
    Nc: int = 32,
    coil_sens: Optional[np.ndarray] = None,
    # ---- artefatos ----
    motion_max_shift: float = 0.5,
    motion_line_jitter: float = 0.02,
    bias_strength: float = 0.06,
    bias_scale_rel: float = 0.15,
    # ---- contraste (opcional) ----
    usar_spgr: bool = False,
    TR: float = FLASH_TR_s,
    TE: float = FLASH_TE_s,
    FA_deg: float = FLASH_FA_deg,
    PD_map: Optional[np.ndarray] = None,
    T1_map_7T: Optional[np.ndarray] = None,
    T2s_map_7T: Optional[np.ndarray] = None,
    B1_map: Optional[np.ndarray] = None,
    # ---- salvamento (mantido na assinatura; não usado aqui) ----
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simulação 7T -> {3T, 1.5T} (FLASH): k-space low-pass (downsample físico SEMPRE),
    motion, bias e ruído (Chi/Rician), com SNR alvo ∝ campo e voxel.
    """
    rng = np.random.default_rng(seed)
    slc = np.asarray(imagem, dtype=np.float32)
    if slc.ndim != 2:
        raise ValueError("Forneça um slice 2D (use data[:, :, idx]).")
    H, W = slc.shape
    if fator_reducao not in (2, 3, 4):
        raise ValueError("fator_reducao deve ser 2, 3 ou 4")

    # Campo alvo numérico
    alvo = str(alvo_campo).lower().replace("t", "")
    try:
        B0_target_T = float(alvo.replace(",", "."))
    except Exception:
        B0_target_T = 3.0

    # (Opcional) síntese de contraste se houver mapas
    if usar_spgr and (PD_map is not None and T1_map_7T is not None and T2s_map_7T is not None):
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
        PD = _fit(PD_map); T1_7 = _fit(T1_map_7T); T2s_7 = _fit(T2s_map_7T)
        B1 = _fit(B1_map) if B1_map is not None else None
        T1_tgt, T2s_tgt = _scale_T_maps_for_field(T1_7, T2s_7, B0_target_T=B0_target_T)
        img_spgr_7t = _spgr_signal(PD, T1_7, T2s_7, TR=TR, TE=TE, FA_deg=FA_deg, B1_map=B1)
        img_spgr_tgt = _spgr_signal(PD, T1_tgt, T2s_tgt, TR=TR, TE=TE, FA_deg=FA_deg, B1_map=B1)
        scale = (np.mean(slc) + 1e-8) / (np.mean(img_spgr_7t) + 1e-8)
        img_7t = (img_spgr_7t * scale).astype(np.float32, copy=False)
        base_for_acq = (img_spgr_tgt * scale).astype(np.float32, copy=False)
    else:
        img_7t = slc.copy()
        base_for_acq = slc.copy()

    # ===== Resolução (k-space) — downsample físico SEMPRE =====
    if crop_factors is None:
        cf = (1.0 / float(fator_reducao), 1.0 / float(fator_reducao))
        voxel_factor = float(fator_reducao) ** 2  # área ↑ f^2 (espessura constante)
    else:
        cf = (float(crop_factors[0]), float(crop_factors[1]))
        voxel_factor = 1.0 / max(1e-6, (cf[0] * cf[1]))  # área ↑ 1/(fy*fx)
    img_res = _kspace_lowpass(base_for_acq, crop_factors=cf, window_type=window_type)

    # Motion + bias
    img_art = _apply_motion(img_res, max_shift=motion_max_shift, line_jitter=motion_line_jitter,
                            rng=np.random.default_rng(seed))
    img_art = _apply_bias_field(img_art, strength=bias_strength, scale_rel=bias_scale_rel,
                                rng=np.random.default_rng(seed+1 if seed is not None else None))

    # Ruído físico (Chi por padrão, Nc=32)
    img_noisy = _add_noise_and_combine(
        img_mag=img_art,
        noise_model=noise_model,
        Nc=Nc,
        noise_sigma=noise_sigma,
        snr_target=snr_target,
        snr_7t_estimate=snr_7t_estimate,
        B0_target_T=B0_target_T,
        alpha_snr=alpha_snr,
        account_voxel_size_factor=(voxel_factor if account_voxel_size else 1.0),
        coil_sens=coil_sens,
        rng=np.random.default_rng(seed+2 if seed is not None else None),
        prescan_beta=0.8,
        prescan_blur_rel=0.12
    )

    meta = dict(
        alvo_campo=alvo_campo, B0_target_T=B0_target_T,
        crop_factors=cf, window_type=window_type,
        fator_reducao=fator_reducao,
        noise_model=noise_model, Nc=Nc,
        snr_7t_estimate=snr_7t_estimate, snr_target=snr_target, alpha_snr=alpha_snr,
        account_voxel_size=account_voxel_size, voxel_factor=voxel_factor,
        motion_max_shift=motion_max_shift, motion_line_jitter=motion_line_jitter,
        bias_strength=bias_strength, bias_scale_rel=bias_scale_rel,
        usar_spgr=usar_spgr, TR=TR, TE=TE, FA_deg=FA_deg,
        tem_mapas=bool(PD_map is not None and T1_map_7T is not None and T2s_map_7T is not None)
    )
    return {
        "imagem_7t": img_7t.astype(np.float32, copy=False),
        "imagem_3t": img_noisy.astype(np.float32, copy=False),
        "meta": meta
    }

# ===============================
# PRESETS (apenas dois)
# ===============================

PRESETS_BRAIN: Dict[str, Dict[str, Any]] = {
    # 7T FLASH -> 3T FLASH (clínico), 32 canais
    "3tFlash": dict(
        fator_reducao=2,
        crop_factors=None,            # usa 1/fator
        window_type="hamming",
        alvo_campo="3T",
        noise_model="chi",
        Nc=32,
        snr_target=None,              # será calculado a partir de 153.03 @7T e B0=3.0T
        alpha_snr=1.0,
        bias_strength=0.08,
        bias_scale_rel=0.15,
        motion_max_shift=0.5,
        motion_line_jitter=0.02,
        usar_spgr=False,
        TR=FLASH_TR_s, TE=FLASH_TE_s, FA_deg=FLASH_FA_deg
    ),

    # 7T FLASH -> 1.5T FLASH (clínico), 32 canais
    "1.5Flash": dict(
        fator_reducao=3,
        crop_factors=None,
        window_type="hamming",
        alvo_campo="1.5T",
        noise_model="chi",
        Nc=32,
        snr_target=None,              # calculado a partir de 153.03 @7T e B0=1.5T
        alpha_snr=1.0,
        bias_strength=0.12,           # mais homogêneo
        bias_scale_rel=0.18,
        motion_max_shift=0.6,
        motion_line_jitter=0.03,
        usar_spgr=False,
        TR=FLASH_TR_s, TE=FLASH_TE_s, FA_deg=FLASH_FA_deg
    ),
}

def lower_field_degradation(
    imagem: np.ndarray,
    preset: str = "3tFlash",
    seed: Optional[int] = None,
    **overrides: Any
) -> Dict[str, Any]:
    """
    Wrapper com presets: '3tFlash' e '1.5Flash'.
    (Presets antigos foram removidos.)
    """
    if preset not in PRESETS_BRAIN:
        raise ValueError(f"Preset inválido: {preset}. Opções: {list(PRESETS_BRAIN.keys())}")

    params = dict(PRESETS_BRAIN[preset])
    params.update(overrides)

    return degradation_function(
        imagem=imagem,
        seed=seed,
        **params
    )
