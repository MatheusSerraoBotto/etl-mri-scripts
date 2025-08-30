# -*- coding: utf-8 -*-
# snr_flash7t.py — SNR para volumes 7T (FLASH/Siemens 7T, 32 canais), Python 3.8+

import os
from typing import Optional, Union, Dict, Any, Tuple, List
import math

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

# ============================================================
# Utilidades gerais
# ============================================================

def _robust_minmax(x: np.ndarray, pmin=0.5, pmax=99.5) -> Tuple[float, float]:
    lo = float(np.nanpercentile(x, pmin))
    hi = float(np.nanpercentile(x, pmax))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi

def _otsu_threshold(x: np.ndarray, bins: int = 256, clip_percentiles=(0.5, 99.5)) -> float:
    """
    Otsu global com clipping por percentil para robustez.
    Retorna o limiar escalar.
    """
    x = np.asarray(x, dtype=np.float32)
    lo, hi = _robust_minmax(x, *clip_percentiles)
    x_clipped = np.clip(x, lo, hi)
    hist, edges = np.histogram(x_clipped, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    cdf = np.cumsum(p)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * bin_centers)
    mu_t = mu[-1]
    # variância inter-classes
    denom = (cdf * (1.0 - cdf) + 1e-12)
    sigma_b2 = (mu_t * cdf - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b2))
    return float(bin_centers[k])

def _make_masks(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera (brain_mask, noise_mask) 3D:
      - brain: Otsu + morfologia leve
      - noise: baixos valores fora do cérebro (ar)
    """
    vol = np.asarray(vol, dtype=np.float32)
    thr = _otsu_threshold(vol, bins=256, clip_percentiles=(0.5, 99.5))
    brain = vol > thr
    # morfologia: limpa buracos/artes
    st = generate_binary_structure(3, 1)  # 6-connected
    brain = binary_dilation(brain, structure=st, iterations=1)
    brain = binary_erosion(brain, structure=st, iterations=2)

    # máscara de ar: intensidades baixas e longe do cérebro
    brain_dil = binary_dilation(brain, structure=st, iterations=2)
    low = vol < np.nanpercentile(vol, 5.0)
    noise = np.logical_and(low, np.logical_not(brain_dil))

    # fallbacks
    if not np.any(brain):
        brain = vol > np.nanpercentile(vol, 70.0)
    if not np.any(noise):
        noise = vol < np.nanpercentile(vol, 5.0)
    return brain, noise

# ============================================================
# Núcleo: SNR de magnitude (Rician / Chi-RSS)
# ============================================================

def _sigma_from_background_std(std_air: float, mode: str, Nc: int) -> Tuple[float, int]:
    """
    Converte std(magnitude no fundo) -> sigma do ruído Gaussiano subjacente e k (GL da chi).
    - Rician (single-coil) ~ Rayleigh no fundo: std_rayleigh = sigma * sqrt((4 - pi)/2)
    - Chi (RSS multi-coil): Chi_k com k=2*Nc; var_chi = sigma^2 * (k - mu_chi^2/sigma^2)
      onde mu_chi/sigma = sqrt(2) * Γ((k+1)/2) / Γ(k/2)
    """
    mode = (mode or 'chi').lower()
    if mode == 'rician':
        k = 2
        sigma = float(std_air) / math.sqrt((4.0 - math.pi) / 2.0)  # ≈ / 0.655
        return sigma, k
    elif mode in ('chi', 'rss', 'multi'):
        k = 2 * int(max(1, Nc))
        # razão de gammas com math.gamma (suficiente p/ k moderado)
        mu_over_sigma = math.sqrt(2.0) * (math.gamma((k + 1) / 2.0) / math.gamma(k / 2.0))
        var_over_sigma2 = k - mu_over_sigma ** 2
        var_over_sigma2 = max(var_over_sigma2, 1e-6)
        sigma = float(std_air) / math.sqrt(var_over_sigma2)
        return sigma, k
    else:
        raise ValueError("mode deve ser 'rician' ou 'chi'/'rss'/'multi'")

def _snr_from_roi(signal_vals: np.ndarray, sigma: float, k: int) -> Dict[str, float]:
    """
    Estimativas de SNR:
    - naive: mean(signal)/std(signal)  [NEMA-like; enviesado para magnitude]
    - corrected (2º momento): SNR ≈ sqrt( E[M^2]/sigma^2 - k )
      (k=2 p/ Rician; k=2*Nc p/ Chi/RSS)
    """
    signal_vals = np.asarray(signal_vals, dtype=np.float32)
    m = float(np.mean(signal_vals))
    std_meas = float(np.std(signal_vals, ddof=1))
    m2 = float(np.mean(signal_vals ** 2))
    snr_corr_sq = max(m2 / (sigma ** 2 + 1e-12) - k, 0.0)
    snr_corr = float(math.sqrt(snr_corr_sq))
    return {
        "snr_naive_mean_over_std": m / (std_meas + 1e-12),
        "snr_corrected_m2": snr_corr,
        "mean_signal": m,
        "std_signal": std_meas,
        "mean_square_signal": m2
    }

def compute_snr_volume_7t(
    vol_or_path: Union[str, np.ndarray],
    brain_mask: Optional[np.ndarray] = None,
    noise_mask: Optional[np.ndarray] = None,
    mode: str = 'chi',           # 'rician' ou 'chi'/'rss'/'multi'
    Nc: int = 32,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    Calcula SNR para um volume 7T (magnitude).
      Retorna:
        - snr_corrected_m2 (recomendado)
        - snr_naive_mean_over_std (referência)
        - sigma_noise (Gauss subjacente), k_dof, contagens, etc.
    """
    if isinstance(vol_or_path, str):
        img = nib.load(vol_or_path)
        vol = np.asanyarray(img.dataobj, dtype=np.float32)
    else:
        vol = np.asarray(vol_or_path, dtype=np.float32)
    assert vol.ndim == 3, "Esperado volume 3D (H,W,Z)."

    if brain_mask is None or noise_mask is None:
        bm, nm = _make_masks(vol)
        if brain_mask is None:
            brain_mask = bm
        if noise_mask is None:
            noise_mask = nm

    brain_mask = brain_mask.astype(bool, copy=False)
    noise_mask = noise_mask.astype(bool, copy=False)

    signal_vals = vol[brain_mask]
    noise_vals = vol[noise_mask]

    std_air = float(np.std(noise_vals, ddof=1))
    mode_eff = 'chi' if mode.lower() in ('chi', 'rss', 'multi') else 'rician'
    sigma, k = _sigma_from_background_std(std_air, mode=mode_eff, Nc=Nc)

    snr_dict = _snr_from_roi(signal_vals, sigma=sigma, k=k)

    out = dict(
        snr_naive_mean_over_std=snr_dict["snr_naive_mean_over_std"],
        snr_corrected_m2=snr_dict["snr_corrected_m2"],
        sigma_noise=sigma,
        k_dof=k,
        mode_used=mode_eff,
        n_signal_voxels=int(signal_vals.size),
        n_noise_voxels=int(noise_vals.size),
        mean_signal=snr_dict["mean_signal"],
        std_signal=snr_dict["std_signal"],
        mean_square_signal=snr_dict["mean_square_signal"],
    )
    if return_details:
        lo, hi = _robust_minmax(vol, 0.5, 99.5)
        out.update(dict(
            robust_min=lo, robust_max=hi,
            brain_mask_coverage=float(signal_vals.size) / float(vol.size),
            noise_mask_coverage=float(noise_vals.size) / float(vol.size),
        ))
    return out

# ============================================================
# Fator de sequência (SPGR/FLASH) e wrapper 7T (Siemens 7T, 32ch)
# ============================================================

def _flash_spgr_factor(TR_s: float, TE_s: float, FA_deg: float,
                       T1_s: float, T2s_s: float) -> float:
    """
    Fator relativo de amplitude de sinal SPGR/FLASH (sem PD/ganho):
      S ∝ sin(FA) * (1 - E1) / (1 - E1*cos(FA)) * exp(-TE/T2*)
      E1 = exp(-TR/T1)
    """
    FA = math.radians(float(FA_deg))
    E1 = math.exp(-float(TR_s) / (float(T1_s) + 1e-8))
    num = (1.0 - E1)
    den = (1.0 - E1 * math.cos(FA) + 1e-8)
    return float(math.sin(FA) * (num / den) * math.exp(-float(TE_s) / (float(T2s_s) + 1e-8)))

def compute_snr_flash7t(
    vol_or_path: Union[str, np.ndarray],
    # defaults do seu FLASH 7T Siemens + 32-ch (doc que você forneceu)
    mode: str = 'chi',
    Nc: int = 32,
    TR_s: float = 40e-3,
    TE_s: float = 14.2e-3,
    FA_deg: float = 20.0,
    voxel_size_mm: Optional[Tuple[float, float, float]] = None,  # se ndarray
    target_vox_mm3: float = 1.0,  # normalização de SNR para 1 mm³
    # normalização opcional para remover efeito de sequência (pede T1/T2*)
    tissue_T1_s: Optional[float] = None,
    tissue_T2s_s: Optional[float] = None,
    # máscaras opcionais
    brain_mask: Optional[np.ndarray] = None,
    noise_mask: Optional[np.ndarray] = None,
    return_all: bool = True
) -> Dict[str, Any]:
    """
    Wrapper: SNR de magnitude em FLASH 7T com defaults do seu protocolo.
    - Retorna:
        snr_corrected_m2        (recomendado)
        snr_corrected_mm1       (normalizado para 1 mm³)
        snr_seq0, snr_seq0_mm1  (se T1/T2* fornecidos, remove efeito de TR/TE/FA)
    """
    # carregar volume e voxel size
    if isinstance(vol_or_path, str):
        img = nib.load(vol_or_path)
        vol = np.asanyarray(img.dataobj, dtype=np.float32)
        vz = img.header.get_zooms()[:3]
        vox = voxel_size_mm or (float(vz[0]), float(vz[1]), float(vz[2]))
    else:
        vol = np.asarray(vol_or_path, dtype=np.float32)
        if voxel_size_mm is None:
            raise ValueError("Para ndarray, informe voxel_size_mm=(dy,dx,dz) em mm.")
        vox = voxel_size_mm

    # SNR empírico corrigido
    snr_res = compute_snr_volume_7t(
        vol_or_path=vol,
        brain_mask=brain_mask,
        noise_mask=noise_mask,
        mode=mode,
        Nc=Nc,
        return_details=True
    )
    snr_corr = float(snr_res['snr_corrected_m2'])

    # Normalização por volume do voxel
    vox_vol = float(vox[0] * vox[1] * vox[2])  # mm^3
    snr_mm1 = snr_corr / max(vox_vol / float(target_vox_mm3), 1e-12)

    # Normalização opcional por sequência (remove TR/TE/FA)
    if (tissue_T1_s is not None) and (tissue_T2s_s is not None):
        f_seq = _flash_spgr_factor(TR_s, TE_s, FA_deg, tissue_T1_s, tissue_T2s_s)
        f_seq = max(float(f_seq), 1e-8)
        snr_seq0 = snr_corr / f_seq
        snr_seq0_mm1 = snr_mm1 / f_seq
    else:
        f_seq = None
        snr_seq0 = None
        snr_seq0_mm1 = None

    out = dict(
        snr_corrected_m2=snr_corr,
        snr_naive_mean_over_std=snr_res['snr_naive_mean_over_std'],
        snr_corrected_mm1=snr_mm1,
        voxel_size_mm=vox,
        voxel_volume_mm3=vox_vol,
        mode_used=snr_res['mode_used'],
        Nc=Nc,
        TR_s=TR_s, TE_s=TE_s, FA_deg=FA_deg,
        seq_factor=f_seq,
        snr_seq0=snr_seq0,
        snr_seq0_mm1=snr_seq0_mm1,
        sigma_noise=snr_res['sigma_noise'],
        k_dof=snr_res['k_dof'],
        n_signal_voxels=snr_res['n_signal_voxels'],
        n_noise_voxels=snr_res['n_noise_voxels'],
        mean_signal=snr_res['mean_signal'],
        std_signal=snr_res['std_signal'],
        mean_square_signal=snr_res['mean_square_signal'],
    )
    if return_all:
        # extras úteis
        out.update(dict(
            robust_min=snr_res.get('robust_min'),
            robust_max=snr_res.get('robust_max'),
            brain_mask_coverage=snr_res.get('brain_mask_coverage'),
            noise_mask_coverage=snr_res.get('noise_mask_coverage'),
        ))
    return out

# ============================================================
# Fase: utilitários simples (opcional)
# ============================================================

def sigma_phase_from_snr(snr_complex: float) -> float:
    """
    Predição de desvio-padrão da fase (rad) em alto SNR:
      sigma_phi ≈ 1 / SNR_complex
    Observação: SNR_complex ≈ SNR_magnitude quando SNR é alto.
    """
    return 1.0 / max(float(snr_complex), 1e-8)

def estimate_phase_sigma_from_repeats(
    phase_vols: List[Union[str, np.ndarray]],
    brain_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Estima sigma_phi (rad) por voxel via repetições (mesma geometria), e retorna:
      - sigma_phi_global (média sobre ROI)
      - sigma_phi_map (H,W,Z float32)
    As fases devem estar em radianos e desenroladas/compatíveis.
    """
    vols = []
    for v in phase_vols:
        if isinstance(v, str):
            img = nib.load(v)
            data = np.asanyarray(img.dataobj, dtype=np.float32)
        else:
            data = np.asarray(v, dtype=np.float32)
        assert data.ndim == 3
        vols.append(data)
    arr = np.stack(vols, axis=0)  # (R,H,W,Z)
    if brain_mask is None:
        # usa magnitude aproximada da fase? alternativa: ROI cheia e reporta média
        brain_mask = np.ones(arr.shape[1:], dtype=bool)
    else:
        brain_mask = brain_mask.astype(bool, copy=False)

    sigma_map = np.std(arr, axis=0, ddof=1).astype(np.float32)
    sigma_global = float(np.mean(sigma_map[brain_mask]))
    return {
        "sigma_phi_global": sigma_global,
        "sigma_phi_map": sigma_map,
        "n_repeats": arr.shape[0]
    }

# ============================================================
# Uso (exemplos)
# ============================================================

if __name__ == "__main__":
    """
    Exemplos de uso:
      1) SNR empírico corrigido em FLASH 7T (lendo NIfTI):
         - normalizado para 1 mm³
         - com/sem normalização de sequência
      2) SNR a partir de ndarray + voxel_size_mm
      3) (Opcional) Fase: estimar sigma_phi por repetições
    """

    # -------- 1) NIfTI PATH (recomendado) --------
    nifti_path = "ds006001/sub-C1/anat/sub-C1_acq_FLASH20_200um.nii.gz"

    # (A) sem remover TR/TE/FA (só SNR de magnitude corrigido + @1mm³)
    try:
        res = compute_snr_flash7t(
            nifti_path,
            mode='chi', Nc=32,              # RSS multi-coil coerente com 7T 32 ch
            TR_s=40e-3, TE_s=14.2e-3, FA_deg=20.0,
            target_vox_mm3=1.0,             # SNR normalizado para 1 mm³
            tissue_T1_s=None, tissue_T2s_s=None
        )
        print("=== FLASH 7T (magnitude) ===")
        print(f"SNR corrigido (empírico):      {res['snr_corrected_m2']:.2f}")
        print(f"SNR @ 1 mm³:                   {res['snr_corrected_mm1']:.2f}")
        print(f"Voxel size (mm):               {res['voxel_size_mm']}, vol={res['voxel_volume_mm3']:.4f} mm³")
        print(f"sigma_noise (Gauss subjacente):{res['sigma_noise']:.4g}, k={res['k_dof']} (mode={res['mode_used']})")
    except Exception as e:
        print("Exemplo 1(A) falhou (defina o caminho do NIfTI):", e)

    # (B) removendo efeito de sequência (requer T1/T2* estimados para o tecido de interesse)
    # Ajuste estes valores conforme seu caso (in vivo vs ex vivo, WM vs GM):
    try:
        res_seq = compute_snr_flash7t(
            nifti_path,
            mode='chi', Nc=32,
            TR_s=40e-3, TE_s=14.2e-3, FA_deg=20.0,
            target_vox_mm3=1.0,
            tissue_T1_s=2.0,     # exemplo (WM ~1.9–2.2 s em 7T in vivo; ex vivo difere)
            tissue_T2s_s=0.020   # exemplo (~20 ms em 7T; ex vivo é menor)
        )
        print("\n=== FLASH 7T (magnitude) com normalização de sequência ===")
        print(f"SNR (seq0):                    {res_seq['snr_seq0']:.2f}")
        print(f"SNR (seq0) @ 1 mm³:           {res_seq['snr_seq0_mm1']:.2f}")
        print(f"Fator de sequência (TR/TE/FA): {res_seq['seq_factor']:.4f}")
    except Exception as e:
        print("Exemplo 1(B) falhou (defina o caminho do NIfTI):", e)

    # -------- 2) ndarray + voxel_size_mm --------
    # Exemplo sintético (não esqueça de fornecer voxel_size_mm)
    try:
        vol_fake = np.random.RandomState(0).rand(128, 128, 64).astype(np.float32)
        res_nd = compute_snr_flash7t(
            vol_fake,
            mode='chi', Nc=32,
            TR_s=40e-3, TE_s=14.2e-3, FA_deg=20.0,
            voxel_size_mm=(0.2, 0.2, 0.2),  # 200 µm isotrópico
            target_vox_mm3=1.0
        )
        print("\n=== ndarray (exemplo sintético) ===")
        print(f"SNR corrigido:                 {res_nd['snr_corrected_m2']:.2f}")
        print(f"SNR @ 1 mm³:                   {res_nd['snr_corrected_mm1']:.2f}")
    except Exception as e:
        print("Exemplo 2 falhou:", e)

    # -------- 3) (Opcional) Fase por repetições --------
    # Se você tem volumes de fase repetidos na mesma geometria:
    # phase_paths = ["/path/phase_run1.nii.gz", "/path/phase_run2.nii.gz", ...]
    # phase_info = estimate_phase_sigma_from_repeats(phase_paths, brain_mask=None)
    # print("\n=== Fase (repetições) ===")
    # print(f"sigma_phi_global (rad): {phase_info['sigma_phi_global']:.4f}")
    # sigma_phi_map = phase_info['sigma_phi_map']  # (H,W,Z)
