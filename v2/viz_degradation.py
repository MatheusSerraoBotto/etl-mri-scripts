# viz_degradacao.py
# Uso:
#   python viz_degradacao.py --nii /caminho/vol.nii.gz --axis axial --slice 200 --out out_dir
#   (axis em {axial, coronal, sagittal}; slice default = meio)

import os
import argparse
import numpy as np
import nibabel as nib
import imageio.v3 as iio

from degradation_function_v2 import lower_field_degradation  # seus presets: '3T_FLASH', '1p5T_FLASH'

ORIENT = {
    "axial":    lambda vol: vol,                  # (X,Y,Z)
    "coronal":  lambda vol: np.transpose(vol, (1, 0, 2)),  # (Y,X,Z)
    "sagittal": lambda vol: np.transpose(vol, (2, 0, 1)),  # (Z,X,Y)
}

def pnorm_uint8(x, pmin=1.0, pmax=99.0):
    x = np.asarray(x, dtype=np.float32)
    vmin = float(np.nanpercentile(x, pmin))
    vmax = float(np.nanpercentile(x, pmax))
    if vmax <= vmin + 1e-8:
        vmax = vmin + 1e-8
    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser("Visualiza a degradação (3T/1.5T) em um único slice")
    ap.add_argument("--nii", required=True, help="Caminho do NIfTI (.nii/.nii.gz)")
    ap.add_argument("--axis", default="axial", choices=["axial", "coronal", "sagittal"])
    ap.add_argument("--slice", type=int, default=None, help="Índice do slice (default: meio)")
    ap.add_argument("--out", required=True, help="Pasta de saída para PNGs")
    ap.add_argument("--snr7t", type=float, default=153.03, help="SNR corrigido 7T (âncora)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Carrega volume
    img = nib.load(args.nii)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    vol = ORIENT[args.axis](data)  # shape (H, W, Z) após reorientação
    H, W, Z = vol.shape

    idx = args.slice if args.slice is not None else (Z // 2)
    idx = max(0, min(Z - 1, idx))
    sl = vol[:, :, idx]

    # HR visual (normalizado, para referência)
    hr_vis = pnorm_uint8(sl)

    # Degrada 3T (downsample físico; preset já faz isso)
    res3 = lower_field_degradation(
        sl, preset="3tFlash", seed=args.seed,
    )
    print(res3["meta"])
    lr3 = res3["imagem_3t"]
    lr3_vis = pnorm_uint8(lr3)

    # Degrada 1.5T
    res15 = lower_field_degradation(
        sl, preset="1.5Flash", seed=args.seed
    )
    print(res15["meta"])
    lr15 = res15["imagem_3t"]
    lr15_vis = pnorm_uint8(lr15)

    # Salva imagens individuais
    base = os.path.splitext(os.path.basename(args.nii))[0].replace(".nii", "").replace(".gz", "")
    iio.imwrite(os.path.join(args.out, f"{base}__{args.axis}__s{idx:03d}__HR.png"), hr_vis)
    iio.imwrite(os.path.join(args.out, f"{base}__{args.axis}__s{idx:03d}__3T.png"), lr3_vis)
    iio.imwrite(os.path.join(args.out, f"{base}__{args.axis}__s{idx:03d}__1p5T.png"), lr15_vis)

    # Triptych: HR | 3T | 1.5T (ajusta alturas via padding se necessário)
    h_max = max(hr_vis.shape[0], lr3_vis.shape[0], lr15_vis.shape[0])
    def pad_to_h(img, h=h_max):
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        return np.pad(img, ((0, pad), (0, 0)), mode='edge')

    hr_pad = pad_to_h(hr_vis)
    lr3_pad = pad_to_h(lr3_vis)
    lr15_pad = pad_to_h(lr15_vis)
    trip = np.concatenate([hr_pad, lr3_pad, lr15_pad], axis=1)
    iio.imwrite(os.path.join(args.out, f"{base}__{args.axis}__s{idx:03d}__HR_3T_1p5T_triptych.png"), trip)

    print("OK: imagens salvas em", args.out)

if __name__ == "__main__":
    main()
