# -*- coding: utf-8 -*-
# nii_to_lmdb.py — split GLOBAL por SLICE + logs detalhados
import os
import argparse
import logging
from typing import List, Dict, Tuple
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import nibabel as nib
from tqdm import tqdm
import traceback

from degradation_function_v2 import lower_field_degradation
from lmdb_npy import LmdbMakerNpy, np_to_npy_bytes
from patch_utils import (
    ORIENTATIONS, scandir_paths, robust_percentile_normalize,
    make_crop_spaces, normalize_key, patch_is_dark
)

# -----------------------------
# Logging
# -----------------------------
def setup_logger(output_dir: str, level=logging.INFO) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("nii2lmdb")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(ch)

    # Arquivo
    fh = logging.FileHandler(os.path.join(output_dir, "build.log"), mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger

# -----------------------------
# Split por SLICE (global)
# -----------------------------
def plan_slice_split(
    nii_paths: List[str],
    orientations: List[str],
    rng_seed: int,
    split_ratio=(0.95, 0.03, 0.02)
) -> Tuple[Dict[str, str], Dict[str, Counter], Dict[str, Tuple[int,int,int]]]:
    """
    Cria o mapeamento slice_id -> split ('train'/'val'/'test'), sem ler dados,
    apenas olhando shapes e orientações.

    Retorna:
      - split_by_slice: dict {slice_id: 'train'|'val'|'test'}
      - per_volume_counts: dict {base: Counter({'train':a,'val':b,'test':c})}
      - per_volume_slices_tot: dict {base: (n_axial, n_coronal, n_sagittal)}
    """
    rng = np.random.default_rng(rng_seed)
    slice_ids: List[str] = []
    per_volume_counts = defaultdict(Counter)
    per_volume_slices_tot: Dict[str, Tuple[int,int,int]] = {}

    for nii_path in nii_paths:
        base = os.path.splitext(os.path.basename(nii_path))[0].replace(".nii", "").replace(".gz", "")
        img = nib.load(nii_path)
        # shape padrão NIfTI tipicamente (X,Y,Z)
        sx, sy, sz = img.shape[:3]

        counts_by_orient = {}
        for orient in orientations:
            # replicar a mesma convenção do pipeline (fatiar ao longo do último eixo após ORIENTATIONS)
            if orient == "axial":
                n_slices = sz
            elif orient == "coronal":
                n_slices = sz  # transpose (1,0,2) -> eixo de slice continua sendo o 2
            elif orient == "sagittal":
                # transpose (2,0,1) -> shape (Z,X,Y); eixo 2 = Y
                n_slices = sy
            else:
                raise ValueError(f"Orientação inválida: {orient}")

            for i in range(n_slices):
                slice_id = f"{base}__{orient}__slice_{i:03d}"
                slice_ids.append(slice_id)

            counts_by_orient[orient] = n_slices

        per_volume_slices_tot[base] = (
            counts_by_orient.get("axial", 0),
            counts_by_orient.get("coronal", 0),
            counts_by_orient.get("sagittal", 0),
        )

    # Split global por slice_id
    idxs = np.arange(len(slice_ids))
    rng.shuffle(idxs)

    n = len(slice_ids)
    n_tr = int(n * split_ratio[0])
    n_va = int(n * split_ratio[1])
    n_te = n - n_tr - n_va
    tr_idx = idxs[:n_tr]
    va_idx = idxs[n_tr:n_tr+n_va]
    te_idx = idxs[n_tr+n_va:]

    split_by_slice = {}
    for i in tr_idx: split_by_slice[slice_ids[i]] = "train"
    for i in va_idx: split_by_slice[slice_ids[i]] = "val"
    for i in te_idx: split_by_slice[slice_ids[i]] = "test"

    # Contagem por volume
    for sid, split in split_by_slice.items():
        base = sid.split("__")[0]
        per_volume_counts[base][split] += 1

    return split_by_slice, per_volume_counts, per_volume_slices_tot

# -----------------------------
# Núcleo
# -----------------------------
def process_to_lmdb(
    nii_paths: List[str],
    output_dir: str,
    scales: List[int],
    orientations: List[str],
    crop_hr: int,
    step_hr: int,
    thresh_size: int,
    dark_thr01: float,           # ex.: 20/255
    dark_pct: float,             # ex.: 0.7
    dtype_out: str,
    preset_brain: str,
    normalize_percentiles,
    map_size_gb: float,
    split_ratio=(0.95, 0.03, 0.02),
    hr_from_input: bool = False,   # HR pela função (default)
    seed: int = 123
):
    logger = setup_logger(output_dir)

    # Ordena caminhos para estabilidade
    nii_paths = sorted(nii_paths)

    # Planeja SPLIT GLOBAL POR SLICE (sem carregar dados)
    split_by_slice, per_volume_counts, per_volume_slices_tot = plan_slice_split(
        nii_paths=nii_paths,
        orientations=orientations,
        rng_seed=seed,
        split_ratio=split_ratio
    )

    # Resumo do split por slice
    split_counter = Counter(split_by_slice.values())
    total_slices = sum(split_counter.values())
    logger.info("===== SPLIT POR SLICE (GLOBAL) =====")
    logger.info(f"Total de volumes: {len(nii_paths)}")
    logger.info(f"Total de slices:  {total_slices}")
    logger.info(f"Train slices: {split_counter.get('train',0)} | Val slices: {split_counter.get('val',0)} | Test slices: {split_counter.get('test',0)}")

    # Arquivos de split
    with open(os.path.join(output_dir, "split_info.txt"), "w") as sf:
        sf.write("=== SLICES por VOLUME (planejado) ===\n")
        for nii_path in nii_paths:
            base = os.path.splitext(os.path.basename(nii_path))[0].replace(".nii", "").replace(".gz", "")
            ax, co, sa = per_volume_slices_tot[base]
            c = per_volume_counts[base]
            sf.write(f"{base}: axial={ax}, coronal={co}, sagittal={sa} | train={c.get('train',0)}, val={c.get('val',0)}, test={c.get('test',0)}\n")

    with open(os.path.join(output_dir, "slice_split_info.txt"), "w") as sfile:
        sfile.write("=== TRAIN ===\n")
        for sid, sp in split_by_slice.items():
            if sp == "train": sfile.write(sid + "\n")
        sfile.write("\n=== VAL ===\n")
        for sid, sp in split_by_slice.items():
            if sp == "val": sfile.write(sid + "\n")
        sfile.write("\n=== TEST ===\n")
        for sid, sp in split_by_slice.items():
            if sp == "test": sfile.write(sid + "\n")

    # Abre writers por split
    writers: Dict[tuple, LmdbMakerNpy] = {}
    for split in ["train", "val", "test"]:
        writers[("HR", split)] = LmdbMakerNpy(os.path.join(output_dir, "HR", "lmdb", f"{split}.lmdb"),
                                              map_size_gb=map_size_gb, batch=5000)
        for s in scales:
            writers[(f"LRx{s}", split)] = LmdbMakerNpy(os.path.join(output_dir, f"LRx{s}", "lmdb", f"{split}.lmdb"),
                                                       map_size_gb=map_size_gb, batch=5000)

    # Contadores
    counts = {
        "slices_seen": Counter(),                     # quantos slices processados por split
        "hr_patches_written": Counter(),              # por split
        "hr_patches_discarded_dark": Counter(),       # por split
        "lr_patches_written": {s: Counter() for s in scales},  # por scale e split
    }

    logger.info("Iniciando processamento...")
    logger.info(f"Escalas: {scales} | Eixos: {orientations} | crop_hr={crop_hr} step_hr={step_hr} thr_size={thresh_size}")
    logger.info(f"dark_thr01={dark_thr01:.6f} dark_pct={dark_pct} | dtype_out={dtype_out} | preset={preset_brain}")
    logger.info(f"normalize_percentiles={normalize_percentiles} | map_size_gb={map_size_gb} | seed={seed}")
    logger.info("HR de origem: %s", "slice normalizado (hr_from_input=True)" if hr_from_input else "função de degradação (keep_size=True)")

    # Loop volumes
    for nii_path in tqdm(nii_paths, total=len(nii_paths), desc="Volumes", unit="vol"):
        try:
            base = os.path.splitext(os.path.basename(nii_path))[0].replace(".nii", "").replace(".gz", "")

            img = nib.load(nii_path)
            data = np.asanyarray(img.dataobj, dtype=np.float32)
            data, vmin, vmax = robust_percentile_normalize(data, *normalize_percentiles)
            logger.info(f"[VOL] {base} | shape={data.shape} | vmin={vmin:.4f} vmax={vmax:.4f}")

            for orient in orientations:
                vol = ORIENTATIONS[orient](data)  # (H, W, Z_após_map)
                H, W, Z = vol.shape
                logger.info(f"  Orientação={orient} | (H,W,Z)={vol.shape}")

                for i in range(Z):
                    sl_id = f"{base}__{orient}__slice_{i:03d}"
                    split = split_by_slice[sl_id]
                    counts["slices_seen"][split] += 1

                    sl = vol[:, :, i]  # 2D float32 [0,1]

                    # HR
                    if hr_from_input:
                        img_hr = sl.copy()
                    else:
                        res_hr = lower_field_degradation(sl, preset=preset_brain, seed=seed)
                        img_hr = np.asarray(res_hr["imagem_7t"], dtype=np.float32)

                    hH, wH = img_hr.shape
                    h_space, w_space = make_crop_spaces(hH, wH, crop=crop_hr, step=step_hr, thresh_size=thresh_size)

                    approved_keys = set()
                    index = 0
                    dark_discards = 0

                    for x in h_space:
                        for y in w_space:
                            index += 1
                            patch_hr = img_hr[x:x+crop_hr, y:y+crop_hr]
                            if patch_hr.shape != (crop_hr, crop_hr):
                                py = crop_hr - patch_hr.shape[0]
                                px = crop_hr - patch_hr.shape[1]
                                patch_hr = np.pad(patch_hr, ((0, py), (0, px)), mode='edge')

                            if patch_is_dark(patch_hr, thr01=dark_thr01, percentage=dark_pct):
                                dark_discards += 1
                                continue

                            full_key = f"{base}__{orient}__slice_{i:03d}__HR_s{index:03d}"
                            norm_key = normalize_key(full_key)
                            approved_keys.add(norm_key)

                            npy_bytes, dtype_str = np_to_npy_bytes(patch_hr, dtype_out=dtype_out)
                            writers[("HR", split)].put(norm_key, npy_bytes, (crop_hr, crop_hr, 1), dtype_str)
                            counts["hr_patches_written"][split] += 1

                    counts["hr_patches_discarded_dark"][split] += dark_discards

                    # LRs por escala (somente se HR aprovou)
                    for scale in scales:
                        res_lr = lower_field_degradation(sl, preset=preset_brain, seed=seed)
                        img_lr = np.asarray(res_lr["imagem_3t"], dtype=np.float32)

                        crop_lr = crop_hr // scale
                        index = 0
                        for x in h_space:
                            for y in w_space:
                                index += 1
                                norm_key = normalize_key(f"{base}__{orient}__slice_{i:03d}__LRx{scale}_s{index:03d}")
                                if norm_key not in approved_keys:
                                    continue

                                xx = int(x // scale)
                                yy = int(y // scale)
                                patch_lr = img_lr[xx:xx+crop_lr, yy:yy+crop_lr]
                                if patch_lr.shape != (crop_lr, crop_lr):
                                    py = crop_lr - patch_lr.shape[0]
                                    px = crop_lr - patch_lr.shape[1]
                                    patch_lr = np.pad(patch_lr, ((0, py), (0, px)), mode='edge')

                                npy_bytes, dtype_str = np_to_npy_bytes(patch_lr, dtype_out=dtype_out)
                                writers[(f"LRx{scale}", split)].put(norm_key, npy_bytes, (crop_lr, crop_lr, 1), dtype_str)
                                counts["lr_patches_written"][scale][split] += 1

                # progresso por orientação
                logger.info(f"  -> [{base} | {orient}] slices processados: {Z} | "
                            f"HR escritos (acum. {dict(counts['hr_patches_written'])}) | "
                            f"HR escuros (acum. {dict(counts['hr_patches_discarded_dark'])})")

        except Exception as e:
            logger.error(f"Falha ao processar volume '{nii_path}': {e}")
            logger.debug(traceback.format_exc())

    # Fecha writers
    for w in writers.values():
        w.close()

    # Meta + Sumário
    meta = dict(
        created=datetime.now().isoformat(),
        scales=scales,
        orientations=orientations,
        crop_hr=crop_hr, step_hr=step_hr,
        thresh_size=thresh_size,
        dark_thr01=dark_thr01, dark_pct=dark_pct,
        dtype_out=dtype_out,
        preset_brain=preset_brain,
        normalize_percentiles=normalize_percentiles,
        split_ratio=split_ratio,
        hr_from_input=hr_from_input,
        split_unit="slice"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dataset_meta.txt"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    # Relatório final nos logs
    logging.getLogger("nii2lmdb").info("===== RESUMO FINAL =====")
    logging.getLogger("nii2lmdb").info("Slices por split (processados): %s", dict(counts["slices_seen"]))
    logging.getLogger("nii2lmdb").info("HR patches salvos: %s", dict(counts["hr_patches_written"]))
    logging.getLogger("nii2lmdb").info("HR escuros descartados: %s", dict(counts["hr_patches_discarded_dark"]))
    for s in scales:
        logging.getLogger("nii2lmdb").info("LRx%d patches salvos: %s", s, dict(counts["lr_patches_written"][s]))

    print("\n✅ LMDBs (HR + LRs) criados com sucesso — split global por SLICE. Veja build.log, dataset_meta.txt e slice_split_info.txt.")

def main():
    ap = argparse.ArgumentParser("NIfTI → (degradação 7T→3T/1.5T) → patches → LMDB (.npy) — SPLIT POR SLICE")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pattern", default=r".*/anat/.*\.nii(\.gz)?$")
    ap.add_argument("--scales", nargs="*", type=int, default=[2])
    ap.add_argument("--axes", nargs="*", default=["axial", "coronal", "sagittal"],
                    choices=["axial", "coronal", "sagittal"])
    ap.add_argument("--crop_hr", type=int, default=128)
    ap.add_argument("--step_hr", type=int, default=64)
    ap.add_argument("--thresh_size", type=int, default=0)
    ap.add_argument("--dark_thr01", type=float, default=20.0/255.0, help="Limiar 0..1 p/ patch escuro (ex.: 20/255 ≈ 0.0784)")
    ap.add_argument("--dark_pct", type=float, default=0.7, help="Fraç. mínima de pixels < limiar p/ descartar o patch")
    ap.add_argument("--dtype_out", choices=["float32", "float16"], default="float32")
    ap.add_argument("--preset", default="3tFlash")
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.0)
    ap.add_argument("--map_size_gb", type=float, default=200.0)
    ap.add_argument("--hr_from_input", action="store_true",
                    help="Se setado: HR = slice normalizado do volume (sem passar na função).", default=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_ratio", type=float, default=0.95)
    ap.add_argument("--val_ratio", type=float, default=0.03)
    args = ap.parse_args()

    nii_paths = scandir_paths(args.input_dir, args.pattern)
    if not nii_paths:
        print("⚠️ Nenhum NIfTI encontrado.")
        return

    split_ratio = (args.train_ratio, args.val_ratio, max(0.0, 1.0 - args.train_ratio - args.val_ratio))

    print(f"🔧 {len(nii_paths)} volumes | escalas {args.scales} | eixos {args.axes}")
    process_to_lmdb(
        nii_paths=nii_paths,
        output_dir=args.output_dir,
        scales=args.scales,
        orientations=args.axes,
        crop_hr=args.crop_hr,
        step_hr=args.step_hr,
        thresh_size=args.thresh_size,
        dark_thr01=args.dark_thr01,
        dark_pct=args.dark_pct,
        dtype_out=args.dtype_out,
        preset_brain=args.preset,
        normalize_percentiles=(args.pmin, args.pmax),
        map_size_gb=args.map_size_gb,
        split_ratio=split_ratio,
        hr_from_input=args.hr_from_input,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
