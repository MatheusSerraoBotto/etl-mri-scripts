# -*- coding: utf-8 -*-
# nii_to_lmdb_parallel.py — split GLOBAL por SLICE + logs + PROCESSAMENTO EM PARALELO
import os
import argparse
import logging
from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import Counter

import numpy as np
import nibabel as nib
from tqdm import tqdm
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from degradation_function import funcao_degradacao_brain
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
    logger = logging.getLogger("nii2lmdb-par")
    logger.setLevel(level)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, "build.log"), mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger

# -----------------------------
# Split por SLICE (global)
# -----------------------------
def plan_slice_split(nii_paths: List[str], orientations: List[str], rng_seed: int,
                     split_ratio=(0.95, 0.03, 0.02)) -> Tuple[Dict[str, str], Dict[str, Tuple[int,int,int]]]:
    rng = np.random.default_rng(rng_seed)
    slice_ids: List[Tuple[str, str, int]] = []  # (base, orient, i)
    per_volume_slices_tot: Dict[str, Tuple[int,int,int]] = {}

    for nii_path in nii_paths:
        base = os.path.splitext(os.path.basename(nii_path))[0].replace(".nii", "").replace(".gz", "")
        img = nib.load(nii_path)
        sx, sy, sz = img.shape[:3]  # (X,Y,Z)
        counts_by_orient = {}
        for orient in orientations:
            if orient == "axial":
                n_slices = sz
            elif orient == "coronal":
                n_slices = sz
            elif orient == "sagittal":
                n_slices = sy
            else:
                raise ValueError(f"Orientação inválida: {orient}")
            for i in range(n_slices):
                slice_ids.append((base, orient, i))
            counts_by_orient[orient] = n_slices
        per_volume_slices_tot[base] = (
            counts_by_orient.get("axial", 0),
            counts_by_orient.get("coronal", 0),
            counts_by_orient.get("sagittal", 0),
        )

    idxs = np.arange(len(slice_ids))
    rng.shuffle(idxs)
    n = len(slice_ids)
    n_tr = int(n * split_ratio[0])
    n_va = int(n * split_ratio[1])
    n_te = n - n_tr - n_va
    tr_idx = set(idxs[:n_tr].tolist())
    va_idx = set(idxs[n_tr:n_tr+n_va].tolist())
    te_idx = set(idxs[n_tr+n_va:].tolist())

    split_by_slice: Dict[str, str] = {}
    for j, trip in enumerate(slice_ids):
        base, orient, i = trip
        sid = f"{base}__{orient}__slice_{i:03d}"
        split_by_slice[sid] = "train" if j in tr_idx else ("val" if j in va_idx else "test")

    return split_by_slice, per_volume_slices_tot

# -----------------------------
# Utilidades
# -----------------------------
def get_slice_normalized(nii_path: str, orient: str, i: int, vmin: float, vmax: float) -> np.ndarray:
    """Lê apenas o slice necessário e normaliza para 0..1 com vmin/vmax do VOLUME."""
    img = nib.load(nii_path)
    dataobj = img.dataobj  # proxy
    if orient == "axial":
        sl = np.asanyarray(dataobj[:, :, i], dtype=np.float32)
    elif orient == "coronal":
        sl = np.asanyarray(dataobj[:, :, i], dtype=np.float32).T
    elif orient == "sagittal":
        sl = np.asanyarray(dataobj[i, :, :], dtype=np.float32).T
    else:
        raise ValueError("orient inválido")
    den = max(1e-8, (vmax - vmin))
    sl = (sl - vmin) / den
    sl = np.clip(sl, 0.0, 1.0).astype(np.float32, copy=False)
    return sl

def worker_slice_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executa o processamento de UM slice e retorna os bytes para LMDB.
    Retorna:
      {
        'split': 'train'|'val'|'test',
        'hr': [(key, npy_bytes, (H,W,1), dtype_str), ...],
        'lr': {scale: [(key, npy_bytes, (h,w,1), dtype_str), ...]},
        'stats': {'hr_written':int, 'hr_dark':int, 'lr_written':{scale:int}}
      }
    """
    try:
        (nii_path, base, orient, i, split, vmin, vmax, crop_hr, step_hr, thresh_size,
         dark_thr01, dark_pct, dtype_out, preset_brain, hr_from_input, seed, scales) = (
            job['nii_path'], job['base'], job['orient'], job['i'], job['split'], job['vmin'], job['vmax'],
            job['crop_hr'], job['step_hr'], job['thresh_size'],
            job['dark_thr01'], job['dark_pct'], job['dtype_out'], job['preset_brain'],
            job['hr_from_input'], job['seed'], job['scales']
        )

        # seed por slice (determinístico)
        slice_seed = (hash((base, orient, i)) ^ seed) & 0x7FFFFFFF

        # Slice normalizado
        sl = get_slice_normalized(nii_path, orient, i, vmin, vmax)

        # HR
        if hr_from_input:
            img_hr = sl
        else:
            res_hr = funcao_degradacao_brain(sl, preset=preset_brain, seed=slice_seed)
            img_hr = np.asarray(res_hr["imagem_7t"], dtype=np.float32)

        # espaços de recorte
        hH, wH = img_hr.shape
        h_space, w_space = make_crop_spaces(hH, wH, crop=crop_hr, step=step_hr, thresh_size=thresh_size)

        hr_out = []
        approved_keys = set()
        idx = 0
        hr_dark = 0
        hr_written = 0

        for x in h_space:
            for y in w_space:
                idx += 1
                patch_hr = img_hr[x:x+crop_hr, y:y+crop_hr]
                if patch_hr.shape != (crop_hr, crop_hr):
                    py = crop_hr - patch_hr.shape[0]
                    px = crop_hr - patch_hr.shape[1]
                    patch_hr = np.pad(patch_hr, ((0, py), (0, px)), mode='edge')

                if patch_is_dark(patch_hr, thr01=dark_thr01, percentage=dark_pct):
                    hr_dark += 1
                    continue

                full_key = f"{base}__{orient}__slice_{i:03d}__HR_s{idx:03d}"
                norm_key = normalize_key(full_key)
                approved_keys.add(norm_key)

                npy_bytes, dtype_str = np_to_npy_bytes(patch_hr, dtype_out=dtype_out)
                hr_out.append((norm_key, npy_bytes, (crop_hr, crop_hr, 1), dtype_str))
                hr_written += 1

        # LR (por escala) – segue approved_keys
        lr_out: Dict[int, List[Tuple[str, bytes, Tuple[int,int,1], str]]] = {s: [] for s in scales}
        lr_written = {s: 0 for s in scales}

        for scale in scales:
            res_lr = funcao_degradacao_brain(sl, preset=preset_brain, seed=slice_seed)
            img_lr = np.asarray(res_lr["imagem_3t"], dtype=np.float32)
            crop_lr = crop_hr // scale

            idx2 = 0
            for x in h_space:
                for y in w_space:
                    idx2 += 1
                    norm_key = normalize_key(f"{base}__{orient}__slice_{i:03d}__LRx{scale}_s{idx2:03d}")
                    if norm_key not in approved_keys:
                        continue
                    xx = int(x // scale); yy = int(y // scale)
                    patch_lr = img_lr[xx:xx+crop_lr, yy:yy+crop_lr]
                    if patch_lr.shape != (crop_lr, crop_lr):
                        py = crop_lr - patch_lr.shape[0]
                        px = crop_lr - patch_lr.shape[1]
                        patch_lr = np.pad(patch_lr, ((0, py), (0, px)), mode='edge')
                    npy_bytes, dtype_str = np_to_npy_bytes(patch_lr, dtype_out=dtype_out)
                    lr_out[scale].append((norm_key, npy_bytes, (crop_lr, crop_lr, 1), dtype_str))
                    lr_written[scale] += 1

        return {
            "ok": True,
            "split": split,
            "hr": hr_out,
            "lr": lr_out,
            "stats": {
                "hr_written": hr_written,
                "hr_dark": hr_dark,
                "lr_written": lr_written
            },
            "slice_id": f"{base}__{orient}__slice_{i:03d}"
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }

# -----------------------------
# Núcleo (paralelo)
# -----------------------------
def process_to_lmdb_parallel(
    nii_paths: List[str],
    output_dir: str,
    scales: List[int],
    orientations: List[str],
    crop_hr: int,
    step_hr: int,
    thresh_size: int,
    dark_thr01: float,
    dark_pct: float,
    dtype_out: str,
    preset_brain: str,
    normalize_percentiles,
    map_size_gb: float,
    split_ratio=(0.95, 0.03, 0.02),
    hr_from_input: bool = False,
    seed: int = 123,
    workers: int = None
):
    logger = setup_logger(output_dir)
    nii_paths = sorted(nii_paths)

    # Calcula vmin/vmax por VOLUME (uma vez) para manter a normalização consistente
    logger.info("Calculando percentis por volume para normalização...")
    vol_norm: Dict[str, Tuple[float,float]] = {}
    for nii in tqdm(nii_paths, desc="Percentis", unit="vol"):
        base = os.path.splitext(os.path.basename(nii))[0].replace(".nii", "").replace(".gz", "")
        img = nib.load(nii)
        vol = np.asanyarray(img.dataobj, dtype=np.float32)
        _, vmin, vmax = robust_percentile_normalize(vol, *normalize_percentiles)
        vol_norm[base] = (vmin, vmax)
        del vol

    # Planeja split global por slice
    split_by_slice, per_volume_slices_tot = plan_slice_split(
        nii_paths=nii_paths, orientations=orientations, rng_seed=seed, split_ratio=split_ratio
    )
    split_counter = Counter(split_by_slice.values())
    total_slices = sum(split_counter.values())
    logger.info("===== SPLIT POR SLICE (GLOBAL) =====")
    logger.info(f"Total de volumes: {len(nii_paths)} | Total de slices: {total_slices}")
    logger.info(f"Train: {split_counter.get('train',0)} | Val: {split_counter.get('val',0)} | Test: {split_counter.get('test',0)}")

    with open(os.path.join(output_dir, "split_info.txt"), "w") as sf:
        sf.write("=== SLICES por VOLUME (planejado) ===\n")
        for nii in nii_paths:
            base = os.path.splitext(os.path.basename(nii))[0].replace(".nii", "").replace(".gz", "")
            ax, co, sa = per_volume_slices_tot[base]
            # contagem por volume/split
            c_train = c_val = c_test = 0
            for orient in orientations:
                n_slices = ax if orient=="axial" else (co if orient=="coronal" else sa)
                for i in range(n_slices):
                    sid = f"{base}__{orient}__slice_{i:03d}"
                    sp = split_by_slice[sid]
                    if sp=="train": c_train+=1
                    elif sp=="val": c_val+=1
                    else: c_test+=1
            sf.write(f"{base}: axial={ax}, coronal={co}, sagittal={sa} | "
                     f"train={c_train}, val={c_val}, test={c_test}\n")

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

    # Abre writers (apenas no processo principal)
    writers: Dict[tuple, LmdbMakerNpy] = {}
    for split in ["train", "val", "test"]:
        writers[("HR", split)] = LmdbMakerNpy(os.path.join(output_dir, "HR", "lmdb", f"{split}.lmdb"),
                                              map_size_gb=map_size_gb, batch=5000)
        for s in scales:
            writers[(f"LRx{s}", split)] = LmdbMakerNpy(os.path.join(output_dir, f"LRx{s}", "lmdb", f"{split}.lmdb"),
                                                       map_size_gb=map_size_gb, batch=5000)

    # Monta lista de jobs
    jobs: List[Dict[str, Any]] = []
    for nii in nii_paths:
        base = os.path.splitext(os.path.basename(nii))[0].replace(".nii", "").replace(".gz", "")
        vmin, vmax = vol_norm[base]
        # precisamos das contagens de slices por orientação
        img = nib.load(nii)
        sx, sy, sz = img.shape[:3]
        for orient in orientations:
            if orient == "axial":
                n_slices = sz
            elif orient == "coronal":
                n_slices = sz
            else:  # sagittal
                n_slices = sy
            for i in range(n_slices):
                sid = f"{base}__{orient}__slice_{i:03d}"
                jobs.append({
                    "nii_path": nii, "base": base, "orient": orient, "i": i,
                    "split": split_by_slice[sid],
                    "vmin": vmin, "vmax": vmax,
                    "crop_hr": crop_hr, "step_hr": step_hr, "thresh_size": thresh_size,
                    "dark_thr01": dark_thr01, "dark_pct": dark_pct,
                    "dtype_out": dtype_out, "preset_brain": preset_brain,
                    "hr_from_input": hr_from_input, "seed": seed, "scales": scales
                })

    logger.info(f"Total de jobs (slices): {len(jobs)}")
    max_workers = workers or min( max(1, multiprocessing.cpu_count()-1), 32 )
    logger.info(f"Usando {max_workers} processos de worker")

    # Contadores
    counts = {
        "slices_seen": Counter(),
        "hr_patches_written": Counter(),
        "hr_patches_discarded_dark": Counter(),
        "lr_patches_written": {s: Counter() for s in scales},
        "failures": 0
    }

    # Executa em paralelo e escreve conforme recebe
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = [ex.submit(worker_slice_job, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Slices", unit="slice"):
            res = fut.result()
            if not res.get("ok", False):
                counts["failures"] += 1
                logging.getLogger("nii2lmdb-par").error("Worker falhou: %s\n%s", res.get("error"), res.get("trace"))
                continue

            split = res["split"]
            # HR
            for key, npy_bytes, shp, dtype_str in res["hr"]:
                writers[("HR", split)].put(key, npy_bytes, shp, dtype_str)
                counts["hr_patches_written"][split] += 1
            # LR
            for scale, items in res["lr"].items():
                for key, npy_bytes, shp, dtype_str in items:
                    writers[(f"LRx{scale}", split)].put(key, npy_bytes, shp, dtype_str)
                    counts["lr_patches_written"][scale][split] += 1

            counts["slices_seen"][split] += 1
            counts["hr_patches_discarded_dark"][split] += int(res["stats"]["hr_dark"])

    # Fecha writers
    for w in writers.values():
        w.close()

    # Meta + Sumário
    meta = dict(
        created=datetime.now().isoformat(),
        scales=scales,
        orientations=orientations,
        crop_hr=crop_hr, step_hr=step_hr, thresh_size=thresh_size,
        dark_thr01=dark_thr01, dark_pct=dark_pct,
        dtype_out=dtype_out, preset_brain=preset_brain,
        normalize_percentiles=normalize_percentiles,
        split_ratio=split_ratio, hr_from_input=hr_from_input,
        split_unit="slice", workers=max_workers
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dataset_meta.txt"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    lg = logging.getLogger("nii2lmdb-par")
    lg.info("===== RESUMO FINAL (PARALELO) =====")
    lg.info("Falhas de worker: %d", counts["failures"])
    lg.info("Slices por split: %s", dict(counts["slices_seen"]))
    lg.info("HR patches salvos: %s", dict(counts["hr_patches_written"]))
    lg.info("HR escuros descartados: %s", dict(counts["hr_patches_discarded_dark"]))
    for s in scales:
        lg.info("LRx%d patches salvos: %s", s, dict(counts["lr_patches_written"][s]))

    print("\n✅ LMDBs criados com sucesso (pipeline paralelo). Confira build.log e dataset_meta.txt.")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser("NIfTI → (degradação) → patches → LMDB (.npy) — PARALELO por SLICE")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pattern", default=r".*/anat/.*\.nii(\.gz)?$")
    ap.add_argument("--scales", nargs="*", type=int, default=[2])
    ap.add_argument("--axes", nargs="*", default=["axial", "coronal", "sagittal"],
                    choices=["axial", "coronal", "sagittal"])
    ap.add_argument("--crop_hr", type=int, default=480)
    ap.add_argument("--step_hr", type=int, default=240)
    ap.add_argument("--thresh_size", type=int, default=0)
    ap.add_argument("--dark_thr01", type=float, default=20.0/255.0)
    ap.add_argument("--dark_pct", type=float, default=0.7)
    ap.add_argument("--dtype_out", choices=["float32", "float16"], default="float32")
    ap.add_argument("--preset", default="3T_T1W")
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.0)
    ap.add_argument("--map_size_gb", type=float, default=80.0)
    ap.add_argument("--hr_from_input", action="store_true", default=True,
                    help="Se setado: HR = slice normalizado (não passa pela função).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_ratio", type=float, default=0.95)
    ap.add_argument("--val_ratio", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=None, help="Nº de processos (default: cpu_count-1, máx 32)")
    args = ap.parse_args()

    nii_paths = scandir_paths(args.input_dir, args.pattern)
    if not nii_paths:
        print("⚠️ Nenhum NIfTI encontrado.")
        return

    split_ratio = (args.train_ratio, args.val_ratio, max(0.0, 1.0 - args.train_ratio - args.val_ratio))

    print(f"🔧 {len(nii_paths)} volumes | escalas={args.scales} | eixos={args.axes} | workers={args.workers or (multiprocessing.cpu_count()-1)}")
    process_to_lmdb_parallel(
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
        seed=args.seed,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
