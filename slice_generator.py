import os
import re
import csv
import nibabel as nib
import numpy as np
from datetime import datetime
from skimage.io import imsave
import argparse
from extract_subimages import extract_subimages
from lmdb_util import make_lmdb_from_imgs
import cv2
from utils_scripts import scandir, split_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import re

# --- Configuracoes gerais ---
ORIENTATIONS = {
    "axial": lambda img: img,
    "coronal": lambda img: np.transpose(img, (1, 0, 2)),
    "sagittal": lambda img: np.transpose(img, (2, 0, 1)),
}

SLICE_LOG_FILE = "slice_log.csv"
IMAGE_EXT = ".png"

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)

def log_slice(path, force):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exists = os.path.exists(SLICE_LOG_FILE)
    with open(SLICE_LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not exists:
            writer.writerow(["timestamp", "file", "force"])
        writer.writerow([now, path, force])

def generate_and_crop_slices(img_data, name_prefix, output_dir, scales, selected_orientations, force):
    for orientation_name in selected_orientations:
        oriented = ORIENTATIONS[orientation_name](img_data)
        for i in range(oriented.shape[2]):
            slice_data = oriented[:, :, i]
            h, w = slice_data.shape

            base_name = f"{name_prefix}__{orientation_name}__slice_{i:03d}"

            hr_path = os.path.join(output_dir, "HR", base_name + "__HR" + IMAGE_EXT)
            if not os.path.exists(hr_path) or force:
                save_image(slice_data, hr_path)
                log_slice(hr_path, force)

            for scale in scales:
                low_res_size = (w // scale, h // scale)
                lr_dir = os.path.join(output_dir, f"LRx{scale}")
                lr_path = os.path.join(lr_dir, base_name + f"__LRx{scale}" + IMAGE_EXT)
                if not os.path.exists(lr_path) or force:
                    lr_img = cv2.resize(slice_data, low_res_size, interpolation=cv2.INTER_AREA)
                    save_image(lr_img, lr_path)
                    log_slice(lr_path, force)

def process_file(nii_path, output_dir, scales, selected_orientations, force, normalize, to_uint8):
    img = nib.load(nii_path)
    data = img.get_fdata()

    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0

    data = data.astype(np.uint8)

    name_prefix = os.path.splitext(os.path.basename(nii_path))[0].replace('.nii.gz', '').replace('.nii', '')
    generate_and_crop_slices(data, name_prefix, output_dir, scales, selected_orientations, force)

def match_filename(filename, pattern):
    return re.match(pattern, filename)

def extract_subimages_call(output_dir, scales):
    approved_keys_file = os.path.join(output_dir, 'HR', 'approved_keys.txt')

    extract_subimages({
        'input_folder': os.path.join(output_dir, 'HR'),
        'save_folder': os.path.join(output_dir, 'HR', 'sub'),
        'crop_size': 480,
        'step': 240,
        'thresh_size': 0,
        'n_thread': 8,
        'compression_level': 3,
        'apply_dark_filter': True,
        'approved_keys_file': approved_keys_file
    })

    with open(approved_keys_file, 'r') as f:
        approved_keys = set(line.strip() for line in f)

    for scale in scales:
        extract_subimages({
            'input_folder': os.path.join(output_dir, f'LRx{scale}'),
            'save_folder': os.path.join(output_dir, f'LRx{scale}', 'sub'),
            'crop_size': 480 // scale,
            'step': 240 // scale,
            'thresh_size': 0,
            'n_thread': 8,
            'compression_level': 3,
            'approved_keys': approved_keys
        })

def prepare_keys(img_path_list_or_folder):
    """Gera lista de caminhos e chaves sem __HR ou __LRxN, mantendo __sXXX no final."""
    if isinstance(img_path_list_or_folder, str):
        img_path_list = sorted(list(scandir(img_path_list_or_folder, suffix='png', recursive=False)))
    else:
        img_path_list = sorted(img_path_list_or_folder)

    def normalize_key(path):
        name = os.path.basename(path).split('.png')[0]
        return re.sub(r'__(HR|LRx\d+)_(s\d+)$', r'__\2', name)

    keys = [normalize_key(p) for p in img_path_list]
    return img_path_list, keys


def create_lmdb(output_dir, scales, division=(0.7, 0.2, 0.1)):
    """Cria LMDBs para HR e múltiplos LR, com splits consistentes e verificação prévia."""

    def strip_resolution_tag(filename):
        return '__'.join(filename.split('__')[:-1])  # Remove __HR_sXXX ou __LRxN_sXXX

    def split_dataset_by_tag(img_paths):
        base_names = [strip_resolution_tag(f) for f in img_paths]
        return split_dataset(base_names, division)

    def filter_by_base(img_list, base_list):
        return [img for img in img_list if strip_resolution_tag(img) in base_list]

    # === Carrega HR ===
    hr_folder = os.path.join(output_dir, 'HR', 'sub')
    hr_img_list, hr_keys = prepare_keys(hr_folder)
    train_base, val_base, test_base = split_dataset_by_tag(hr_img_list)

    # === Verificação antecipada de consistência LR x HR ===
    for scale in scales:
        lr_folder = os.path.join(output_dir, f'LRx{scale}', 'sub')
        lr_img_list, lr_keys = prepare_keys(lr_folder)

        for split_name, base_subset in zip(['train', 'val', 'test'], [train_base, val_base, test_base]):
            hr_filtered = filter_by_base(hr_img_list, base_subset)
            lr_filtered = filter_by_base(lr_img_list, base_subset)

            _, hr_split_keys = prepare_keys(hr_filtered)
            _, lr_split_keys = prepare_keys(lr_filtered)

            if len(hr_split_keys) != len(lr_split_keys):
                raise ValueError(
                    f"[Erro: {split_name} | LRx{scale}] Qtd diferente: HR={len(hr_split_keys)} vs LR={len(lr_split_keys)}"
                )
            if hr_split_keys != lr_split_keys:
                raise ValueError(
                    f"[Erro: {split_name} | LRx{scale}] Chaves diferentes entre HR e LR"
                )

    # === Criação dos LMDBs (após validação completa) ===
    hr_keys_splits = {}
    hr_imgs_splits = {}
    for split_name, base_subset in zip(['train', 'val', 'test'], [train_base, val_base, test_base]):
        split_imgs = filter_by_base(hr_img_list, base_subset)
        _, split_keys = prepare_keys(split_imgs)
        hr_imgs_splits[split_name] = split_imgs
        hr_keys_splits[split_name] = split_keys

        lmdb_path = os.path.join(output_dir, 'HR', 'lmdb', f'{split_name}.lmdb')
        make_lmdb_from_imgs(hr_folder, lmdb_path, split_imgs, split_keys)

    for scale in scales:
        lr_folder = os.path.join(output_dir, f'LRx{scale}', 'sub')
        lr_img_list, _ = prepare_keys(lr_folder)

        for split_name in ['train', 'val', 'test']:
            base_subset = train_base if split_name == 'train' else val_base if split_name == 'val' else test_base

            split_imgs_lr = filter_by_base(lr_img_list, base_subset)
            _, split_keys_lr = prepare_keys(split_imgs_lr)

            lmdb_path = os.path.join(output_dir, f'LRx{scale}', 'lmdb', f'{split_name}.lmdb')
            make_lmdb_from_imgs(lr_folder, lmdb_path, split_imgs_lr, split_keys_lr)

def main():
    parser = argparse.ArgumentParser(description="Slice NIfTI volumes into HR/LR subimages.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default=r".*/anat/.*nii.gz")
    # parser.add_argument("--pattern", default=r".*/anat/.*nii.gz")
    parser.add_argument("--scales", nargs="*", type=int, default=[2, 4])
    # parser.add_argument("--save_full_image", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--normalize", default=True, help="Normalize images to 0-255")
    parser.add_argument("--to_uint8", default=False, help="Convert images to uint8 format")
    parser.add_argument("--axes", nargs="*", default=['axial', 'coronal', 'sagittal'],
                        choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--only_lmdb", default=False)
    args = parser.parse_args()

    if not args.only_lmdb:
        nii_paths = []
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                nii_path = os.path.join(root, f)
                if match_filename(nii_path, args.pattern):
                    nii_paths.append(nii_path)

        if not nii_paths:
            print("⚠️ Nenhum arquivo .nii encontrado.")
            return

        max_workers = min(28, multiprocessing.cpu_count())
        print(f"🔧 Processando {len(nii_paths)} arquivos usando {max_workers} núcleos...\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_file,
                    nii_path,
                    args.output_dir,
                    args.scales,
                    args.axes,
                    args.force,
                    args.normalize,
                    args.to_uint8
                )
                for nii_path in nii_paths
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processando arquivos"):
                pass

        print("\n✅ Todos os arquivos foram processados.")

        # Agora, processa subimagens e LMDB
        extract_subimages_call(args.output_dir, args.scales)
    create_lmdb(args.output_dir, args.scales)

    print("\n✅ Subimagens e LMDB criados com sucesso.")

if __name__ == "__main__":
    main()
