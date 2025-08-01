import os
from multiprocessing import Pool
from tqdm import tqdm
from utils_scripts import scandir
import cv2
import numpy as np
from os import path as osp
from filters import is_predominantly_dark
import re

def extract_subimages(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']

    os.makedirs(save_folder, exist_ok=True)
    print(f'mkdir {save_folder} ...')

    img_list = list(scandir(input_folder, full_path=True))
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')

    pool = Pool(opt['n_thread'])
    approved_keys = set()

    def collect_keys(result):
        pbar.update(1)
        if result:
            approved_keys.update(result)

    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=collect_keys)

    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

    # Salvar chaves apenas se for HR
    if 'approved_keys_file' in opt:
        with open(opt['approved_keys_file'], 'w') as f:
            for k in sorted(approved_keys):
                f.write(f"{k}\n")

def normalize_key(full_name):
    # Remove __HR ou __LRxN e mantém __sXXX
    return re.sub(r'__(HR|LRx\d+)_(s\d+)$', r'__\2', full_name)

def worker(path, opt):
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    approved_keys = []

    for x in h_space:
        for y in w_space:
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            index += 1
            full_key = f'{img_name}_s{index:03d}'
            norm_key = normalize_key(full_key)

            if opt.get('apply_dark_filter') and is_predominantly_dark(cropped_img):
                continue

            if 'approved_keys' in opt and norm_key not in opt['approved_keys']:
                continue

            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{full_key}{extension}'),
                cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']]
            )

            if opt.get('apply_dark_filter'):
                approved_keys.append(norm_key)

    return approved_keys if opt.get('apply_dark_filter') else None