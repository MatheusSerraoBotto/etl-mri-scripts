# -*- coding: utf-8 -*-
# lmdb_npy.py
import os
import io
import lmdb
from os import path as osp
from typing import Tuple
import numpy as np

def np_to_npy_bytes(arr: np.ndarray, dtype_out: str = "float32") -> Tuple[bytes, str]:
    """Serializa ndarray como .npy (sem perdas). Retorna (bytes, dtype_str)."""
    if dtype_out == "float32":
        arr = arr.astype(np.float32, copy=False)
        dtype_str = "float32"
    elif dtype_out == "float16":
        arr = arr.astype(np.float16)
        dtype_str = "float16"
    else:
        raise ValueError("dtype_out deve ser 'float32' ou 'float16'")
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue(), dtype_str

class LmdbMakerNpy:
    """
    Writer de LMDB para arrays .npy (sem PNG).
    meta_info.txt: '{key}.npy (H,W,C) {dtype_str}'
    """
    def __init__(self, lmdb_path: str, map_size_gb: float = 80.0, batch: int = 5000):
        if not lmdb_path.endswith('.lmdb'):
            raise ValueError("lmdb_path must end with '.lmdb'.")
        if osp.exists(lmdb_path) and os.listdir(lmdb_path):
            raise FileExistsError(f"Folder {lmdb_path} already exists and is not empty.")
        os.makedirs(lmdb_path, exist_ok=True)
        map_size = int(map_size_gb * (1024**3))
        self.env = lmdb.open(lmdb_path, map_size=map_size, subdir=True, lock=True,
                             readahead=False, writemap=False)
        self.txn = self.env.begin(write=True)
        self.txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
        self.counter = 0
        self.batch = batch

    def put(self, key: str, npy_bytes: bytes, img_shape: Tuple[int, int, int], dtype_str: str):
        """key sem extensão; img_shape=(H,W,C)"""
        self.counter += 1
        self.txn.put(key.encode('ascii'), npy_bytes)
        h, w, c = img_shape
        self.txt_file.write(f'{key}.npy ({h},{w},{c}) {dtype_str}\n')
        if self.counter % self.batch == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        self.txn.commit()
        self.env.sync()
        self.env.close()
        self.txt_file.close()

def read_array_from_lmdb(lmdb_path: str, key: str) -> np.ndarray:
    """Le array .npy do LMDB e retorna np.ndarray (float32/float16)."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    env.close()
    if buf is None:
        raise KeyError(f"Key not found: {key}")
    return np.load(io.BytesIO(buf), allow_pickle=False)
