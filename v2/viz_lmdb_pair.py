# -*- coding: utf-8 -*-
# viz_lmdb_pair.py
import os
import io
import re
import lmdb
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def read_array_from_lmdb(lmdb_path: str, key: str) -> np.ndarray:
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048)
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    env.close()
    if buf is None:
        raise KeyError(f"Key não encontrada no LMDB: {key}")
    return np.load(io.BytesIO(buf), allow_pickle=False)

def list_keys(lmdb_path: str, limit=None, regex: str = None):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048)
    pat = re.compile(regex) if regex else None
    keys = []
    with env.begin(write=False) as txn:
        with txn.cursor() as cur:
            for k, _ in cur:
                k = k.decode('ascii')
                if (pat is None) or pat.search(k):
                    keys.append(k)
                if limit is not None and len(keys) >= limit:
                    break
    env.close()
    return keys

def resize_like(img: np.ndarray, target_shape, method: str = "bicubic") -> np.ndarray:
    th, tw = target_shape[:2]
    if img.shape[:2] == (th, tw):
        return img
    if _HAS_CV2:
        interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA
        }.get(method.lower(), cv2.INTER_CUBIC)
        return cv2.resize(img, (tw, th), interpolation=interp)
    # fallback sem OpenCV (lento)
    from math import floor
    ys = (np.linspace(0, img.shape[0]-1, th)).astype(np.float32)
    xs = (np.linspace(0, img.shape[1]-1, tw)).astype(np.float32)
    yi = np.clip(np.round(ys).astype(int), 0, img.shape[0]-1)
    xi = np.clip(np.round(xs).astype(int), 0, img.shape[1]-1)
    return img[yi][:, xi]

def to_uint8_same_window(hr: np.ndarray, lr: np.ndarray, pmin=1.0, pmax=99.0):
    hr = np.asarray(hr, dtype=np.float32)
    lr = np.asarray(lr, dtype=np.float32)
    vmin = float(np.nanpercentile(hr, pmin))
    vmax = float(np.nanpercentile(hr, pmax))
    if vmax <= vmin + 1e-8:
        vmax = vmin + 1e-8
    def norm(x):
        y = (x - vmin) / (vmax - vmin)
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)
    return norm(hr), norm(lr), (vmin, vmax)

def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[2] == 1:
        return x[:, :, 0]
    if x.ndim == 2:
        return x
    # se vier algo fora do padrão, pega o primeiro canal
    return x[..., 0]

def main():
    ap = argparse.ArgumentParser("Visualizador de pares HR/LR de LMDB (.npy)")
    ap.add_argument("--hr_lmdb", required=True, help="Caminho para HR/lmdb/{train,val,test}.lmdb")
    ap.add_argument("--lr_lmdb", required=True, help="Caminho para LRxS/lmdb/{train,val,test}.lmdb")
    ap.add_argument("--key", default=None, help="Chave exata (ex.: base__axial__slice_000__s001)")
    ap.add_argument("--filter_regex", default=None, help="Regex para filtrar chaves (usado se --key não for dado)")
    ap.add_argument("--random", action="store_true", help="Escolhe uma chave aleatória (com o filtro, se houver)")
    ap.add_argument("--method", default="bicubic", choices=["nearest","bilinear","bicubic","area"], help="Interpolação para redimensionar LR")
    ap.add_argument("--pmin", type=float, default=1.0, help="Percentil mínimo para janela de visualização (baseada no HR)")
    ap.add_argument("--pmax", type=float, default=99.0, help="Percentil máximo para janela de visualização (baseada no HR)")
    ap.add_argument("--save_fig", default=None, help="Se setado, salva a figura em PNG")
    ap.add_argument("--no_show", action="store_true", help="Não abrir janela; apenas salvar (se --save_fig)")
    args = ap.parse_args()

    # Decide a chave
    if args.key:
        key = args.key
    else:
        keys = list_keys(args.hr_lmdb, limit=None, regex=args.filter_regex)
        if not keys:
            raise RuntimeError("Nenhuma chave encontrada no HR LMDB (verifique caminhos/regex).")
        key = random.choice(keys) if args.random else keys[0]

    # Lê arrays
    hr = ensure_2d(read_array_from_lmdb(args.hr_lmdb, key))
    lr = ensure_2d(read_array_from_lmdb(args.lr_lmdb, key))

    # Redimensiona LR para o shape da HR (apenas para visualização lado a lado)
    lr_up = resize_like(lr, hr.shape, method=args.method)

    # Normaliza ambos para 8-bit com a MESMA janela (percentis do HR)
    hr8, lr8, (vmin, vmax) = to_uint8_same_window(hr, lr_up, pmin=args.pmin, pmax=args.pmax)

    # Monta figura
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Key: {key}\nHR shape={hr.shape} | LR shape={lr.shape} (upsampled→{lr_up.shape})\nwindow based on HR: vmin={vmin:.4f} vmax={vmax:.4f}", fontsize=9)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(hr8, cmap="gray")
    ax1.set_title("HR")
    ax1.axis("off")
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(lr8, cmap="gray")
    ax2.set_title("LR (upsampled)")
    ax2.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if args.save_fig:
        os.makedirs(os.path.dirname(args.save_fig) or ".", exist_ok=True)
        plt.savefig(args.save_fig, dpi=150)
        print(f"[OK] Figura salva em: {args.save_fig}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # Log rápido no terminal
    print(f"[INFO] key: {key}")
    print(f"[INFO] HR stats: mean={hr.mean():.4f}, std={hr.std():.4f}, min={hr.min():.4f}, max={hr.max():.4f}")
    print(f"[INFO] LR stats: mean={lr.mean():.4f}, std={lr.std():.4f}, min={lr.min():.4f}, max={lr.max():.4f}")

if __name__ == "__main__":
    main()
