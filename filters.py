import cv2
import numpy as np

def is_predominantly_dark(img, threshold=20, percentage=0.7):
    """Retorna True se mais de `percentage` dos pixels forem menores que `threshold`."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(img < threshold)
    total_pixels = img.shape[0] * img.shape[1]
    return dark_pixels / total_pixels > percentage