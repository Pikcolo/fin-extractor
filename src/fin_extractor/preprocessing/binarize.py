"""แปลงภาพเป็นขาวดำ (ตัวอักษรดำ พื้นขาว) — ปิดเป็นค่าตั้งต้น: Otsu ทิ้งตัวอักษรสีอ่อน (ดู docs/02-preprocessing.md)."""

from __future__ import annotations

import cv2
import numpy as np

from .enhance import to_gray


def otsu(gray: np.ndarray) -> np.ndarray:
    """threshold เดียวทั้งภาพ — ดีกับภาพสแกนที่แสงสม่ำเสมอ."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, out = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def adaptive(gray: np.ndarray, block_size: int = 31, c: int = 15) -> np.ndarray:
    """threshold ตามพื้นที่ — ดีกับภาพถ่ายมือถือที่มีเงา."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def sauvola(gray: np.ndarray, window: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Sauvola: T = mean * (1 + k * (std / r - 1)) — นิยมกับเอกสารที่พื้นหลังไม่เรียบ/กระดาษเก่า."""
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window))
    sq_mean = cv2.boxFilter(g * g, -1, (window, window))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    thresh = mean * (1 + k * (std / r - 1))
    return np.where(g > thresh, 255, 0).astype(np.uint8)


def binarize(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method in (None, "none"):
        return img
    gray = to_gray(img)
    if method == "otsu":
        return otsu(gray)
    if method == "adaptive":
        return adaptive(gray)
    if method == "sauvola":
        return sauvola(gray)
    raise ValueError(f"unknown binarize method: {method}")
