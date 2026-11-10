"""ปรับคุณภาพภาพ: grayscale, upscale, denoise, contrast."""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale(img: np.ndarray, min_long_side: int) -> tuple[np.ndarray, float]:
    """ขยายภาพถ้าด้านยาวเล็กกว่า ``min_long_side`` — ภาพสแกนความละเอียดต่ำทำให้ OCR พลาดเยอะ."""
    long_side = max(img.shape[:2])
    if not min_long_side or long_side >= min_long_side:
        return img, 1.0
    scale = min_long_side / long_side
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale


def denoise(img: np.ndarray, method: str = "median") -> np.ndarray:
    """ลด noise จากการสแกน (จุดดำ/เกรน JPEG).

    - median    : เร็ว เหมาะกับ salt-and-pepper
    - nlmeans   : คุณภาพดีสุด แต่ช้ากว่า
    - bilateral : ลด noise แต่รักษาขอบตัวอักษร
    """
    if method in (None, "none"):
        return img
    if method == "median":
        return cv2.medianBlur(img, 3)
    if method == "nlmeans":
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 50, 50)
    raise ValueError(f"unknown denoise method: {method}")


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """เพิ่ม contrast เฉพาะที่ — ช่วยภาพที่แสงไม่สม่ำเสมอ/ซีด. ภาพสีจะปรับเฉพาะช่อง L."""
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    if img.ndim == 2:
        return op.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
