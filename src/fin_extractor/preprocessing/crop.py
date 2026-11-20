"""ขั้น crop: ตัดขอบว่าง/ขอบดำจากการสแกน (ปิดเป็นค่าตั้งต้น ดู docs/02)."""

from __future__ import annotations

import cv2
import numpy as np

from .enhance import to_gray


def crop_to_content(img: np.ndarray, margin: int = 20) -> np.ndarray:
    """ตัดขอบว่าง/ขอบดำจากการสแกนออก โดยหา bounding box ของหมึกทั้งหมด."""
    gray = to_gray(img)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ขอบดำของสแกนเนอร์จะติดขอบภาพ — ลบ component ที่แตะขอบก่อน
    h, w = inv.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            inv[labels == i] = 0
    pts = cv2.findNonZero(inv)
    if pts is None:
        return img
    x, y, bw, bh = cv2.boundingRect(pts)
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + bw + margin, w), min(y + bh + margin, h)
    return img[y0:y1, x0:x1]
