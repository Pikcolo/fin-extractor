"""ขั้น deskew: ประมาณมุมเอียงเล็กน้อยด้วย projection profile แล้วหมุนแก้."""

from __future__ import annotations

import cv2
import numpy as np

from .enhance import to_gray


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """หมุนภาพทวนเข็มนาฬิกา ``angle`` องศา โดยขยาย canvas ไม่ให้มุมโดนตัด และเติมพื้นขาว."""
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    border = (255,) * (3 if img.ndim == 3 else 1)
    return cv2.warpAffine(
        img, m, (new_w, new_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def _projection_score(binary: np.ndarray, angle: float) -> float:
    # บรรทัดข้อความที่ตรง → ปริมาณหมึกต่อแถวมีพีคคม → ผลต่างระหว่างแถวติดกันสูง
    # rotate เติมขอบสีขาว จึงต้องกลับสีหลังหมุน ไม่ให้ขอบถูกนับเป็นหมึก
    ink = 255 - rotate(binary, angle)
    rows = ink.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew(img: np.ndarray, max_angle: float = 15.0) -> float:
    """ประมาณมุมเอียงด้วย projection profile (coarse 1° → fine 0.1°).

    ใช้ได้ทั้งเอกสารมีตารางและไม่มีตาราง; คืนมุมที่ต้องหมุน (ทวนเข็มนาฬิกา) เพื่อแก้เอียง
    """
    gray = to_gray(img)
    scale = 1000 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def score(a: float) -> float:
        return _projection_score(binary, a)

    coarse = np.arange(-max_angle, max_angle + 0.5, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, 0.1)
    return round(float(max(fine, key=score)), 2)


def deskew(img: np.ndarray, max_angle: float = 15.0, min_angle: float = 0.1) -> tuple[np.ndarray, float]:
    angle = estimate_skew(img, max_angle)
    if abs(angle) < min_angle:
        return img, 0.0
    return rotate(img, angle), angle
