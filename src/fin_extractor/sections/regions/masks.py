"""หน้ากากหมึก/เส้น/ข้อความ ที่ใช้ตรวจโครงหน้า (ไม่ส่งเข้า OCR)

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/imaging/binarize.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน 18 ใบที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Shared binary-mask helpers used by detection stages (never exported to OCR)."""
from __future__ import annotations

import cv2
import numpy as np

from ...preprocessing.enhance import to_gray


def ink_mask(img: np.ndarray) -> np.ndarray:
    """255 where there is ink (text/lines), 0 for paper. Otsu on a lightly blurred gray."""
    gray = cv2.GaussianBlur(to_gray(img), (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def line_masks(mask: np.ndarray, min_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Extract long horizontal and vertical rules with morphological opening.

    min_frac: minimum line length as a fraction of image width (horizontal) / height (vertical).
    """
    h, w = mask.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * min_frac)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * min_frac * 0.5))))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    return horiz, vert


def text_mask(mask: np.ndarray) -> np.ndarray:
    """Ink mask with long rules removed, so only text/handwriting remains."""
    horiz, vert = line_masks(mask, min_frac=0.1)
    return cv2.subtract(mask, cv2.bitwise_or(horiz, vert))
