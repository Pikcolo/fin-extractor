"""กล่อง: ขอบเขตเนื้อหา (normalize) กรอบฟอร์ม ตัดขอบว่างของ crop (trim_box) — ลบเส้นสกปรกไม่อยู่ที่นี่ ใช้ preprocessing/clean.py

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/preprocess/crop.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน 18 ใบที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Find the reference frame that region ratios are measured against.

Ratios applied to the raw scan drift because scan margins and page sizes differ.
So every page is first reduced to a normalized frame:
  - "content": bounding box of all ink (scanner edge shadows removed)
  - "frame":   the largest ruled rectangle (e.g. the border of the 50 ทวิ form)
"""
from __future__ import annotations

import cv2
import numpy as np

from .masks import ink_mask

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels


def content_bbox(img: np.ndarray, min_area_frac: float = 2e-5, edge_px: int = 8) -> Box:
    mask = ink_mask(img)
    h, w = mask.shape
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_frac * h * w:
            continue
        touches_edge = x <= edge_px or y <= edge_px or x + bw >= w - edge_px or y + bh >= h - edge_px
        thin = min(bw, bh) < 0.02 * min(h, w) and max(bw, bh) > 0.3 * min(h, w)
        if touches_edge and (thin or area > 0.5 * bw * bh):
            continue  # scanner shadow / dark border strip
        keep.append((x, y, x + bw, y + bh))
    if not keep:
        return 0, 0, w, h
    arr = np.array(keep)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 2].max()), int(arr[:, 3].max())


def pad_box(box: Box, shape: tuple[int, ...], pad_px: int) -> Box:
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return max(0, x0 - pad_px), max(0, y0 - pad_px), min(w, x1 + pad_px), min(h, y1 + pad_px)


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()


def trim_box(text: np.ndarray, box: Box, pad_px: int = 6, min_area: int = 8) -> Box | None:
    """Shrink box to the text inside it (+pad). Returns None when nothing is left.

    `text` is the PAGE-level text mask (binarize.text_mask): rules are detected relative to
    the page, so a short stroke inside a small cell is not mistaken for a rule. Ruling lines
    and specks are ignored, so a cell is trimmed to its content, not to its border.
    """
    x0, y0, x1, y1 = box
    region = text[y0:y1, x0:x1]
    if region.size == 0:
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    st = stats[1:]
    # specks, and 1-2 px slivers left over where a ruling line was removed (310: a table
    # rule's residue made a body row look like it had text down to the next rule)
    short = np.minimum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    long_ = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    sliver = (short <= 2) & (long_ >= 15)  # thin AND long = rule residue; Thai marks (่ ั ุ) are small, not long
    keep = st[(st[:, cv2.CC_STAT_AREA] >= min_area) & ~sliver]
    if len(keep) == 0:
        return None
    tx0 = int(keep[:, 0].min())
    ty0 = int(keep[:, 1].min())
    tx1 = int((keep[:, 0] + keep[:, 2]).max())
    ty1 = int((keep[:, 1] + keep[:, 3]).max())
    return (max(x0, x0 + tx0 - pad_px), max(y0, y0 + ty0 - pad_px),
            min(x1, x0 + tx1 + pad_px), min(y1, y0 + ty1 + pad_px))
