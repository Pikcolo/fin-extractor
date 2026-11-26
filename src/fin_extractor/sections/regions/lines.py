"""เส้นตารางแนวนอน/แนวตั้ง + หาแนวตัดที่ไม่ผ่าข้อความ

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/layout/lines.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน 18 ใบที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Horizontal ruling lines: the most reliable structure on table-style finance documents.

Used two ways:
  - cut:   split generic documents into strips between lines (header / table rows / footer)
"""
from __future__ import annotations

import cv2
import numpy as np

from .masks import ink_mask, line_masks


def detect_hlines(img: np.ndarray, min_len_frac: float = 0.25, merge_px: int = 8) -> list[dict]:
    """Return horizontal rules as [{y, x0, x1, len_frac}], sorted by y.

    No gap closing on purpose: closing turns dotted leaders ("......") into fake rules
    (50 ทวิ: 11 real rules -> 32). Scanner streaks are removed earlier, in normalize.
    Rules closer than merge_px (double rules, thick bars) are merged into one.
    """
    mask = ink_mask(img)
    h, w = mask.shape
    horiz, _ = line_masks(mask, min_frac=0.08)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))  # join broken thickness only
    n, _, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    raw = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bw < min_len_frac * w or bh > 0.02 * h:
            continue
        raw.append({"y": int(y + bh // 2), "x0": int(x), "x1": int(x + bw)})
    raw.sort(key=lambda l: l["y"])
    merged: list[dict] = []
    for l in raw:
        if merged and l["y"] - merged[-1]["y"] <= merge_px:
            m = merged[-1]
            m["x0"], m["x1"] = min(m["x0"], l["x0"]), max(m["x1"], l["x1"])
            m["y"] = (m["y"] + l["y"]) // 2
        else:
            merged.append(dict(l))
    for l in merged:
        l["len_frac"] = round((l["x1"] - l["x0"]) / w, 3)
    return merged


def detect_vlines(img: np.ndarray, min_len_frac: float = 0.03, side_max_ink: float = 0.5,
                  merge_px: int = 6) -> list[dict]:
    """Vertical rules as [{x, y0, y1}], sorted by x.

    min_len_frac is relative to page WIDTH, so it stays above a character's height
    (~30 px at 200 DPI) also on landscape pages. A rule must have paper on both sides:
    inside a filled header bar (306 blue, 318 purple) every column is "ink", which would
    otherwise look like dozens of vertical lines.
    """
    mask = ink_mask(img)
    h, w = mask.shape
    min_len = int(min_len_frac * w)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))  # join broken thickness only
    n, _, stats, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    raw = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bh < min_len or bw > 0.015 * w:
            continue
        left, right = max(0, x - 5), min(w - 1, x + bw + 4)
        if (mask[y:y + bh, left] > 0).mean() > side_max_ink or (mask[y:y + bh, right] > 0).mean() > side_max_ink:
            continue  # part of a filled block, not a rule
        raw.append({"x": int(x + bw // 2), "y0": int(y), "y1": int(y + bh)})
    raw.sort(key=lambda l: (l["x"], l["y0"]))
    merged: list[dict] = []
    for l in raw:  # same x and touching/overlapping in y -> one rule
        m = next((m for m in merged if abs(m["x"] - l["x"]) <= merge_px and l["y0"] <= m["y1"] + 10), None)
        if m:
            m["y0"], m["y1"] = min(m["y0"], l["y0"]), max(m["y1"], l["y1"])
        else:
            merged.append(dict(l))
    return sorted(merged, key=lambda l: l["x"])


MIN_COLUMN_RULES = 3  # ตารางที่มีอย่างน้อย 2 คอลัมน์มีเส้นตั้งอย่างน้อย 3 เส้น (ซ้าย · เส้นแบ่ง · ขวา) ขอบกรอบเฉย ๆ มีแค่ 2 (ซ้าย/ขวา) ไม่นับเป็นตาราง


def crosses_column_rule(y: int, vlines: list[dict], x0: int, x1: int, tol_px: int) -> bool:
    """แนวตัด y อยู่กลางตารางที่มีคอลัมน์ไหม: เส้นตั้งในช่วง x0–x1 ที่พาดผ่าน y (ห่างจากปลายเส้นเกิน ``tol_px``) มีอย่างน้อย ``MIN_COLUMN_RULES`` เส้น
    ตัดตรงนั้นแถวจะแยกจากหัวคอลัมน์ ปลายเส้นตรงขอบบน/ล่างของตารางไม่นับ และขอบกรอบเอกสาร (ซ้าย/ขวา 2 เส้น) อย่างเดียวไม่นับ
    (1308: ขอบกรอบพาดตั้งแต่กล่องลูกค้าถึงยอดรวม ถ้านับ ทั้งหน้าจะกลายเป็นภาพเดียว)."""
    return sum(x0 <= v["x"] <= x1 and v["y0"] + tol_px < y < v["y1"] - tol_px for v in vlines) >= MIN_COLUMN_RULES


def safe_cut(line: dict, text: np.ndarray, search_px: int, max_ink_frac: float = 0.001,
             margin_px: int = 10) -> int | None:
    """Where to cut for this rule without slicing text, or None.

    On the rule itself only the columns OUTSIDE the rule's span matter (inside, the rule
    separates rows by definition; line-removal residue sits there too). If text crosses
    that row beside a short rule, look for a fully empty row (5 rows tall) within search_px.
    """
    h, w = text.shape
    y, x0, x1 = line["y"], max(0, line["x0"] - margin_px), min(w, line["x1"] + margin_px)
    rows = text[max(0, y - 2):y + 3] > 0
    if (rows[:, :x0].sum() + rows[:, x1:].sum()) <= max_ink_frac * w:
        return y
    row_ink = (text > 0).sum(axis=1) / w
    for d in range(1, search_px + 1):
        for yy in (y - d, y + d):
            if 2 <= yy < h - 2 and row_ink[yy - 2:yy + 3].max() <= max_ink_frac:
                return yy
    return None
