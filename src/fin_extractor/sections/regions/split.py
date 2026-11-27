"""แบ่ง region: แถบตามเส้นนอน / แถบตามช่องว่าง (ไม่มี template รายชนิด: ทุกเอกสารแบ่งแบบเดียวกัน วัด 50 ทวิ 4 ใบแล้วดีกว่า template 3 ใบ ดู docs/03-regions.md)

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/layout/regions.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน (ชุดสแกนทดสอบ) ที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Split a normalized page into OCR-sized regions.

Strips along long horizontal rules; with fewer rules than min_lines, horizontal bands at whitespace gaps.

All boxes are in pixels of the normalized page: (x0, y0, x1, y1).
"""
from __future__ import annotations

import numpy as np

from . import lines
from .geometry import Box, pad_box
from .masks import ink_mask, text_mask



# --------------------------------------------------------------------------- generic

def whitespace_bands(mask: np.ndarray, min_gap: int, min_band: int) -> list[tuple[int, int]]:
    """Return [(y0, y1)] runs of rows containing ink, separated by gaps >= min_gap rows."""
    rows = (mask > 0).sum(axis=1) > max(2, mask.shape[1] * 0.002)
    bands, start, gap = [], None, 0
    for y, has_ink in enumerate(rows):
        if has_ink:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                bands.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        bands.append((start, len(rows) - gap))
    return [b for b in bands if b[1] - b[0] >= min_band]


def merge_bands(bands: list[tuple[int, int]], max_bands: int) -> list[tuple[int, int]]:
    """Merge the pair of neighbours with the smallest gap until at most max_bands remain."""
    bands = list(bands)
    while len(bands) > max_bands:
        gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
        i = int(np.argmin(gaps))
        bands[i:i + 2] = [(bands[i][0], bands[i + 1][1])]
    return bands


def line_strips(img: np.ndarray, frame: Box, hlines: list[dict], cfg: dict) -> tuple[list[dict], list[int]]:
    """Cut the frame into full-width strips at horizontal rules.

    Cuts that would pass through text (rules spanning only part of the page) are moved to
    the nearest empty row or dropped. Thin strips are merged; empty strips are dropped.
    """
    fx0, fy0, fx1, fy1 = frame
    text = text_mask(ink_mask(img))
    cuts = [fy0, fy1]
    for l in hlines:
        if l["len_frac"] >= cfg["cut_min_len_frac"] and fy0 < l["y"] < fy1:
            c = lines.safe_cut(l, text, cfg["cut_search_px"])
            if c is not None:
                cuts.append(c)
    cuts = sorted(set(cuts))
    spans = [(a, b) for a, b in zip(cuts, cuts[1:])]
    merged: list[tuple[int, int]] = []
    for a, b in spans:  # a strip thinner than one text line is a double rule / cell gap: join the next one
        if merged and merged[-1][1] - merged[-1][0] < cfg["min_strip_px"]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < cfg["min_strip_px"]:
        merged[-2:] = [(merged[-2][0], merged[-1][1])]
    out = []
    for a, b in merged:
        if (text[a:b, fx0:fx1] > 0).mean() < cfg["min_strip_ink"]:
            continue  # empty table row / blank area
        out.append({"name": f"strip_{len(out):02d}", "box": (fx0, a, fx1, b)})
    return out, cuts


def generic_regions(img: np.ndarray, frame: Box, cfg: dict, hlines: list[dict] | None = None) -> list[dict]:
    mask = ink_mask(img)
    h, w = mask.shape
    fx0, fy0, fx1, fy1 = frame
    pad_px = int(cfg["pad_frac"] * (fy1 - fy0))
    out: list[dict] = []

    cut_lines = [l for l in (hlines or []) if l["len_frac"] >= cfg["cut_min_len_frac"]]
    if len(cut_lines) >= cfg["min_lines"]:
        out, _ = line_strips(img, frame, hlines, cfg)
        layout = "lines"
    else:
        tm = text_mask(mask)[fy0:fy1, fx0:fx1]
        bands = whitespace_bands(tm, int(cfg["band_gap_frac"] * (fy1 - fy0)), cfg["min_band_px"])
        bands = merge_bands(bands, cfg["max_bands"])
        for i, (y0, y1) in enumerate(bands):
            out.append({"name": f"band_{i:02d}", "box": (fx0, fy0 + y0, fx1, fy0 + y1)})
        layout = "bands"

    if layout == "lines":
        pad_px = cfg["line_pad_px"]  # strips already end on rules; big padding would pull in the next row
    for r in out:
        r["box"] = pad_box(r["box"], img.shape, pad_px)
        r["mode"] = f"generic:{layout}"
    return out


def merge_ruled(regs: list[dict], vlines: list[dict], tol_px: int) -> list[dict]:
    """รวมแถบที่ติดกันซึ่งรอยต่อมีเส้นตั้งพาดผ่าน (= กลางตารางที่มีคอลัมน์ / กล่องที่มีเส้นขอบ) เป็น region เดียวก่อนส่ง OCR
    ตัดตารางตามแถวแล้ว แถวข้อมูลแยกจากหัวคอลัมน์ OCR ไม่รู้ว่าค่าอยู่คอลัมน์ไหน แถบรายแถวยังใช้ในขั้นตาราง (ช่อง/คีย์) ตามเดิม
    รอยต่อ = กึ่งกลางระหว่างขอบล่างของแถบบนกับขอบบนของแถบล่าง ปลายเส้นห่างไม่เกิน ``tol_px`` ไม่นับ (ขอบบน/ล่างของตาราง)
    ``header`` = กล่องของแถบแรกของกลุ่ม (แถวหัวคอลัมน์ ถึงเส้นแบ่งแถวเส้นแรก) ใช้แปะซ้ำบนท่อนถัดไปเมื่อตารางถูกตัดตามความสูง."""
    out: list[dict] = []
    for r in sorted(regs, key=lambda r: r["box"][1]):
        if out:
            prev = out[-1]
            seam = (prev["box"][3] + r["box"][1]) // 2
            x0, x1 = min(prev["box"][0], r["box"][0]), max(prev["box"][2], r["box"][2])
            if lines.crosses_column_rule(seam, vlines, x0, x1, tol_px):
                first = prev["name"].split("-")[0]
                out[-1] = {**prev, "name": f"{first}-{r['name'].split('_')[-1]}", "header": prev.get("header", prev["box"]),
                           "box": (x0, min(prev["box"][1], r["box"][1]), x1, max(prev["box"][3], r["box"][3])), "merged": prev.get("merged", 1) + 1}
                continue
        out.append(dict(r))
    return out
