"""ตาราง: แถบ → แถว × คอลัมน์ (เส้นตั้ง) → ช่อง + คู่ key/value ตามตำแหน่ง

คัดลอกจาก learning/preprocess_finance (src/finance_ocr/layout/tables.py) ตรงตัว ค่าคงที่วัดบนสแกนชุดเดียวกัน 18 ใบที่ 200 dpi — ใช้กับภาพที่ย่อเป็น 200 dpi เท่านั้น (ดู run.py)

Tables from ruling lines: rows (horizontal rules / whitespace) x columns (vertical rules).

Key/value without OCR comes from *position*:
  - a table's first row is its header -> each header cell is a KEY
  - every cell below is a VALUE; its key is the header cell it overlaps most in x
    (overlap, not column index, so a merged header such as "จำนวนเงิน (บาท)" spanning
    the baht and satang columns is the key for both)
The header guess is backed by layout evidence recorded in the manifest (first row, shaded
background, fills every column, shorter than body rows); text-based checks belong after OCR.
"""
from __future__ import annotations

import cv2
import numpy as np

from .masks import to_gray

Box = tuple[int, int, int, int]


# --------------------------------------------------------------------------- columns
def separators(box: Box, vlines: list[dict], cover: float, merge_px: int = 12,
               text_span: tuple[int, int] | None = None) -> list[int]:
    """x positions of vertical rules crossing >= cover of the row's height.

    With text_span, height = the rows that actually hold text: a body strip often runs far
    below the last item (310: items end at y~1010, strip ends at the totals box ~1700).
    """
    x0, y0, x1, y1 = box
    if text_span:
        y0, y1 = text_span
    h = max(1, y1 - y0)
    xs = sorted(l["x"] for l in vlines
                if x0 - merge_px <= l["x"] <= x1 + merge_px
                and min(y1, l["y1"]) - max(y0, l["y0"]) >= cover * h)
    out: list[int] = []
    for x in xs:
        if out and x - out[-1] <= merge_px:
            out[-1] = (out[-1] + x) // 2
        else:
            out.append(x)
    return out


def compatible(a: list[int], b: list[int], tol: int) -> bool:
    """Same table? Most column rules of the sparser row recur in the other row.

    Outer borders are not required to match: a border rule can start a little below the
    header row's top (308) and then fails the coverage test for that row only.
    """
    if len(a) < 2 or len(b) < 2:
        return False
    # judge on INNER rules only: stacked boxes share the page-wide outer borders without being
    # one table (310: customer-info box above the item table)
    lo, hi = max(a[0], b[0]) + tol, min(a[-1], b[-1]) - tol
    ia, ib = [x for x in a if lo < x < hi], [x for x in b if lo < x < hi]
    if not ia or not ib:
        return False
    small, big = (ia, ib) if len(ia) <= len(ib) else (ib, ia)
    if len(small) < 0.5 * len(big):
        return False  # structure changes (310: 8-column item rows -> 2-column totals box)
    hits = sum(any(abs(x - y) <= tol for y in big) for x in small)
    return hits >= 0.6 * len(small)


# --------------------------------------------------------------------------- rows without rules
def _ink_runs(mask: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """[(y0, y1)] runs of rows that contain ink, split at gaps >= min_gap rows."""
    rows = mask.sum(axis=1) > 0
    runs, start, gap = [], None, 0
    for y, ink in enumerate(rows):
        if ink:
            start = y if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                runs.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(rows) - gap))
    return [r for r in runs if r[1] - r[0] >= 4]


def _no_slivers(mask: np.ndarray) -> np.ndarray:
    """Drop thin leftovers of removed column rules (they would look like a new item start)."""
    if mask.size == 0:
        return mask > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    w_, h_ = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    # thin, or taller than two text lines yet narrow (rule residue linking two items, 309)
    keep[1:] = (w_ >= 3) & (stats[1:, cv2.CC_STAT_AREA] >= 12) & ~((h_ > 60) & (w_ <= 8))
    return keep[labels]


def split_rows(text: np.ndarray, box: Box, seps: list[int], min_gap: int, min_row: int,
               text_span: tuple[int, int] | None = None, anchor_tol: int = 40) -> list[dict[int, tuple[int, int]]]:
    """Split a tall body row (no horizontal rules between items) into items.

    Items are found in the ANCHOR column: the leftmost column holding text on the first line
    (item no. / product code). Each new entry there starts an item. A column with exactly one
    text line per item (number columns) is paired up in order - robust when its values are
    printed offset from the product code (309: "2", "3,860.00" sit ~20 px higher). Other
    columns give each line to the last item that started before the line's centre (+ half a
    line), so a wrapped description line ("สีแดง", 310) stays with its item.
    Returns one {col: (y0, y1)} per item.
    """
    x0, y0, x1, y1 = box
    if text_span:  # ignore rule residue at the strip edges
        y0, y1 = max(y0, text_span[0]), min(y1, text_span[1])
    cols = list(zip(seps, seps[1:]))
    runs = {c: [r for r in _ink_runs(_no_slivers(text[y0:y1, a + 6:b - 6]), min_gap) if r[1] - r[0] >= min_row]
            for c, (a, b) in enumerate(cols)}
    filled = [c for c in range(len(cols)) if runs[c]]
    if not filled:
        return []
    top = min(runs[c][0][0] for c in filled)
    anchor = next(c for c in filled if runs[c][0][0] <= top + anchor_tol)
    starts = [r[0] for r in runs[anchor]]
    half_line = int(np.median([r[1] - r[0] for r in runs[anchor]]) / 2)
    items: list[dict[int, tuple[int, int]]] = [{} for _ in starts]
    for c in filled:
        in_order = len(runs[c]) == len(starts)  # one line per item (number columns): pair them up
        for j, (r0, r1) in enumerate(runs[c]):
            centre = (r0 + r1) // 2
            i = j if in_order else max((k for k, st in enumerate(starts) if st <= centre + half_line), default=0)
            a_, b_ = items[i].get(c, (r0, r1))
            items[i][c] = (min(a_, r0), max(b_, r1))
    return [{c: (y0 + max(0, a_ - 3), y0 + b_ + 3) for c, (a_, b_) in it.items()} for it in items if it]


# --------------------------------------------------------------------------- header evidence
def header_evidence(img: np.ndarray, header: Box, body: list[Box], header_cells: list[dict]) -> dict:
    gray = to_gray(img)

    def bg(b):  # median of the lighter half = paper/fill colour, not the text
        px = gray[b[1]:b[3], b[0]:b[2]].ravel()
        return float(np.median(px[px >= np.median(px)])) if px.size else 255.0

    body_bg = np.median([bg(b) for b in body]) if body else 255.0
    body_h = np.median([b[3] - b[1] for b in body]) if body else 0
    ev = {
        "first_row": True,
        "shaded": bool(bg(header) < body_bg - 12),
        "all_columns_filled": bool(header_cells) and all(not c["empty"] for c in header_cells),
        "shorter_than_body": bool(body_h and (header[3] - header[1]) < body_h),
    }
    ev["score"] = round(sum(ev[k] for k in ("first_row", "shaded", "all_columns_filled", "shorter_than_body")) / 4, 2)
    return ev


def _borrow_header_seps(text: np.ndarray, header: Box, inner: list[int], body_seps: list[list[int]],
                        tol: int) -> list[int]:
    """A header may lack a rule that every body row has (309: no line between ส่วนลด and
    จำนวนเงิน in the header). Borrow it - unless header text runs across that x, which means
    the header cell is deliberately merged (320: "จำนวนเงิน (บาท)" over baht | satang)."""
    if not body_seps:
        return inner
    y0, y1 = header[1], header[3]
    lo, hi = header[0] + tol, header[2] - tol
    out = list(inner)
    for x in body_seps[0]:
        if not lo < x < hi or any(abs(x - h) <= tol for h in out):
            continue
        if _text_crosses(text[y0:y1], x):
            continue
        # both halves of the split header cell must hold text, otherwise it was not a header split
        cut = sorted([header[0], *out, header[2]])
        a = max(c for c in cut if c < x)
        b = min(c for c in cut if c > x)
        if (text[y0:y1, a + 6:x - 4] > 0).sum() > 30 and (text[y0:y1, x + 5:b - 6] > 0).sum() > 30:
            out.append(x)
    return sorted(out)


def _text_crosses(band: np.ndarray, x: int) -> bool:
    """Does text run across column x? Glyphs crossing x have ink on both sides in the same
    pixel rows; a short header rule (309: too short to be removed as a rule) has paper beside it."""
    mid = band[:, max(0, x - 4):x + 5].any(axis=1)
    left = band[:, max(0, x - 10):max(0, x - 5)].any(axis=1)
    right = band[:, x + 5:x + 10].any(axis=1)
    return bool((mid & left & right).sum() >= 3)


def _is_header_bar(img: np.ndarray, box: Box, cfg: dict) -> bool:
    x0, y0, x1, y1 = box
    if y1 - y0 > cfg["max_header_h"]:
        return False
    px = to_gray(img)[y0:y1, x0:x1].ravel()
    return px.size > 0 and float(np.median(px)) < cfg["header_bar_max_gray"]


# --------------------------------------------------------------------------- build
def build_tables(img: np.ndarray, text: np.ndarray, strips: list[dict], vlines: list[dict], cfg: dict,
                 trim) -> list[dict]:
    """Group consecutive strips into tables and cut cells. `trim(box)` -> trimmed box or None."""
    def span(s):
        t = trim(s["box"])
        return (t[1], t[3]) if t else None

    seps = [separators(s["box"], vlines, cfg["vline_cover"], text_span=span(s)) for s in strips]
    tabular = [len(s) >= 3 for s in seps]  # >= 2 cells; 2 rules alone are just a box border

    groups: list[list[int]] = []
    i = 0
    while i < len(strips):
        if not tabular[i]:
            i += 1
            continue
        g = [i]
        j = i + 1
        while j < len(strips) and tabular[j] and compatible(seps[g[0]], seps[j], cfg["col_tol_px"]):
            if _is_header_bar(img, strips[j]["box"], cfg):
                break  # a shaded header bar starts a new table; it is never a value row (306)
            g.append(j)
            j += 1
        # header bar without visible rules (white separators inside a filled bar, 306/318): inherit
        # the columns of the row below - only for a short, shaded strip, not any text above a table
        if i > 0 and not tabular[i - 1] and _is_header_bar(img, strips[i - 1]["box"], cfg):
            seps[i - 1] = seps[i]
            g.insert(0, i - 1)
        groups.append(g)
        i = j

    tables = []
    for g in [g for g in groups if len(g) >= 2]:  # a lone ruled row has no values under it
        ti = len(tables)
        # the header uses its own rules if it has them (merged header cells), body rows theirs
        header_seps = seps[g[0]]
        all_seps = sorted({x for k in g for x in seps[k]})
        left, right = all_seps[0], all_seps[-1]
        rows = []
        for k in g:
            sb = strips[k]["box"]
            # every row shares the table's outer borders; inner rules are the row's own
            # (keeps merged header cells merged, and split baht/satang columns split)
            inner = [x for x in seps[k] if left + cfg["col_tol_px"] < x < right - cfg["col_tol_px"]]
            if k == g[0]:
                # rules of the first body row (the item rows); totals rows below often differ
                inner = _borrow_header_seps(text, sb, inner, [seps[g[1]]], cfg["col_tol_px"])
            row_seps = [left, *inner, right]
            items = None
            if k != g[0] and sb[3] - sb[1] > cfg["split_row_min_h"]:
                items = split_rows(text, sb, row_seps, cfg["row_gap_px"], cfg["min_row_h"],
                                   text_span=span(strips[k]), anchor_tol=cfg["anchor_tol_px"]) or None
            if items is None:  # the strip itself is one row
                items = [{c: (sb[1], sb[3]) for c in range(len(row_seps) - 1)}]
            for it in items:
                cells = []
                for ci, (a, b) in enumerate(zip(row_seps, row_seps[1:])):
                    if ci in it:
                        cbox = (a + 2, it[ci][0], b - 2, it[ci][1])
                        tb = trim(cbox)
                    else:
                        cbox, tb = (a + 2, min(v[0] for v in it.values()), b - 2, max(v[1] for v in it.values())), None
                    cells.append({"col": ci, "box": cbox, "trim": tb, "empty": tb is None})
                ry0, ry1 = min(v[0] for v in it.values()), max(v[1] for v in it.values())
                rows.append({"strip": strips[k]["name"], "box": (row_seps[0], ry0, row_seps[-1], ry1),
                             "kind": "header" if k == g[0] else "value", "cells": cells})
        header = rows[0]
        for r in rows[1:]:
            prev = None
            for c in r["cells"]:
                best, best_ov = None, 0
                for hc in header["cells"]:
                    ov = min(c["box"][2], hc["box"][2]) - max(c["box"][0], hc["box"][0])
                    if ov > best_ov:
                        best, best_ov = hc["col"], ov
                c["key_col"] = best      # candidate 1: header cell above (column table)
                c["left_col"] = prev     # candidate 2: text cell to the left (label | value rows)
                if not c["empty"]:
                    prev = c["col"]
        body = [r["box"] for r in rows[1:]]
        ev = header_evidence(img, header["box"], body, header["cells"])
        tables.append({
            "id": ti,
            # grid: header row + value rows. boxes: few columns / weak header -> decide the key
            # per cell after OCR (header above vs label on the left)
            # value rows must be at least as finely split as the header (merged header cells are
            # fine, 320); fewer cells below = stacked info boxes, not a column table (306)
            "layout": "grid" if (len(header["cells"]) >= 3 and ev["score"] >= 0.5 and
                                 np.mean([len(r["cells"]) >= len(header["cells"]) for r in rows[1:]]) >= 0.5)
                      else "boxes",
            "box": (min(all_seps), rows[0]["box"][1], max(all_seps), rows[-1]["box"][3]),
            "strips": [strips[k]["name"] for k in g],
            "header_columns": header_seps,
            "header_evidence": ev,
            "rows": rows,
        })
    return tables
