"""ตัด region สูงเป็นท่อนก่อนส่ง OCR โดยไม่ตัดกลางรายการ (แทน ``sections.layout.split_rows`` ในโหมด regions)

ปัญหาเดิม: ตัดที่แถวว่างที่ใกล้จุดแบ่งเท่า ๆ กันที่สุด แถวว่างนั้นอาจอยู่ระหว่างบรรทัดของรายการเดียวกัน
(1306: บรรทัดสุดท้ายของคำอธิบายรายการหลุดไปอยู่ท่อนถัดไปคนเดียว) และเศษเส้นตารางที่ลบไม่หมดทำให้หดขอบไม่ลง region จึงสูงเกินจริงแล้วถูกตัด

ลำดับ
1. **บรรทัดข้อความ**: ระบายแถบรอบเส้นตั้งที่พาด region (เศษเส้นคอลัมน์) ทิ้งเศษเส้นแบบ ``tables._no_slivers``
   แล้วหาช่วงแถวที่มีหมึก ช่วงที่สูงไม่เกิน ``SLIVER_H`` คือเศษเส้นแนวนอน ทิ้ง (เกณฑ์เดียวกับ ``geometry.trim_box``)
   ช่วงที่สูงถึง ``min_line_h`` นับเป็น **บรรทัด** (ใช้หาจุดตัด) ช่วงที่เตี้ยกว่านั้นยังเป็นหมึกที่ต้องอยู่ในท่อน
   (ตัวเล็กที่ถูกขอบ region ตัด เช่น "Page 1 of 1" ของ 1308 ป้าย "Remark" ของ 1317)
2. **หดขอบบน/ล่าง** ให้เหลือตั้งแต่หมึกแรกถึงหมึกสุดท้าย (+ ``pad``) ถ้าสูงไม่เกิน ``max_h`` ก็จบ ไม่ตัด
3. **จุดตัดที่ยอมได้**: ถ้ามีคอลัมน์ซ้ายสุดแบบตาราง (เส้นตั้งด้านใน) และมีบรรทัดที่มีข้อความในคอลัมน์นั้น ≥ 2 บรรทัด
   (เลขข้อ/รหัสสินค้า = รายการใหม่) ตัดได้เฉพาะช่องว่างก่อนบรรทัดเหล่านั้น ไม่งั้นตัดได้ทุกช่องว่างระหว่างบรรทัด
4. **เลือกจุดตัด** ทีละจุดจากบนลงล่าง: ในช่วงที่ท่อนสูงครึ่งถึงเต็ม ``max_h`` เลือกช่องว่างที่ **กว้างที่สุด**
   (ช่องว่างระหว่างรายการ/ย่อหน้ากว้างกว่าระหว่างบรรทัดในรายการเดียวกัน) ถ้าไม่มีจุดตัดในช่วง ใช้จุดแรกที่เลยช่วงไป
   (ท่อนสูงเกินดีกว่าตัดกลางรายการ) ไม่เหลือจุดตัดก็หยุด
ค่าที่ใช้มาจาก config เดิม: ``min_line_h`` = ``tables.min_row_h`` (12 px) ``pad`` = ``export.trim_pad_px`` (6 px) ไม่มีค่าใหม่
"""

from __future__ import annotations

import numpy as np

from .tables import _no_slivers

Box = tuple[int, int, int, int]
SLIVER_H = 2  # ช่วงแถวหมึกที่สูงไม่เกินนี้ = เศษเส้นตารางที่ลบไม่หมด (``geometry.trim_box`` ทิ้งเศษบาง ≤ 2 px แบบเดียวกัน)


def ink_runs(text: np.ndarray, box: Box, vlines: list[dict], rule_pad: int = 6) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """(หน้ากากข้อความของ region ที่ลบเศษเส้นแล้ว, [(y0, y1)] ช่วงแถวที่มีหมึก สูงเกิน ``SLIVER_H`` ในพิกัดหน้า)."""
    x0, y0, x1, y1 = box
    sub = text[y0:y1, x0:x1].copy()
    for v in vlines:
        if x0 <= v["x"] <= x1 and v["y1"] > y0 and v["y0"] < y1:
            sub[:, max(0, v["x"] - x0 - rule_pad):max(0, v["x"] - x0 + rule_pad)] = 0
    sub = _no_slivers(sub)
    rows = sub.any(axis=1)
    runs, start = [], None
    for y, ink in enumerate(rows):
        if ink and start is None:
            start = y
        elif not ink and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, len(rows)))
    return sub, [(a + y0, b + y0) for a, b in runs if b - a > SLIVER_H]


def text_lines(text: np.ndarray, box: Box, vlines: list[dict], min_line_h: int, rule_pad: int = 6) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """(หน้ากากข้อความ, [(y0, y1)] บรรทัดที่สูงถึง ``min_line_h``)."""
    sub, runs = ink_runs(text, box, vlines, rule_pad)
    return sub, [r for r in runs if r[1] - r[0] >= min_line_h]


def item_starts(sub: np.ndarray, box: Box, lines: list[tuple[int, int]], vlines: list[dict], rule_pad: int = 6) -> list[int]:
    """y ของบรรทัดที่มีข้อความในคอลัมน์ซ้ายสุด (ระหว่างขอบซ้ายของ region กับเส้นตั้งด้านในเส้นแรก) = รายการใหม่
    เส้นตั้งต้องพาดอย่างน้อยครึ่งของช่วงบรรทัด ไม่มีเส้นแบบนั้น = ไม่รู้ว่ารายการเริ่มที่ไหน (คืน [])."""
    if not lines:
        return []
    x0, y0, x1, _y1 = box
    top, bottom = lines[0][0], lines[-1][1]
    inner = sorted(v["x"] for v in vlines
                   if x0 + 2 * rule_pad < v["x"] < x1 - 2 * rule_pad and min(bottom, v["y1"]) - max(top, v["y0"]) >= 0.5 * (bottom - top))
    if not inner:
        return []
    col = sub[:, :inner[0] - x0 - rule_pad]
    return [a for a, b in lines if col[a - y0:b - y0].any()]


def cut_region(text: np.ndarray, box: Box, vlines: list[dict], max_h: float, min_line_h: int, pad: int) -> dict:
    """``{"pieces": [(y0, y1)], "method": "whole" | "items" | "gaps", "lines": n}`` ท่อนอยู่ในพิกัดหน้า หดขอบแล้ว
    ไม่มีบรรทัดเลย = คืนกล่องเดิมทั้งก้อน (ให้ขั้น export ตัดสินว่าว่างไหม)."""
    x0, y0, x1, y1 = box
    sub, runs = ink_runs(text, box, vlines)
    lines = [r for r in runs if r[1] - r[0] >= min_line_h]
    if not runs:
        return {"pieces": [(y0, y1)], "method": "whole", "lines": 0}
    top, bottom = max(y0, runs[0][0] - pad), min(y1, runs[-1][1] + pad)
    if not max_h or bottom - top <= max_h:
        return {"pieces": [(top, bottom)], "method": "whole", "lines": len(lines)}
    starts = set(item_starts(sub, box, lines, vlines))
    method = "items" if len(starts) >= 2 else "gaps"
    # จุดตัด = กลางช่องว่างระหว่างช่วงหมึกที่ติดกัน (ไม่ผ่าหมึกเตี้ย ๆ ที่อยู่ระหว่างบรรทัด) ถ้าใช้รายการ: เฉพาะช่องว่างก่อนบรรทัดที่เริ่มรายการ
    cands = [((a[1] + b[0]) // 2, b[0] - a[1]) for a, b in zip(runs, runs[1:]) if method == "gaps" or b[0] in starts]
    cuts, cur = [], top
    while bottom - cur > max_h:
        # ท่อนต้องสูงอย่างน้อยครึ่งเพดาน (แบบ tile_on_whitespace ของ preprocess_finance ที่หาจุดตัดไม่ต่ำกว่า max_h // 2)
        # ไม่งั้นช่องว่างใต้หัวตารางจะแยกหัวคอลัมน์ออกเป็นท่อนเล็ก ๆ (1317/1318: หัวไทยกับหัวอังกฤษถูกนับเป็นเลขข้อ)
        later = [c for c in cands if c[0] - cur >= max_h / 2]
        if not later:
            break
        fit = [c for c in later if c[0] - cur <= max_h]
        y = max(fit, key=lambda c: c[1])[0] if fit else later[0][0]  # ไม่มีจุดในเพดาน: ท่อนสูงเกินดีกว่าตัดกลางรายการ
        cuts.append(y)
        cur = y
    edges = [top, *cuts, bottom]
    pieces = []
    for a, b in zip(edges, edges[1:]):  # หดแต่ละท่อนตามบรรทัดของท่อนนั้น (ไม่พาช่องว่างยาวท้ายตารางติดไป)
        own = [r for r in runs if a <= r[0] and r[1] <= b]
        pieces.append((max(a, own[0][0] - pad), min(b, own[-1][1] + pad)) if own else (a, b))
    return {"pieces": pieces, "method": method, "lines": len(lines)}
