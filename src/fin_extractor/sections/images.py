"""ภาพของแต่ละส่วนก่อนเข้า OCR (ขั้น 3 ของ flow: 03_sections/.../images/<วิธี>) — luma | flat | bin

ย้ายมาจาก notebooks/preprocess_lab/labkit.py แบบคัดลอกตรงตัว ค่าคงที่ (DOT_MAX_H MIN_GLYPH_* BIN_FLOOR ฯลฯ)
วัดจากไม่กี่ใบ ไม่ใช่ค่าที่พิสูจน์แล้ว ดู docs/13-section-preprocess.md
"""

from __future__ import annotations

import cv2
import numpy as np

VARIANTS = ("luma", "flat", "bin")  # วิธีเตรียมภาพรายส่วน (ที่เดียว: cli/run_sections/เอกสารอ้างจากที่นี่)


def gray_value_channel(crop: np.ndarray) -> np.ndarray:
    """เทาแบบ "ช่องสว่างสุด" (max ของ B,G,R = ช่อง V ของ HSV) แทนสูตร luma

    หมึกดำมืดทุกช่อง จึงยังมืด ส่วนสีที่ไม่ใช่ข้อความ (ลายน้ำฟ้า, ตราประทับแดง/น้ำเงิน, แถบสี, เส้นตารางสี) จะมีอย่างน้อยหนึ่งช่องสว่าง
    เมื่อเอาค่าสว่างสุดมา จึงกลายเป็นเกือบขาว (luma จะถ่วงสีให้เทาเข้มและยังเห็นอยู่)
    """
    return crop.max(axis=2) if crop.ndim == 3 else crop


def flatten_background(gray: np.ndarray, frac: float = 0.04) -> np.ndarray:
    """ทำพื้นหลังให้ขาวเรียบ: หารด้วยภาพพื้นหลังที่ประมาณจาก morphological closing (ลบเงา/สีพื้น) แล้วยืดคอนทราสต์
    ``frac`` = ขนาด kernel ต่อด้านสั้นของภาพ (ทดลอง ต้องใหญ่กว่าตัวอักษรเพื่อให้ closing ลบตัวอักษรออกจากภาพพื้นหลัง)."""
    k = max(15, int(min(gray.shape[:2]) * frac) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    norm = cv2.divide(gray, bg, scale=255)
    lo = np.percentile(norm, 1)
    return np.clip((norm.astype(np.float32) - lo) * (255.0 / max(255.0 - lo, 1.0)), 0, 255).astype(np.uint8)


BG_SCALE = 4  # ย่อภาพก่อนประมาณสีพื้น (ประหยัดเวลา median) ค่าทดลอง


BG_KERNEL = 31  # หน้าต่าง median บนภาพย่อ (= 124 px บนภาพจริง) ต้องกว้างกว่าเส้นหนาสุดของตัวอักษรหัวเรื่อง (~30 px)


BIN_FLOOR = 28  # ระยะสีขั้นต่ำ (Lab) ที่ถือว่าต่างจากพื้น กันรอยยับของกระดาษเมื่อ Otsu ได้ค่าต่ำ ค่าทดลอง (Otsu จริงบน 1310 ได้ 86–102)


MIN_RULE_H = 80  # เส้นตั้งของตารางต้องยาวอย่างน้อยเท่านี้ (พิกเซล ≈ 3 บรรทัด ค่าทดลอง) บน 1307 ความสูงแถว ≈ 50 px


MIN_RULES = 4  # ช่วงแนวตั้งที่นับเป็น "ตารางมีช่องค่า" ต้องมีเส้นตั้งพร้อมกันอย่างน้อยเท่านี้ (ซ้าย, ป้าย|ค่า, คอลัมน์ค่าอีก ≥1, ขวา) กันกรอบของกล่องข้อความ


MAX_GLYPH_H = 40  # ชิ้นหมึกที่สูงกว่านี้ไม่ใช่ตัวอักษร (เศษเส้นตั้งที่เอียงจนไม่ถูกจับเป็นเส้น) วัดบน 1307: ตัวอักษรสูงสุด 30 px เศษเส้น 56–181 px


MIN_GLYPH_AREA = 60  # ชิ้นหมึกที่เล็กกว่านี้ (พิกเซล²) หรือบางกว่า ``MIN_GLYPH_W`` คือเศษเส้นตาราง วัดบน 1307: เศษเส้น กว้าง 1–2 px หรือ 44 px² ส่วนสระ/วรรณยุกต์เล็กสุด 86 px²


MIN_GLYPH_W = 3


DOT_MAX_H = 10  # ชิ้นหมึกที่เตี้ยกว่านี้ (พิกเซลของภาพ section) = จุดไข่ปลา/จุดทศนิยม ไม่นับเป็นข้อความ วัดบน 1307: จุด 4–5 px ตัวอักษร p75 = 21 px


def binarize_inverted(crop: np.ndarray) -> np.ndarray:
    """ขาว-ดำ แบบ **ตัวอักษรขาว กระดาษดำ** ที่ไม่ขึ้นกับสีพื้น/สีตัวอักษร

    ตัดสินจากระยะสี (Lab) ของแต่ละพิกเซลจากสีพื้นรอบข้าง (median หน้าต่างใหญ่) ไม่ใช่จากความสว่าง: ตัวดำบนกระดาษขาว ตัวขาวบนแถบสีเขียว/น้ำเงิน
    และตัวสีบนพื้นสี ได้ผลแบบเดียวกัน (วัดดูด้วยตาบน 1306 s07/s20, 1310, 1312 s02, 1317 s03, 1318 s03, 1319 s02) จุดอ่อน: ขอบของแถบสีที่เป็นมุมโค้งเกิดรอยเส้นบาง ๆ
    """
    dist, _bg = _bg_distance(crop)
    otsu, _ = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((dist > max(otsu, BIN_FLOOR)) * 255).astype(np.uint8)


def _bg_distance(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ระยะสี Lab ของแต่ละพิกเซลจากสีพื้นรอบข้าง, ภาพสีพื้น Lab) สีพื้น = median หน้าต่างใหญ่บนภาพย่อ."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    small = cv2.resize(lab, None, fx=1 / BG_SCALE, fy=1 / BG_SCALE, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, BG_KERNEL), (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.linalg.norm(lab.astype(np.float32) - bg.astype(np.float32), axis=2), bg


def fill_masks(page: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(พื้นสี, ตัวอักษรบนพื้นสี) ของภาพทั้งหน้า เป็นหน้ากาก 255/0 ที่ผังแบ่ง section ใช้ร่วมกับหมึกดำบนกระดาษ
    ผังเดิมนับเฉพาะหมึกเข้มบนพื้นสว่าง ตัวขาวบนแถบสีจึงดูเป็นพื้นว่างและถูกตัดผ่ากลางได้ (1306 s01/s02: แถบเทา "Receipt" ถูกตัดเหลือเสี้ยวให้ OCR
    แล้วโมเดลเดาข้อความเอง) พื้นสี = ที่สีพื้นรอบข้าง (``_bg_distance``) ต่างจากสีกระดาษของหน้า (median ของสีพื้นทั้งหน้า) เกิน ``BIN_FLOOR``
    กัดขอบเข้า ``MIN_GLYPH_W`` px กันรอยเส้นตามขอบแถบ ส่วนบนกระดาษ (พื้นปกติ) ไม่นับ เพื่อไม่ให้รอยยับของกระดาษกลายเป็นหมึก
    ตัวอักษรบนพื้นสีเท่านั้นที่ตามไปนับเป็นหมึก ส่วนพื้นสีทั้งผืนใช้ห้ามตัดผ่า"""
    dist, bg = _bg_distance(page)
    otsu, _ = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = dist > max(otsu, BIN_FLOOR)
    paper = np.median(bg.reshape(-1, 3), axis=0)
    fill = np.linalg.norm(bg.astype(np.float32) - paper.astype(np.float32), axis=2) > BIN_FLOOR
    fill = cv2.erode(fill.astype(np.uint8), np.ones((2 * MIN_GLYPH_W + 1,) * 2, np.uint8)) > 0
    return (fill * 255).astype(np.uint8), ((text & fill) * 255).astype(np.uint8)


def _runs(mask_1d: np.ndarray, gap: int = 0) -> list[tuple[int, int]]:
    """ช่วง [a, b) ที่ mask เป็นจริงต่อเนื่อง (ช่องว่างสั้นกว่าหรือเท่ากับ ``gap`` ถือว่าต่อกัน)."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def table_columns(b: np.ndarray) -> dict | None:
    """โครงคอลัมน์ของตารางในภาพ ``binarize_inverted`` จากเส้นตั้ง: ``ruled`` = ช่วงแนวตั้งที่มีเส้นตั้งพร้อมกัน ≥ ``MIN_RULES`` เส้น
    ``boundary`` = ขอบขวาของเซลล์ที่กว้างที่สุด (= คอลัมน์ป้าย) ขวามือคือช่องค่า ไม่เจอตาราง = None."""
    h, _w = b.shape
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, MIN_RULE_H)))
    ruled = np.array([len(_runs(vert[y] > 0, gap=3)) >= MIN_RULES for y in range(h)])
    if not ruled.any():
        return None
    y0, y1 = int(np.flatnonzero(ruled)[0]), int(np.flatnonzero(ruled)[-1]) + 1
    xs = [int(round((a + z - 1) / 2)) for a, z in _runs((vert[y0:y1] > 0).any(axis=0), gap=3)]
    widest = int(np.argmax(np.diff(xs)))
    return {"ruled_y": [y0, y1], "boundary": xs[widest + 1], "left": xs[widest]}


def drop_empty_rows(b: np.ndarray, boundary: int, left: int = 0, span: tuple[int, int] | None = None) -> tuple[np.ndarray, list[list[int]]]:
    """ทาดำแถวที่ **ช่องค่า (ขวาของ ``boundary``) ไม่มีตัวอักษร** เหลือแถวที่มีค่ากับบรรทัดป้ายที่ซ้อนทับ ภายในช่วงแนวตั้ง ``span`` (ค่าตั้งต้น = ทั้งภาพ)
    ตัวอักษรคือชิ้นหมึกที่ไม่ใช่จุดไข่ปลา/เศษเส้น (ดู ``DOT_MAX_H`` ``MAX_GLYPH_H`` ``MIN_GLYPH_*``) ใช้ภาพล้วน ไม่ใช้ OCR
    คืน (ภาพใหม่ รูปทรงเดิม, ช่วงแนวตั้ง [a, b] ที่เก็บไว้)."""
    h, w = b.shape
    y0, y1 = span or (0, h)
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, MIN_RULE_H)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_and(b, cv2.bitwise_not(cv2.dilate(vert, np.ones((1, 7), np.uint8)))))
    text = np.zeros_like(b)
    for i in range(1, n):
        _x, _y, cw, ch, area = st[i]
        if DOT_MAX_H <= ch <= MAX_GLYPH_H and MIN_GLYPH_W <= cw < w // 3 and area >= MIN_GLYPH_AREA:
            text[lab == i] = 255
    val = (text[:, boundary + 3:] > 0).any(axis=1)
    lbl = (text[:, left + 3:boundary - 3] > 0).any(axis=1)
    keep = np.zeros(h, bool)
    for a, z in _runs(val, gap=DOT_MAX_H):
        keep[a:z] = True
        for la, lz in _runs(lbl, gap=DOT_MAX_H // 2):
            if la < z and lz > a:
                keep[la:lz] = True
    keep = cv2.dilate(keep.astype(np.uint8)[:, None], np.ones((2 * DOT_MAX_H + 1, 1), np.uint8))[:, 0] > 0
    out = b.copy()
    drop = ~keep
    drop[:y0] = False
    drop[y1:] = False
    out[drop] = 0
    return out, [list(r) for r in _runs(keep[y0:y1])]


def drop_empty_rows_by_table(items: list[tuple[dict, np.ndarray]]) -> list[tuple[np.ndarray, dict]]:
    """ใช้ ``drop_empty_rows`` กับทุกส่วนของตารางเดียวกัน (``sec['table']`` เท่ากัน) ด้วยคอลัมน์ป้ายชุดเดียวกัน
    ส่วนที่ถูกตัดจากตารางมักเห็นเส้นตั้งไม่ครบ (แถวยอดรวมที่ช่องรวมกัน) จึงยืมขอบคอลัมน์จากส่วนที่เห็นโครงตารางชัดสุด (ช่วงเส้นตั้งยาวสุด)
    ส่วนข้อความ (ไม่มีตาราง) ไม่แตะ ตารางที่ไม่มีค่าเลยสักส่วนก็ไม่แตะ (ไม่ใช่ฟอร์มที่เข้าใจ) ภาพที่ทาดำหมดคืน info['blank'] = True."""
    out = [(b, {"applied": False}) for _s, b in items]
    for table in {s["table"] for s, _ in items if s["table"] is not None}:
        members = [i for i, (s, _b) in enumerate(items) if s["table"] == table]
        cols = {i: table_columns(items[i][1]) for i in members}
        found = {i: c for i, c in cols.items() if c}
        if not found:
            continue
        ref = max(found.values(), key=lambda c: c["ruled_y"][1] - c["ruled_y"][0])
        done = {}
        for i in members:
            span = tuple(cols[i]["ruled_y"]) if cols[i] else None
            done[i] = drop_empty_rows(items[i][1], ref["boundary"], ref["left"], span)
        if not any(kept for _o, kept in done.values()):
            continue
        for i, (img, kept) in done.items():
            out[i] = (img, {"applied": True, "label_right_x": ref["boundary"], "kept_rows": kept, "blank": not img.any()})
    return out
