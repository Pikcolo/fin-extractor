"""ผังแบ่งหน้าเป็นส่วน (section) จากภาพทั้งหน้าที่ preprocess แล้ว — **ภาพล้วน ไม่เรียก OCR** (ขั้น 3 ของ flow: 03_sections/.../layout)

ย้ายมาจาก notebooks/preprocess_lab/labkit.py แบบคัดลอกตรงตัว (ผังที่ได้ต้องเท่าเดิมทุกไบต์) ค่าคงที่ที่นี่เป็น **ค่าทดลอง**
วัดบนสแกนจริง 18 ใบ ไม่มีเฉลย ดู docs/14-section-preprocess.md ก่อนอ้างตัวเลข
ทำงานบนสำเนาด้านยาว WORK px แล้วตัดภาพจริงจากภาพเต็มสีด้วย crop_full
"""

from __future__ import annotations

import cv2
import numpy as np

WORK = 1600  # ด้านยาวของสำเนาที่วิเคราะห์ (ตรงกับ preprocessing.min_long_side ใน config)


def resize_to_work(img: np.ndarray) -> tuple[np.ndarray, float]:
    scale = min(1.0, WORK / max(img.shape[:2]))
    return (cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img), scale


def line_masks(binary_black_on_white: np.ndarray, divisor: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """เส้นตารางแนวนอน/แนวตั้งด้วย morphological opening (ความยาว kernel = ด้านภาพ / ``divisor`` ค่าทดลอง)."""
    ink = 255 - binary_black_on_white
    h, w = ink.shape
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // divisor, 3), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // divisor, 3))))
    return horiz, vert


def text_ink(binary_black_on_white: np.ndarray, drop_lines: bool = True) -> np.ndarray:
    """หมึกสำหรับแบ่ง section: ตัดเส้นตารางยาว ๆ ออกก่อน (``drop_lines``)

    กรอบตารางของฟอร์มเป็นชิ้นเดียวต่อกันทั้งหน้า ถ้าไม่ตัดออก contour ภายนอกจะได้แค่ "กรอบทั้งหน้า" และเส้นตั้งที่พาดทุกแถว
    ทำให้ไม่มีแถวว่างให้ตัดแถบ (เจอจริงกับ 001307 รอบแรกของ lab)
    """
    ink = 255 - binary_black_on_white
    if not drop_lines:
        return ink
    horiz, vert = line_masks(binary_black_on_white)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(lines))


def cells_from_lines(binary_black_on_white: np.ndarray, min_ratio: float = 0.0002, max_ratio: float = 0.5) -> list[tuple[int, int, int, int]]:
    """ช่องของตารางจากเส้นแนวนอน+แนวตั้ง: ส่วนว่างที่ล้อมด้วยเส้น (ขนาดระหว่าง min_ratio–max_ratio ของพื้นที่หน้า ทดลอง)."""
    horiz, vert = line_masks(binary_black_on_white)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    free = cv2.bitwise_not(lines)
    n, _, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    area = lines.shape[0] * lines.shape[1]
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n) if min_ratio * area <= st[i, 4] <= max_ratio * area]


def table_regions(binary_black_on_white: np.ndarray, min_area_ratio: float = 0.02) -> list[tuple[int, int, int, int]]:
    """บริเวณตาราง = กลุ่มเส้นแนวนอน+แนวตั้งที่เชื่อมกัน มีทั้งสองทิศ และกรอบใหญ่ ≥ ``min_area_ratio`` ของหน้า (ทดลอง)."""
    horiz, vert = line_masks(binary_black_on_white)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    h_img, w_img = binary_black_on_white.shape
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i, :4])
        box = (slice(y, y + h), slice(x, x + w))
        if w * h >= min_area_ratio * h_img * w_img and horiz[box].any() and vert[box].any():
            out.append((x, y, w, h))
    return sorted(out, key=lambda b: (b[1], b[0]))


def free_boxes(tables: list[tuple[int, int, int, int]], w_img: int, h_img: int) -> list[tuple[int, int, int, int]]:
    """พื้นที่ของหน้าที่อยู่นอกตารางทั้งหมด เป็นกล่องสี่เหลี่ยม (x, y, w, h) — รวมข้อความ **ข้างตาราง** (ซ้าย/ขวา) ด้วย

    ตัดหน้าเป็นแถบแนวนอนที่ขอบบน/ล่างของตารางทุกก้อน แต่ละแถบเอาช่วงแนวนอนที่ไม่ถูกตารางบัง แล้วต่อแถบที่ติดกันและช่วงเดียวกันเข้าด้วยกัน
    (ไม่มีตารางเลย = ทั้งหน้าเป็นกล่องเดียว) ผู้เรียกตัดสินเองว่ากล่องไหนมีหมึก: นอกตารางไม่ได้แปลว่าไม่ใช่ข้อมูล
    """
    ys = sorted({0, h_img, *(min(max(v, 0), h_img) for t in tables for v in (t[1], t[1] + t[3]))})
    open_boxes: dict[tuple[int, int], list[int]] = {}
    out: list[tuple[int, int, int, int]] = []
    for y0, y1 in zip(ys, ys[1:]):
        spans = sorted((t[0], t[0] + t[2]) for t in tables if t[1] < y1 and t[1] + t[3] > y0)
        free, cur = [], 0
        for a, b in spans:
            if a > cur:
                free.append((cur, a))
            cur = max(cur, b)
        if cur < w_img:
            free.append((cur, w_img))
        carried: dict[tuple[int, int], list[int]] = {}
        for iv in free:
            if iv in open_boxes and open_boxes[iv][1] == y0:
                carried[iv] = open_boxes.pop(iv)
                carried[iv][1] = y1
            else:
                carried[iv] = [y0, y1]
        out += [(iv[0], r[0], iv[1] - iv[0], r[1] - r[0]) for iv, r in open_boxes.items()]
        open_boxes = carried
    out += [(iv[0], r[0], iv[1] - iv[0], r[1] - r[0]) for iv, r in open_boxes.items()]
    return out


FRAME_INSIDE = 0.9  # ส่วนของตารางในที่อยู่ในกรอบอย่างน้อยเท่านี้จึงนับว่า "อยู่ข้างใน" (ค่าทดลอง)


MAX_PIECE_FRAC = 0.3  # ส่วนที่สูงกว่านี้ (สัดส่วนของความสูงหน้า) ถูกตัดเป็นกลุ่มแถว วัดบน 9 ใบ (OpenThai luma, rules ล้วน ไม่มีเฉลย): 0.5 ขาด 9 ช่อง · 0.3 ขาด 7 (ดีสุด กู้วันที่ 1310, วันที่/เลขที่ 1320 แต่เสีย invoice_no 1317) · 0.2 ขาด 9 และ 1307 ตก failed (ท่อนเล็กเกิน)


def drop_frames(tables: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """ตัดตารางที่เป็น "กรอบครอบ" ออก: กรอบของฟอร์มทั้งใบเชื่อมเส้นทุกอย่างเป็นก้อนเดียว ถ้าเก็บไว้จะกลายเป็นทั้งหน้าซ้ำกับตารางข้างใน

    ตารางที่มีตารางอื่นอยู่ข้างในถือเป็นกรอบ (ข้อมูล 18 ใบ: ตารางซ้อนกัน 4 คู่ เป็นกรอบฟอร์มทั้งหมด พื้นที่ตารางในเป็น 0.73–0.89 ของกรอบ
    ไม่มีกรณีตารางจริงที่มีตารางย่อย — เอกสารชนิดอื่นอาจต่าง: ถ้ามี หัวคอลัมน์จะถูกแยกจากแถวภายในกรอบนั้น)
    ตารางที่ไม่มีตารางอื่นข้างในไม่ถูกแตะ ส่วนนอกตารางที่เหลือเป็นส่วนข้อความตามปกติ (``free_boxes``)
    """
    def inside(inner, outer) -> bool:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        w = min(ix + iw, ox + ow) - max(ix, ox)
        h = min(iy + ih, oy + oh) - max(iy, oy)
        return iw * ih < ow * oh and max(w, 0) * max(h, 0) >= FRAME_INSIDE * iw * ih

    return [t for t in tables if not any(u is not t and inside(u, t) for u in tables)]


def split_rows(ink: np.ndarray, box: tuple[int, int, int, int], max_h: float,
               empty_ratio: float = 0.002) -> list[tuple[int, int, int, int]]:
    """ตัดกล่องที่สูงกว่า ``max_h`` เป็นกลุ่มแถว (ท่อนติดกัน ไม่ซ้อนกัน) ที่ **แถวว่างระหว่างบรรทัด** ไม่ตัดกลางบรรทัด

    จำนวนท่อน = ceil(สูง / max_h) ตัดที่ช่องว่างที่ใกล้ตำแหน่งแบ่งเท่า ๆ กันที่สุด (ตรงกลางช่วงแถวว่าง) ``ink`` = หมึกของหน้า (ตัดเส้นตารางออกแล้ว)
    ช่วงแถวว่าง = แถวที่หมึก < ``empty_ratio`` ของความกว้าง (ค่าเดียวกับ ``bands_by_gaps``) ไม่เหลือช่องว่างให้ตัดแล้วก็หยุดตัด (ท่อนสูงกว่าเกณฑ์ได้ ไม่ตัดผ่ากลางบรรทัด)
    """
    x, y, w, h = box
    n = int(np.ceil(h / max_h)) if max_h > 0 else 1
    if n <= 1:
        return [box]
    empty = (ink[y:y + h, x:x + w] > 0).mean(axis=1) < empty_ratio
    runs, start = [], None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            runs.append((start + i) // 2)
            start = None
    if start is not None:
        runs.append((start + len(empty)) // 2)
    cuts: list[int] = []
    for k in range(1, n):
        ideal = h * k / n
        pool = [r for r in runs if (not cuts or r > cuts[-1]) and 0 < r < h]
        if not pool:  # ไม่มีช่องว่างเหลือให้ตัด: ปล่อยท่อนสูงกว่าเกณฑ์ ดีกว่าตัดผ่ากลางบรรทัด
            break
        cuts.append(int(min(pool, key=lambda r: abs(r - ideal))))
    edges = [0, *sorted(set(cuts)), h]
    return [(x, y + a, w, b - a) for a, b in zip(edges, edges[1:]) if b > a]


RULE_COVER = 0.8  # แถวพิกเซลที่เป็นเส้นแนวนอนอย่างน้อยเท่านี้ของความกว้างตารางจึงนับเป็น "เส้นแบ่งแถว" (ค่าทดลอง)


EMPTY_ROW = 0.002  # แถวที่หมึก < เท่านี้ของความกว้างหน้าถือว่าว่าง (ค่าเดียวกับ ``split_rows`` / ``bands_by_gaps``)


def table_rules(horiz: np.ndarray, table: tuple[int, int, int, int], min_cover: float = RULE_COVER) -> list[int]:
    """ตำแหน่ง y (พิกัดหน้า) ของเส้นแนวนอนยาวเต็มตาราง (เส้นแบ่งแถว) ``horiz`` = ผลของ ``line_masks``"""
    x, y, w, h = table
    cov = (horiz[y:y + h, x:x + w] > 0).mean(axis=1)
    groups: list[list[int]] = []
    for v in np.where(cov >= min_cover)[0]:
        if groups and v - groups[-1][-1] <= 3:
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    return [y + int(round(float(np.mean(g)))) for g in groups]


def band_cuts(ink: np.ndarray, candidates: set[int]) -> list[int]:
    """จุดตัดแนวนอนที่ใช้ได้จริง: ขอบหน้า + ตำแหน่งผู้สมัครที่ **แถวนั้นว่างตลอดความกว้างหน้า** (ไม่ตัดผ่านข้อความข้างตาราง)"""
    h_img = ink.shape[0]
    empty = (ink > 0).mean(axis=1) < EMPTY_ROW
    ok = {0, h_img} | {y for y in candidates if 0 < y < h_img and empty[y]}
    return sorted(ok)


def layout_sections(binary_black_on_white: np.ndarray, min_table_ratio: float = 0.02,
                    min_component_px: int = 8, table_mode: str = "rows",
                    max_piece_frac: float | None = MAX_PIECE_FRAC, fill_region: np.ndarray | None = None,
                    fill_text: np.ndarray | None = None) -> list[dict]:
    """แบ่งหน้าเป็นภาพย่อยตามตาราง แล้วเรียงตามลำดับอ่าน (บนลงล่าง ซ้ายไปขวา)

    - ``table_mode="rows"`` (ค่าตั้งต้น): **แถบแนวนอนเต็มความกว้างหน้า** ตัดที่ขอบตารางและเส้นแนวนอนยาวเต็มตาราง (``table_rules``) เฉพาะแถวที่ **ว่างตลอดความกว้างหน้า**
      (``band_cuts``) ไม่ตัดผ่านข้อความข้างตาราง ป้ายข้างตาราง (เช่น ป้ายยอดรวมทางซ้ายของกล่องตัวเลข) จึงอยู่ภาพเดียวกับค่าของแถวนั้น
      ``"bands"`` ตัดเฉพาะที่ขอบตาราง (วัดแล้วแย่กว่า: ป้าย–ค่าอยู่บรรทัดเดียวกัน 10/22 เทียบ rows 18/22)
      ผลข้างเคียงที่รู้: แถบที่มีแบนเนอร์ใหญ่กับกล่องเล็กอยู่ข้างกัน OpenThai อาจหลอน (1312) และแถบหัวคอลัมน์ถูกตัดแยกบาง ๆ จากแถวรายการ
    - ``table_mode="whole"`` เก็บ **ทั้งก้อนเป็นส่วนเดียว** หัวคอลัมน์กับค่าในแถวจึงอยู่ภาพเดียวกัน (ป้ายข้างตารางเป็นอีกภาพ)
      ตัวอ่านที่รู้จักตารางคืนโครงสร้างแถว–คอลัมน์ให้ ไม่ต้องประกอบกลับจากตำแหน่ง
      กรอบของฟอร์มที่ครอบตารางอื่นถูกตัดออกก่อน (``drop_frames``) ไม่งั้นกลายเป็นทั้งหน้า
      - ส่วนที่สูงกว่า ``max_piece_frac`` ของหน้า (ตารางหรือข้อความ) ถูกตัดเป็นกลุ่มแถวที่แถวว่างระหว่างบรรทัด (``split_rows``) แต่ละท่อนมี
      ``part``/``parts`` บอกลำดับ ``None`` = ไม่ตัด ท่อนหลัง ๆ ไม่มีหัวคอลัมน์ของตารางเดิมซ้ำ (ยังไม่ได้ทำ)
    ``table_mode="cells"`` แตกเป็น **ช่อง (cell)** ตามเส้น (ไม่เจอช่องเลยเก็บทั้งตารางเป็นส่วนเดียว) ช่องแยกหัวออกจากค่า
    - นอกตาราง → **ส่วนข้อความ (text)** ทุกบริเวณที่ไม่ถูกตารางบัง รวมข้อความ **ข้างตาราง** ด้วย (``free_boxes``) ไม่ใช่แค่แถบคั่นระหว่างตาราง
      เพราะป้ายของช่องยอดรวมมักอยู่ซ้ายของตารางตัวเลข ส่วนที่ไม่มีหมึกเลยจะติดธง ``keep=False``
    - ส่วนที่ **ไม่มีชิ้นหมึกเลย** (นับหลังตัดเส้นตารางออก ไม่นับชิ้นเล็กกว่า ``min_component_px`` พิกเซลที่เป็นจุดรบกวน)
      ติดธง ``keep=False`` (ตัดทิ้ง) ใช้จำนวนชิ้น ไม่ใช้สัดส่วนหมึกต่อพื้นที่ เพราะช่องสูงยาวที่มีตัวเลขสองตัว (เช่น คอลัมน์จำนวน)
      จะดูเหมือนว่างและถูกตัดทิ้งผิด (เจอจริงกับ 1309 รอบแรก)
    ค่า ``min_table_ratio`` / ``min_component_px`` เป็นค่าทดลอง ดูตารางสรุปประกอบภาพก่อนเชื่อ
    """
    if table_mode not in ("whole", "cells", "bands", "rows"):
        raise ValueError(f"table_mode ต้องเป็น 'whole', 'cells', 'bands' หรือ 'rows': {table_mode!r}")
    h_img, w_img = binary_black_on_white.shape
    ink = text_ink(binary_black_on_white)
    if fill_text is not None:  # ตัวอักษรบนพื้นสี (images.fill_masks) นับเป็นหมึกด้วย: แถบที่มีแต่ตัวอักษรบนพื้นสีไม่ถูกทิ้งเป็นว่าง
        ink = cv2.bitwise_or(ink, fill_text)
    cut_ink = cv2.bitwise_or(ink, fill_region) if fill_region is not None else ink  # แถวที่อยู่ในพื้นสีไม่ใช่ "แถวว่าง" จึงไม่ตัดผ่าแถบสี
    tables = table_regions(binary_black_on_white, min_table_ratio)
    if table_mode != "cells":
        tables = drop_frames(tables)  # โหมดช่องแตกกรอบเป็นช่องอยู่แล้ว
    secs: list[dict] = []

    def add(kind: str, box: tuple[int, int, int, int], table: int | None, part: tuple[int, int] | None = None) -> None:
        pieces = split_rows(cut_ink, box, max_piece_frac * h_img) if max_piece_frac and part is None else [box]
        if len(pieces) > 1:
            for k, piece in enumerate(pieces, start=1):
                add(kind, piece, table, (k, len(pieces)))
            return
        x, y, w, h = box
        crop = ink[y:y + h, x:x + w]
        ratio = float((crop > 0).mean()) if crop.size else 0.0
        n, _, st, _ = cv2.connectedComponentsWithStats(crop, connectivity=8) if crop.size else (1, None, np.zeros((1, 5), int), None)
        comps = int((st[1:, cv2.CC_STAT_AREA] >= min_component_px).sum())
        secs.append({"kind": kind, "bbox": [x, y, w, h], "table": table, "ink_ratio": ratio, "components": comps, "keep": comps > 0,
                     **({"part": part[0], "parts": part[1]} if part else {})})

    if table_mode in ("bands", "rows"):
        # แถบแนวนอนเต็มความกว้าง: ป้ายข้างตารางอยู่ภาพเดียวกับตัวเลขของแถวนั้นเสมอ (ไม่แยกซ้าย/ขวาเป็นคนละภาพ)
        cands: set[int] = set()
        horiz = line_masks(binary_black_on_white)[0] if table_mode == "rows" else None
        for tx, ty, tw, th in tables:
            cands |= {ty, ty + th}
            if horiz is not None:
                cands |= set(table_rules(horiz, (tx, ty, tw, th)))
        cuts = band_cuts(cut_ink, cands)
        for y0, y1 in zip(cuts, cuts[1:]):
            over = [max(0, min(y1, ty + th) - max(y0, ty)) for tx, ty, tw, th in tables]
            best = int(np.argmax(over)) if over else None
            is_table = best is not None and over[best] >= 0.5 * (y1 - y0)
            add("table" if is_table else "text", (0, y0, w_img, y1 - y0), best if is_table else None)
        tables = []  # ใช้หมดแล้ว ข้ามเส้นทางตารางก้อน/ช่อง และ free_boxes (แถบคลุมทั้งหน้าอยู่แล้ว)
        free = []
    else:
        free = free_boxes([t for t in tables], w_img, h_img)
    for t, (tx, ty, tw, th) in enumerate(tables):
        cells = cells_from_lines(binary_black_on_white[ty:ty + th, tx:tx + tw]) if table_mode == "cells" else []
        for cx, cy, cw, ch in cells:
            add("cell", (tx + cx, ty + cy, cw, ch), t)
        if not cells:
            add("table", (tx, ty, tw, th), t)
    for x, y, w, h in free:
        add("text", (x, y, w, h), None)
    secs.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    for i, s in enumerate(secs):
        s["id"] = f"s{i:02d}"
    return secs


def draw_sections(img: np.ndarray, secs: list[dict]) -> np.ndarray:
    """เขียว = ช่องตารางที่เก็บ, ส้ม = ข้อความนอกตาราง, น้ำเงิน = ตารางทั้งก้อน, แดงหนา = ว่าง (ตัดทิ้ง)."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = {"cell": (0, 160, 0), "text": (0, 140, 255), "table": (200, 90, 0)}
    for s in secs:
        x, y, w, h = s["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), colors[s["kind"]] if s["keep"] else (0, 0, 220), 2 if s["keep"] else 4)
        cv2.putText(out, s["id"][1:], (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 0, 90), 1, cv2.LINE_AA)
    return out


def crop_full(full: np.ndarray, work: np.ndarray, sec: dict) -> np.ndarray:
    """ตัดส่วนนั้นจากภาพเต็ม (พิกัดของ section อยู่บนสำเนาทำงาน จึงคูณสัดส่วนกลับ)."""
    fx, fy = full.shape[1] / work.shape[1], full.shape[0] / work.shape[0]
    x, y, w, h = sec["bbox"]
    return full[int(y * fy):int((y + h) * fy), int(x * fx):int((x + w) * fx)]
