"""Router ก่อน OCR: คัดภาพย่อย (section) ที่เกี่ยวกับข้อมูลการเงินด้วย "ป้าย" และ "ค่า" ไม่ใช้ตำแหน่งตายตัว

เอกสารอินพุตไม่มีแพทเทิร์น ตำแหน่งบอกอะไรไม่ได้ แต่เนื้อหาบอกได้ว่าส่วนไหนเป็นข้อมูลการเงิน ป้ายและรูปแบบค่า **ใช้ชุดเดียวกับ rules สกัด**
(``extraction/rule_extractor.py``, ``extraction/patterns.py``) และ alias ใน config (``core_labels``, ``document_types``) ไม่คิดคำค้นเพิ่ม
ผู้เรียกส่งข้อความที่ตัวอ่านคัดกรองอ่านได้ของแต่ละส่วนเข้ามา (ตัวนี้ไม่เรียก OCR เอง จึงทดสอบได้โดยไม่ใช้ GPU)

ขั้นโครงสร้าง (ก่อนดูเนื้อหา — ตัวแบ่ง section สร้างของซ้ำซ้อนให้ ดู docs)
- **ซ้ำ**: ส่วนที่ bbox ซ้อนกับส่วนก่อนหน้า ≥ ``DUP_IOU`` → ตัดทิ้ง (ใน 18 ไฟล์ซ้ำ ~19%)
- **กล่องครอบ**: ส่วนที่มีส่วนย่อยอื่นอยู่ข้างในอย่างน้อย ``CONTAINER_MIN`` ส่วน → เก็บเป็น **ส่วนที่เหลือ (residual)** คือลบส่วนย่อยออกจากภาพ
  เหลือเฉพาะเนื้อหานอกส่วนย่อย (ผู้เรียกทำภาพ: scripts/make_ocr_sections.py) ไม่ตัดทิ้งทั้งก้อน เพราะมีของนอกส่วนย่อย เช่น เลขที่หนังสือรับรอง
  ของ 50 ทวิ และป้าย "รวมเงินที่จ่ายและภาษีที่หักนำส่ง" ซึ่งอยู่ซ้ายของช่องตัวเลข (ตอนตัดทิ้งทั้งก้อนของสองอย่างนี้หายไป)

ขั้นเนื้อหา (กับส่วนที่เหลือ ข้อแรกที่ตรงเป็นคำตอบ) **ตัดทิ้งเฉพาะข้อความร้อยแก้วยาวในภาพใหญ่ที่ไม่มีค่าและสกัดฟิลด์ไม่ได้**
1. มี **ค่าการเงิน** (ยอดเงิน, วันที่, เลขภาษี 13 หลัก, %) → เก็บ (data) รวมถึง **เศษของตัวเลข**: ส่วนสั้น (≤ ``LABEL_MAX``) ที่มีเลข ≥ 2 หลัก
   เพราะบางฟอร์มแบ่งช่องบาท/สตางค์ (50 ทวิ: "60" กับ ".00" คนละช่อง) ตัวตรวจยอดเงินที่ต้องเห็นทศนิยมในที่เดียวจึงมองไม่เห็น
2. **rules สกัดฟิลด์ได้ ≥ 1 ฟิลด์** จากข้อความของส่วนนี้ (ผู้เรียกส่ง ``contrib``) → เก็บ (contrib) ตัวสกัดจริงเป็นผู้ตัดสินว่าส่วนนี้ช่วยอะไร
3. **ข้อความสั้น** (≤ ``LABEL_MAX`` ตัวอักษร ไม่ว่ามีป้ายหรือไม่) → เก็บ (short) เสียเวลา OCR นิดเดียวแต่เสี่ยงทิ้งชื่อ/ป้าย/ค่าเล็ก ๆ
   (เจอจริง: ชื่อลูกค้า "อลกค้า กฤตวิทย์ ..." ที่ตัวอ่านอ่านป้ายเพี้ยน และหัวคอลัมน์ "จำนวนเงินที่จ่าย" ถูกตัดตอนใช้กฎเข้มกว่านี้)
4. **ขึ้นต้นด้วยป้ายที่อยู่** (``ADDRESS_LED``: ฟิลด์ address ใน schema ของ vendor/buyer) → เก็บ (label)
5. **ภาพเล็ก** (พื้นที่ ≤ ``SMALL_AREA``) → เก็บ (small) ภาพขนาดนั้นบรรจุข้อความยาวจริงไม่ได้ ถ้าตัวอ่านคัดกรองตอบข้อความยาว
   มักเป็นการอ่านเพี้ยนของช่องตัวเลข/ลายมือ ตัดทิ้งเสี่ยงค่าจริงหาย
6. **คอลัมน์ใต้หัวตารางรายการ** (หัวสั้นที่จับป้าย item_header ได้) → เก็บ (companion) รายการสินค้ามักไม่มีตัวเลขทั้งคอลัมน์
7. นอกนั้น → ตัดทิ้ง พร้อมเหตุผล (ข้อความยาวที่ไม่มีค่า/ฟิลด์ เช่น รายการประเภทเงินได้ เงื่อนไขท้ายใบ คำเตือน หมายเหตุท้ายฟอร์ม)

เลือกเก็บเกินดีกว่าทิ้งเกิน: ทิ้งผิดทำให้ข้อมูลหาย เก็บเกินแค่เสียเวลา OCR
ค่าทดลอง (ยังไม่ได้วัดกับเฉลย): ``OVERLAP``, ``DUP_IOU``, ``CONTAINER_MIN``, ``LABEL_MAX``, ``SMALL_AREA``
``LABEL_MAX`` = 64 อยู่ในช่องว่างระหว่างป้ายจริงที่ยาวที่สุด (62 ตัวอักษร) กับส่วนไม่เกี่ยวข้องที่สั้นที่สุดที่จับป้ายได้ (67) ใน 18 ไฟล์ (ช่องว่างแคบมาก)
``SMALL_AREA`` = 25,000 (พิกเซล² ที่ขนาดสำเนาทำงาน ด้านยาว 1600) อยู่ระหว่างช่องที่อ่านเพี้ยน (6–12k) กับข้อความเตือนจริงที่เล็กที่สุด (46k) ใน 18 ไฟล์
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..doctypes import squash
from ..extraction import markdown_to_lines
from ..extraction import rule_extractor as R
from ..extraction.patterns import AMOUNT_RE, PERCENT_RE, find_dates, find_tax_ids

OVERLAP = 0.5  # แถว/คอลัมน์ซ้อนกันอย่างน้อยกี่ส่วนของด้านที่สั้นกว่า จึงถือว่าอยู่แนวเดียวกัน
DUP_IOU = 0.9  # bbox ซ้อนกันเกินนี้ถือว่าเป็นส่วนเดียวกัน
CONTAINER_MIN = 2  # มีส่วนย่อยอยู่ข้างในอย่างน้อยกี่ส่วนจึงเป็น "กล่องครอบ"
LABEL_MAX = 64  # ข้อความสั้นกว่านี้ (ตัวอักษรไม่นับช่องว่าง) เก็บเสมอ ดูหัวไฟล์
SMALL_AREA = 25_000  # ภาพเล็กกว่านี้ (พิกเซล² ของ bbox ที่ขนาดสำเนาทำงาน) เก็บเสมอ ดูหัวไฟล์
ADDRESS_LED = re.compile(r"^\s*(?:ที่อยู่|address)", re.I)  # ฟิลด์ address ใน schema ของ vendor/buyer

# ป้ายจาก rules สกัด (ชื่อ → regex) AMOUNT_FIELDS เป็นลิสต์ (ชื่อ, regex) อยู่แล้ว
LABELS: list[tuple[str, Any]] = [
    *R.AMOUNT_FIELDS,
    ("date", R.DATE_LABEL_RE),
    ("invoice_no", R.INVOICE_NO_RE),
    ("buyer", R.BUYER_RE),
    ("company", R.COMPANY_RE),
    ("item_header", R.ITEM_HEADER_RE),
    ("remark", R.ITEM_END_RE),
]


def build_aliases(cfg: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """alias ของฟิลด์จาก config: core_labels ระดับบนและระดับชนิดเอกสาร + ฟิลด์ประจำชนิดทุกตัว."""
    out: list[tuple[str, list[str]]] = []
    for c in cfg.get("core_labels") or []:
        out.append((c["key"], list(c.get("aliases", []))))
    for name, p in (cfg.get("document_types") or {}).items():
        for c in p.get("core_labels") or []:
            out.append((f"{name}:{c['key']}", list(c.get("aliases", []))))
        for f in p.get("fields") or []:
            out.append((f"{name}:{f['key']}", list(f.get("aliases", []))))
    return out


def plain(text: str) -> str:
    """ข้อความของส่วนหนึ่งแบบไม่มี HTML ของตาราง (ตัวอ่านตอบตารางเป็น <table>) ต่อเป็นบรรทัดเดียว."""
    return " ".join(markdown_to_lines(text or ""))


def signals(text: str, aliases: list[tuple[str, list[str]]] | None = None) -> dict[str, Any]:
    """ป้ายและค่าการเงินที่พบในข้อความของส่วนหนึ่ง."""
    text = plain(text)
    labels = [k for k, rx in LABELS if rx.search(text)]
    sq = squash(text)
    labels += [f"alias:{k}" for k, als in (aliases or []) if any(a and squash(a) in sq for a in als)]
    values = []
    chars = len("".join(text.split()))
    if AMOUNT_RE.search(text):
        values.append("amount")
    if find_dates(text):
        values.append("date")
    if find_tax_ids(text):
        values.append("tax_id")
    if PERCENT_RE.search(text):
        values.append("percent")
    if not values and chars <= LABEL_MAX and re.search(r"\d{2,}", text):
        values.append("number")  # เศษของตัวเลข (ดูหัวไฟล์): เฉพาะส่วนสั้น ข้อความยาวที่มีเลขปนเป็นแค่ประโยค
    return {"labels": labels, "values": values, "chars": chars}


@dataclass
class Decision:
    id: str
    keep: bool
    role: str  # data | contrib | short | label | small | companion | residual | drop | duplicate
    reason: str
    labels: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(min(a1 - a0, b1 - b0), 1e-9)


def _inter(a: list, b: list) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def _iou(a: list, b: list) -> float:
    i = _inter(a, b)
    return i / max(a[2] * a[3] + b[2] * b[3] - i, 1e-9)


def structure(sections: list[dict]) -> dict[str, Decision]:
    """ส่วนซ้ำ (ตัดทิ้ง) และกล่องครอบ (เก็บเป็น residual) ก่อนดูเนื้อหา คืนเฉพาะส่วนที่เข้าข่ายสองแบบนี้."""
    out: dict[str, Decision] = {}
    kept: list[dict] = []
    for s in sections:
        twin = next((k for k in kept if _iou(s["bbox"], k["bbox"]) >= DUP_IOU), None)
        if twin is not None:
            out[s["id"]] = Decision(s["id"], False, "duplicate", f"ซ้ำกับ {twin['id']} (bbox ซ้อนกัน ≥ {DUP_IOU:.0%})")
        else:
            kept.append(s)
    for c in kept:
        area_c = c["bbox"][2] * c["bbox"][3]
        inside = [d for d in kept if d is not c and d["bbox"][2] * d["bbox"][3] < area_c
                  and _inter(c["bbox"], d["bbox"]) >= 0.9 * d["bbox"][2] * d["bbox"][3]]
        if len(inside) >= CONTAINER_MIN:
            out[c["id"]] = Decision(c["id"], True, "residual", f"กล่องครอบ {len(inside)} ส่วน: เก็บเฉพาะส่วนที่เหลือนอกส่วนย่อย (ลบส่วนย่อยออกจากภาพ)")
    return out


def route(sections: list[dict], texts: dict[str, str], aliases: list[tuple[str, list[str]]] | None = None,
          page_size: tuple[float, float] | None = None, contrib: dict[str, list[str]] | None = None) -> list[Decision]:
    """``sections`` = รายการ {id, kind, bbox=[x,y,w,h], table} ตามลำดับอ่าน, ``texts`` = ข้อความที่ตัวอ่านคัดกรองอ่านได้ของแต่ละ id,
    ``contrib`` = ฟิลด์ที่ rules สกัดได้จากข้อความของแต่ละ id (ว่าง/ไม่ส่ง = ไม่ใช้กฎข้อ 2)
    ``page_size`` = (กว้าง, สูง) ของหน้าในหน่วยเดียวกับ bbox (ไม่ใส่ = ใช้ขอบของทุกส่วนรวมกัน)."""
    contrib = contrib or {}
    skipped = structure(sections)
    active = [s for s in sections if s["id"] not in skipped]
    if page_size is None and sections:
        page_size = (max(s["bbox"][0] + s["bbox"][2] for s in sections), max(s["bbox"][1] + s["bbox"][3] for s in sections))
    sig = {s["id"]: signals(texts.get(s["id"], ""), aliases) for s in active}
    out: dict[str, Decision] = dict(skipped)
    for s in active:
        g = sig[s["id"]]
        area = s["bbox"][2] * s["bbox"][3]
        if g["values"]:
            out[s["id"]] = Decision(s["id"], True, "data", "มีค่าการเงิน: " + ", ".join(g["values"]), g["labels"], g["values"])
        elif contrib.get(s["id"]):
            out[s["id"]] = Decision(s["id"], True, "contrib", "rules สกัดได้: " + ", ".join(contrib[s["id"]]), g["labels"], g["values"])
        elif g["chars"] <= LABEL_MAX:
            out[s["id"]] = Decision(s["id"], True, "short", f"ข้อความสั้น ({g['chars']} ตัวอักษร)", g["labels"], g["values"])
        elif ADDRESS_LED.search(plain(texts.get(s["id"], ""))):
            out[s["id"]] = Decision(s["id"], True, "label", "ขึ้นต้นด้วยป้ายที่อยู่", g["labels"], g["values"])
        elif area <= SMALL_AREA:
            out[s["id"]] = Decision(s["id"], True, "small", f"ภาพเล็ก ({area:,.0f} px²) ข้อความยาวน่าจะอ่านเพี้ยน", g["labels"], g["values"])

    for s in active:  # 6: คอลัมน์ใต้หัวตารางรายการ
        if "item_header" not in sig[s["id"]]["labels"] or sig[s["id"]]["chars"] > LABEL_MAX or s.get("table") is None:
            continue
        header_row = [h for h in active if h.get("table") == s["table"]
                      and _overlap(s["bbox"][1], s["bbox"][1] + s["bbox"][3], h["bbox"][1], h["bbox"][1] + h["bbox"][3]) >= OVERLAP]
        for h in header_row:
            hx, hy, hw, hh = h["bbox"]
            for c in active:
                if c["id"] not in out and c.get("table") == s["table"] and c["bbox"][1] >= hy + hh - 0.1 * hh                         and _overlap(hx, hx + hw, c["bbox"][0], c["bbox"][0] + c["bbox"][2]) >= OVERLAP:
                    out[c["id"]] = Decision(c["id"], True, "companion", f"คอลัมน์ใต้หัวตารางรายการ {h['id']}", [], [])
    return [out.get(s["id"]) or Decision(s["id"], False, "drop", "ข้อความยาวที่ไม่มีค่าการเงินและสกัดฟิลด์ไม่ได้") for s in sections]
