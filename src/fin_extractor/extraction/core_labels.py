"""อ่านฟิลด์แกนกลางจากคู่ ป้าย→ค่า ที่พิมพ์บนเอกสาร (ตอนนี้: ``buyer.name`` ชื่อลูกค้า).

ป้ายมาจาก config (``core_labels`` ค่าตั้งต้น และ ``document_types.<ชนิด>.core_labels`` ทับต่อชนิด) ไม่ฝังในโค้ด
ใช้เฉพาะป้ายที่เห็นจริงบนสแกน (ระบุ ``source``) ป้ายกว้าง ๆ อย่าง "ลูกค้า" เปล่า ๆ ไม่ใช้ เพราะเจอเป็นป้ายหลอกหลายแบบ:
``สำหรับลูกค้า / For Customer`` (ป้ายระบุสำเนา), ``รหัสลูกค้า C00964``, ``ราคาสำหรับลูกค้าพิเศษ``, ``Customer Tax ID.:``

วิธีอ่านค่าหลังป้าย (ลำดับสำคัญ):
1. ตัดคำนำหน้าป้ายที่ OCR ต่อติดมา (``Name``, ``Address``, ``ชื่อ.``, ``ซื่อ``, ``ชือซี``)
2. ตัดค่าที่คำหยุดแรกที่เจอ: ป้ายของช่องอื่น (วันที่/Invoice/ที่อยู่/เลขประจำตัว ...), ตัวคั่นตาราง ``|``, เลขภาษี 13 หลัก,
   เลขที่บ้านที่ติดท้ายชื่อ (``จำกัต49/12``) หรือคำกำกับในวงเล็บ (``(ให้ระบุ ...``, ``(ผู้ซื้อ/ผู้รับบริการ)``)
3. ตรวจค่าก่อนรับ: ว่าง / ไม่มีตัวอักษร / เป็นรหัส (``C00964``) / ขึ้นต้นด้วยป้ายอื่น ถือว่าไม่ใช่ชื่อ
4. ถ้าบรรทัดเดียวกันไม่ได้ค่า ลองบรรทัดถัดไป: ``next_line`` = บรรทัดอะไรก็ได้ (50 ทวิ: ชื่ออยู่บรรทัดใต้ป้ายผู้ถูกหักภาษี)
   ``next_line_caption`` = เฉพาะบรรทัดที่ขึ้นต้นด้วยป้ายกำกับชื่อ (``Name |``, ``Customer Name:``) ซึ่งเป็นรูปที่ Typhoon ให้
   เมื่อป้ายไทยอยู่บรรทัดเดียวและค่าอยู่ในเซลล์ตาราง (001306, 001317, 001319 ในผล vlm)
5. ข้อความแบบตาราง (``|``): ข้ามเซลล์ที่เป็นแค่ป้ายภาษาอังกฤษ (``Customer name``) แล้วเอาเซลล์แรกที่มีเนื้อหา

ไม่เจอป้ายหรือค่าไม่ผ่านการตรวจ = ไม่มีค่า (ไม่เดา) บรรทัดที่มีสองชื่อในบรรทัดเดียว (``Shipper:- Consignee:-``) ไม่มีป้ายที่ตรงจึงไม่ให้ค่า
"""

from __future__ import annotations

import re

from ..doctypes import FieldSpec, alias_regex
from .patterns import TAX_ID_RE

_LEAD = re.compile(r"^[\s:：.\-/|]+")
# คำนำหน้าที่ OCR ต่อติดหลังป้าย (เห็นใน 1307, 1311, 1317, 1318, 1321, 1323)
_PREFIX = re.compile(r"^(?:(?:customer\s*name|name|address|code)\b|ชื่อ|ซื่อ|ชือซี|ที่อยู่|นามลูกค้า)[\s:：.\-,]*", re.I)
# เซลล์/ค่าที่เป็นแค่ป้ายกำกับ (เห็นใน 1308, 1309, 1317, 1319 ของผล vlm): ไม่ใช่ชื่อ
_LABEL_ONLY = re.compile(r"(?:customer\s*)?name|customer|address|ชื่อ|ที่อยู่|นามลูกค้า", re.I)
# บรรทัดถัดไปที่ขึ้นต้นด้วยป้ายกำกับชื่อ (ใช้กับ next_line_caption)
_CAPTION = re.compile(r"^\s*(?:(?:customer\s*)?name|ชื่อ|นามลูกค้า)\s*[:：|]", re.I)
# ป้ายของช่องอื่นที่ตามหลังชื่อในบรรทัดเดียวกัน (เห็นใน 1306, 1308, 1309, 1323)
_STOP = re.compile(
    r"วันที่|เลขที่|เลขประจำตัว|ที่อยู่|พนักงานขาย|รหัสลูกค้า|กำหนดชำระ|"
    r"\b(?:issue\s*date|date|invoice|address|order|salesman|term|tax\s*id|customer\s*code)\b",
    re.I,
)
_HOUSE_NO = re.compile(r"\d{1,4}/\d{1,4}")          # 49/12 ติดท้ายชื่อ (1317, 1318)
_HINT = re.compile(r"\s*\((?:ให้|ผู้)")             # (ให้ระบุ ...) / (ผู้ซื้อ/ผู้รับบริการ) (1307, 1320, 1322)
_CODE_LIKE = re.compile(r"[A-Za-z]{0,2}\d+[\d\-/.]*")  # C00964
_BRANCH = re.compile(r"\s*\(\s*(สำนักงานใหญ่|head\s*office)\s*\)\s*$", re.I)


def split_branch(name: str) -> tuple[str, str | None]:
    """``บริษัท ก จำกัด (สำนักงานใหญ่)`` → (``บริษัท ก จำกัด``, ``สำนักงานใหญ่``)."""
    m = _BRANCH.search(name)
    return (name[: m.start()].strip(), m.group(1)) if m else (name.strip(), None)


def _clean(text: str) -> str:
    t = _LEAD.sub("", text)
    cells = [c.strip() for c in t.split("|")]
    while cells and (not cells[0].strip(" :：.-") or _LABEL_ONLY.fullmatch(cells[0].strip(" :：.-"))):  # ข้ามเซลล์ว่าง/เซลล์ป้ายล้วน
        cells.pop(0)
    t = cells[0] if cells else ""
    for _ in range(3):  # คำนำหน้าซ้อนกันได้ เช่น "ชือซี" ตามหลัง ":-"
        nxt = _PREFIX.sub("", t)
        if nxt == t:
            break
        t = _LEAD.sub("", nxt)
    cuts = [m.start() for rx in (_STOP, _HOUSE_NO, _HINT, TAX_ID_RE) if (m := rx.search(t))]
    return (t[: min(cuts)] if cuts else t).strip(" :：-,")  # ไม่ตัด "." ท้ายค่า ("CO., LTD." เป็นส่วนของชื่อ)


def _valid(value: str) -> bool:
    if not value or not re.search(r"[^\W\d_]", value):  # ว่าง / ไม่มีตัวอักษร
        return False
    if _CODE_LIKE.fullmatch(value) or value.startswith("(") or _LABEL_ONLY.fullmatch(value):
        return False
    return not _STOP.match(value)  # ขึ้นต้นด้วยป้ายอื่น = ไม่ใช่ชื่อ


def _read(spec: FieldSpec, lines: list[str]) -> str | None:
    for i, line in enumerate(lines):  # บรรทัดแรกที่ให้ค่าที่ใช้ได้ชนะ
        for alias in spec.aliases:
            for m in alias_regex(alias).finditer(line):
                value = _clean(line[m.end():])
                if _valid(value):
                    return value
                nxt_line = lines[i + 1] if i + 1 < len(lines) else ""
                if nxt_line and (spec.next_line or (spec.next_line_caption and _CAPTION.match(nxt_line))):
                    nxt = _clean(nxt_line)
                    if _valid(nxt):
                        return nxt
    return None


def extract_buyer_name(lines: list[str], specs: tuple[FieldSpec, ...]) -> tuple[str | None, str | None]:
    """(ชื่อลูกค้า, สาขา) จากป้ายใน ``specs`` (key ``buyer.name``) สาขา = ``(สำนักงานใหญ่)`` ที่ติดท้ายชื่อถ้ามี."""
    for spec in specs:
        if spec.key == "buyer.name" and (value := _read(spec, lines)):
            name, branch = split_branch(value)
            return (name or None), branch
    return None, None
