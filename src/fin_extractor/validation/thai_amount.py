"""จำนวนเงินตัวอักษรภาษาไทย → ตัวเลข ใช้ตรวจว่าตัวเลขกับตัวอักษรบนเอกสารตรงกัน (check ``amount_in_words``)

อ่านตามหลักการอ่านตัวเลขไทย: หลัก (ศูนย์–เก้า, เอ็ด = หนึ่งที่หลักหน่วย, ยี่ = สองที่หลักสิบ) คูณหน่วย (สิบ ร้อย พัน หมื่น แสน)
และ "ล้าน" คูณทุกอย่างทางซ้าย · "บาท" แบ่งส่วนบาทกับสตางค์ · "ถ้วน" / "ตรง" = ไม่มีสตางค์
อ่านไม่ได้ทั้งหมด (มีคำที่ไม่รู้จัก เช่น OCR อ่านเพี้ยน "หกสืบข้าพ") = None ไม่เดา — check จะเป็น "ตรวจไม่ได้" ไม่ใช่ "ไม่ผ่าน"
"""

from __future__ import annotations

import re
from decimal import Decimal

DIGITS = {"ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "ยี่": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9}
UNITS = {"สิบ": 10, "ร้อย": 100, "พัน": 1000, "หมื่น": 10_000, "แสน": 100_000}
_TOKEN = re.compile("|".join(sorted([*DIGITS, *UNITS], key=len, reverse=True)))


def _below_million(s: str) -> int | None:
    """คำที่ไม่มี "ล้าน" → จำนวนเต็ม (None = มีส่วนที่อ่านไม่ได้)."""
    total, pending, pos = 0, None, 0
    for m in _TOKEN.finditer(s):
        if m.start() != pos:
            return None
        pos = m.end()
        w = m.group()
        if w in DIGITS:
            if pending is not None:
                return None  # ตัวเลขสองตัวติดกัน
            pending = DIGITS[w]
        else:
            total += (1 if pending is None else pending) * UNITS[w]  # "สิบ" เปล่า ๆ = สิบ
            pending = None
    if pos != len(s):
        return None
    return total + (pending or 0)


def _integer(s: str) -> int | None:
    if not s:
        return 0
    parts = s.split("ล้าน")
    value = 0
    for p in parts[:-1]:  # "หนึ่งล้านสองล้าน" ไม่ใช่รูปปกติ แต่ "ล้านล้าน" ใช้คูณซ้อนได้
        v = _below_million(p) if p else 1
        if v is None:
            return None
        value = (value + v) * 1_000_000
    last = _below_million(parts[-1])
    return None if last is None else value + last


def words_to_amount(text: str | None) -> Decimal | None:
    """"(สามพันหนึ่งร้อยแปดสิบเจ็ดบาทห้าสิบสตางค์)" → Decimal('3187.50') อ่านไม่ได้ = None."""
    if not text:
        return None
    s = re.sub(r"[\s()\[\]\-–.,:]|ถ้วน|ตรง|จำนวนเงิน|ตัวอักษร|รวม", "", text)
    if "บาท" not in s:
        return None
    baht, _, rest = s.partition("บาท")
    satang = rest.removesuffix("สตางค์") if rest else ""
    b = _integer(baht)
    st = _below_million(satang) if satang else 0
    if b is None or st is None or st >= 100 or not (baht or satang):
        return None
    return (Decimal(b) + Decimal(st) / 100).quantize(Decimal("0.01"))
