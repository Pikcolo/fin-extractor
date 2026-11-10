"""Regex + ตัว normalize สำหรับข้อมูลการเงินไทย: เลขผู้เสียภาษี, วันที่ (พ.ศ./ค.ศ.), ยอดเงิน."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date
from decimal import Decimal, InvalidOperation

# ---------- เลขประจำตัวผู้เสียภาษี (13 หลัก อาจมีขีด/ช่องว่างคั่น เช่น 0-1055-59117-12-8)
TAX_ID_RE = re.compile(r"(?<![\d])\d(?:[ \-]?\d){12}(?![\d])")


def normalize_tax_id(text: str | None) -> str | None:
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return digits if len(digits) == 13 else None


def find_tax_ids(text: str) -> list[str]:
    return [d for m in TAX_ID_RE.finditer(text) if (d := normalize_tax_id(m.group()))]


# ---------- ยอดเงิน: ต้องมีทศนิยม 2 ตำแหน่ง หรือมี comma คั่นหลักพัน
# (กันไม่ให้จับ "7" จาก "VAT 7%" หรือเลขลำดับรายการ)
AMOUNT_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+\.\d{2})(?![\d])")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def parse_amount(text: str | int | float | Decimal | None) -> Decimal | None:
    """แปลงเป็น Decimal — ตัวเลขจาก float ผ่าน ``str`` เสมอ ไม่ใช้ ``Decimal(float)`` ที่ได้เศษทศนิยมไบนารี."""
    if text is None:
        return None
    if isinstance(text, Decimal):
        return text
    if isinstance(text, (int, float)):
        return Decimal(str(text))
    cleaned = re.sub(r"[^\d.\-]", "", str(text))
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def find_amounts(text: str) -> list[Decimal]:
    return [Decimal(m.group().replace(",", "")) for m in AMOUNT_RE.finditer(text)]


# ตัวเลขทุกแบบในข้อความ (รวมจำนวนเต็มไม่มีทศนิยม) — ใช้ตรวจว่าค่าที่สกัดได้ปรากฏในข้อความจริง
NUMBER_RE = re.compile(r"(?<![\d.,])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|(?<![\d.,])-?\d+(?:\.\d+)?(?!\d)")


def find_numbers(text: str) -> set[Decimal]:
    return {Decimal(m.group().replace(",", "")) for m in NUMBER_RE.finditer(text)}


def find_percents(text: str) -> set[Decimal]:
    return {Decimal(m.group(1)) for m in PERCENT_RE.finditer(text)}


# ---------- วันที่
TH_MONTHS = {
    "มกราคม": 1, "ม.ค.": 1, "กุมภาพันธ์": 2, "ก.พ.": 2, "มีนาคม": 3, "มี.ค.": 3,
    "เมษายน": 4, "เม.ย.": 4, "พฤษภาคม": 5, "พ.ค.": 5, "มิถุนายน": 6, "มิ.ย.": 6,
    "กรกฎาคม": 7, "ก.ค.": 7, "สิงหาคม": 8, "ส.ค.": 8, "กันยายน": 9, "ก.ย.": 9,
    "ตุลาคม": 10, "ต.ค.": 10, "พฤศจิกายน": 11, "พ.ย.": 11, "ธันวาคม": 12, "ธ.ค.": 12,
}
EN_MONTHS = {
    m: i for i, names in enumerate([
        ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
        ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
        ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"), ("dec", "december"),
    ], start=1) for m in names
}
_TH_MONTH_ALT = "|".join(re.escape(m) for m in sorted(TH_MONTHS, key=len, reverse=True))
_EN_MONTH_ALT = "|".join(sorted(EN_MONTHS, key=len, reverse=True))

DATE_PATTERNS = [
    # 2023-05-18
    (re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)"), "ymd"),
    # 18/05/2566, 18-05-23
    (re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?!\d)"), "dmy"),
    # 18 พฤษภาคม 2566, 18 พ.ค. 66
    (re.compile(rf"(?<!\d)(\d{{1,2}})\s*({_TH_MONTH_ALT})\s*(\d{{4}}|\d{{2}})(?!\d)"), "d_thmon_y"),
    # 18 May 2023, 18-May-23
    (re.compile(rf"(?<!\d)(\d{{1,2}})[\s\-]*({_EN_MONTH_ALT})[a-z]*\.?[\s\-,]*(\d{{4}}|\d{{2}})(?!\d)", re.I), "d_enmon_y"),
]


def _normalize_year(y: int, today: date | None = None) -> int:
    """แปลงปีเป็น ค.ศ.: 4 หลัก > 2400 ถือเป็น พ.ศ.; 2 หลักเลือกระหว่าง พ.ศ./ค.ศ. ที่ใกล้ปัจจุบันกว่า."""
    today = today or date.today()
    if y >= 2400:
        return y - 543
    if y >= 100:
        return y
    candidates = [2000 + y, 2500 + y - 543]
    valid = [c for c in candidates if c <= today.year + 1] or candidates
    return min(valid, key=lambda c: abs(today.year - c))


def _iter_dates(text: str, today: date | None = None) -> Iterator[date]:
    for regex, kind in DATE_PATTERNS:
        for m in regex.finditer(text):
            a, b, c = m.groups()
            try:
                if kind == "ymd":
                    y, mo, d = int(a), int(b), int(c)
                elif kind == "dmy":
                    d, mo, y = int(a), int(b), int(c)
                elif kind == "d_thmon_y":
                    d, mo, y = int(a), TH_MONTHS[b], int(c)
                else:
                    d, mo, y = int(a), EN_MONTHS[b.lower()], int(c)
                yield date(_normalize_year(y, today), mo, d)
            except (ValueError, KeyError):
                continue


def parse_date(text: str, today: date | None = None) -> date | None:
    """คืนวันที่แรกที่ parse ได้ในข้อความ (ค.ศ.)."""
    return next(_iter_dates(text, today), None)


def find_dates(text: str, today: date | None = None) -> set[date]:
    """วันที่ทุกค่าที่ parse ได้ในข้อความ (ค.ศ.)."""
    return set(_iter_dates(text, today))
