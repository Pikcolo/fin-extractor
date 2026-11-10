"""สกัด field ด้วย keyword + regex จากข้อความ OCR ทีละบรรทัด.

ไม่ต้องใช้ model เพิ่ม เร็ว อธิบายได้ แต่เปราะกับ layout แปลก ๆ —
ใช้เป็น baseline หรือ fallback เมื่อ LLM ใช้ไม่ได้
"""

from __future__ import annotations

import html
import re
from decimal import Decimal

from ..schema import InvoiceData, LineItem
from ..validation.tax_id import is_valid_thai_tax_id
from .patterns import AMOUNT_RE, PERCENT_RE, find_amounts, find_tax_ids, parse_date

DOC_TYPES = [
    (r"ใบกำกับภาษี|tax\s*invoice", "tax_invoice"),
    (r"ใบเสร็จ|receipt", "receipt"),
    (r"ใบแจ้งหนี้|invoice", "invoice"),
    (r"ใบเสนอราคา|quotation", "quotation"),
]
COMPANY_RE = re.compile(r"บริษัท|ห้างหุ้นส่วน|หจก\.?|co\.?,?\s*ltd|company|limited|corporation|จำกัด", re.I)
BUYER_RE = re.compile(r"ลูกค้า|ผู้ซื้อ|customer|buyer|bill\s*to|sold\s*to", re.I)
DATE_LABEL_RE = re.compile(r"วันที่|date", re.I)
INVOICE_NO_RE = re.compile(
    r"(?:เลขที่(?:ใบกำกับ(?:ภาษี)?|เอกสาร)?|invoice\s*no\.?|receipt\s*no\.?|doc(?:ument)?\s*no\.?|\bno\.)"
    r"[\s:/#.]*(?:no\.?)?[\s:#]*([A-Za-z0-9][A-Za-z0-9\-/]{2,})",
    re.I,
)
ITEM_HEADER_RE = re.compile(r"รายการ|description|item|สินค้า", re.I)
ITEM_END_RE = re.compile(r"หมายเหตุ|remark|note", re.I)

# ลำดับสำคัญ: เช็คแบบเจาะจงก่อน ("Grand Total (VAT Excluded)" ต้องไม่ถูกจับเป็น total)
AMOUNT_FIELDS: list[tuple[str, re.Pattern]] = [
    ("amount_before_vat", re.compile(
        r"ก่อน(?:รวม)?ภาษี|ไม่รวมภาษี|before\s*vat|vat\s*excl|excl(?:uding|\.)?\s*vat|pre[-\s]?vat", re.I)),
    ("total_amount", re.compile(
        r"รวมภาษี|รวมทั้งสิ้น|ยอดสุทธิ|ยอดชำระ|จำนวนเงินรวม|vat\s*incl|incl(?:uding|\.)?\s*vat|"
        r"grand\s*total|net\s*total|total\s*amount|amount\s*due", re.I)),
    ("vat_amount", re.compile(r"ภาษีมูลค่าเพิ่ม|\bvat\b", re.I)),
    ("discount", re.compile(r"ส่วนลด|discount", re.I)),
    ("subtotal", re.compile(r"รวมเงิน|^\s*รวม|sub\s*-?\s*total|\btotal\b", re.I)),
]


def markdown_to_lines(markdown: str) -> list[str]:
    """แปลง Markdown/HTML table จาก VLM ให้เป็นบรรทัดข้อความ (1 แถวตาราง = 1 บรรทัด)."""
    text = re.sub(r"</t[dh]>", " | ", markdown, flags=re.I)
    text = re.sub(r"</tr>|<br\s*/?>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\*\*|__", "", text)  # ตัวหนาใน markdown
    lines = [re.sub(r"[ \t]+", " ", ln).strip(" |#*") for ln in text.splitlines()]
    return [ln for ln in lines if ln]


AMOUNT_ONLY_RE = re.compile(r"^[\s|฿]*-?[\d,]+\.\d{2}[\s|บาทTHB]*$")
MAX_SUBTOTAL_LABEL = 30  # "Total" เป็นคำทั่วไป เช่น "Total THB withdrawal fee ..." ในรายการ
# ป้ายที่บอกชัดว่าเป็นยอดย่อย — ต่างจาก "Total" / "รวม" เปล่า ๆ ที่เป็นได้ทั้งยอดย่อยและยอดรวม
EXPLICIT_SUBTOTAL_RE = re.compile(r"รวมเงิน|sub\s*-?\s*total", re.I)


def _classify_amount_line(line: str, seen_subtotal_or_vat: bool = False) -> str | None:
    """จำแนกบรรทัดยอดเงิน

    "Total" / "รวม" เปล่า ๆ ตีความตามตำแหน่ง: ถ้ามาก่อนยอดย่อยและ VAT คือยอดย่อย
    แต่ถ้ามาหลังจากนั้นคือยอดรวม — ใบแจ้งหนี้ทั่วไปเรียง Subtotal → VAT → TOTAL
    เดิมจัดเป็นยอดย่อยเสมอ บรรทัด TOTAL จึงถูกทิ้ง (ช่องยอดย่อยมีค่าแล้ว) และ
    เอกสารได้ ``failed`` เพราะไม่มียอดรวม ทั้งที่ OCR อ่านได้ครบ (VATGB-001)
    """
    for field, regex in AMOUNT_FIELDS:
        if regex.search(line):
            if field == "subtotal":
                if len(AMOUNT_RE.sub("", line).strip()) > MAX_SUBTOTAL_LABEL:
                    return None
                if seen_subtotal_or_vat and not EXPLICIT_SUBTOTAL_RE.search(line):
                    return "total_amount"
            return field
    return None


def _line_amount(lines: list[str], i: int) -> Decimal | None:
    """ยอดเงินของบรรทัดสรุป — ถ้าไม่มีในบรรทัด ดูบรรทัดถัดไปที่มีแต่ตัวเลข (OCR แยกคอลัมน์)."""
    amounts = find_amounts(lines[i])
    if not amounts and i + 1 < len(lines) and AMOUNT_ONLY_RE.match(lines[i + 1]):
        amounts = find_amounts(lines[i + 1])
    return amounts[-1] if amounts else None


def _clean_company_name(line: str) -> str:
    return re.sub(r"^(?:ผู้ขาย|vendor|seller|from)\s*[:/]?\s*", "", line, flags=re.I).strip()


def _extract_items(lines: list[str], start: int, end: int) -> list[LineItem]:
    items: list[LineItem] = []
    pending: list[str] = []
    for line in lines[start:end]:
        amounts = find_amounts(line)
        desc = re.sub(r"^\s*\d{1,3}[.)]?\s+", "", line)  # ตัดเลขลำดับ
        for a in re.findall(r"-?[\d,]+\.\d{2}", desc):
            desc = desc.replace(a, " ")
        desc = re.sub(r"[\s|]+", " ", desc).strip()
        if amounts:
            text = " ".join([*pending, desc]).strip()
            items.append(LineItem(description=text or "-", amount=amounts[-1]))
            pending = []
        elif desc:
            pending.append(desc)
    if pending and items:  # บรรทัดคำอธิบายที่อยู่ใต้ยอดเงิน
        items[-1].description += " " + " ".join(pending)
    return items


def extract_fields(lines: list[str]) -> InvoiceData:
    data = InvoiceData()
    full_text = "\n".join(lines)

    for pattern, doc_type in DOC_TYPES:
        if re.search(pattern, full_text, re.I):
            data.document_type = doc_type
            break

    in_buyer = False
    item_start = summary_start = None
    for i, line in enumerate(lines):
        if BUYER_RE.search(line):
            in_buyer = True
        party = data.buyer if in_buyer else data.vendor

        if not data.vendor.name and not in_buyer and COMPANY_RE.search(line):
            data.vendor.name = _clean_company_name(line)

        for tid in find_tax_ids(line):
            if not party.tax_id and is_valid_thai_tax_id(tid):
                party.tax_id = tid
        if "สำนักงานใหญ่" in line or re.search(r"head\s*office", line, re.I):
            party.branch = party.branch or "สำนักงานใหญ่"

        if not data.invoice_no and (m := INVOICE_NO_RE.search(line)):
            val = m.group(1)
            if re.search(r"\d", val) and not find_tax_ids(val):
                data.invoice_no = val

        if not data.date and DATE_LABEL_RE.search(line):
            data.date = parse_date(line)

        if item_start is None and ITEM_HEADER_RE.search(line) and not find_amounts(line):
            item_start = i + 1

        seen = data.subtotal is not None or data.vat_amount is not None
        field = _classify_amount_line(line, seen_subtotal_or_vat=seen)
        amount = _line_amount(lines, i) if field else None
        ends_items = amount is not None or ITEM_END_RE.search(line)
        if ends_items and item_start is not None and summary_start is None and i >= item_start:
            summary_start = i
        if amount is not None and getattr(data, field) is None:
            setattr(data, field, amount)
            if field == "vat_amount" and (p := PERCENT_RE.search(line)):
                data.vat_rate = Decimal(p.group(1)) / 100

    data.date = data.date or parse_date(full_text)
    if data.total_amount is None and data.vat_amount is None and data.discount is None:
        data.total_amount = data.subtotal  # เอกสารที่มีแค่ "รวม/Total" บรรทัดเดียว
    if item_start is not None:
        data.items = _extract_items(lines, item_start, summary_start or len(lines))
    return data

