"""รายการสินค้าจากตาราง HTML ในข้อความดิบ (ตัวอ่านที่ตอบเป็นตาราง เช่น OpenThai) อ่านตามหัวคอลัมน์.

ตัวแยกแบบทีละบรรทัด (``rule_extractor._extract_items``) ไม่รู้ว่าตัวเลขไหนอยู่คอลัมน์ไหน: จำนวน 1.00 กลายเป็นยอดของรายการ
หน่วยกับราคาแตกเป็นรายการใหม่ (1310) และแถวยอดรวมท้ายตารางถูกนับเป็นสินค้า (1308, 1313) ที่นี่:

1. หาแถวหัวตาราง: มีคอลัมน์คำอธิบาย + คอลัมน์ยอดเงิน และไม่มีตัวเลขยอดเงิน (ไม่จำเป็นต้องเป็นแถวแรก: 1313 หัวอยู่กลางตาราง)
2. จัดบทบาทคอลัมน์จากคำในหัว (เช็คแบบเจาะจงก่อน เช่น "จำนวนเงิน" ก่อน "จำนวน") หัวที่ colspan หลายช่อง (จำนวนเงิน (บาท) = บาท|สตางค์ 1320) ขยายตามจำนวนช่อง
3. แถวข้อมูลหลังหัว = รายการ จนเจอแถวสรุป (ช่องที่เป็นป้ายยอดเงินสั้น ๆ หรือจำนวนเงินตัวอักษร) แถวที่ไม่มียอดเงินไม่ใช่รายการ
4. ตารางที่ถูกตัดเป็นหลายท่อนมีหัวซ้ำทุกท่อน (regions repeat-header) จึงอ่านทีละตารางได้เลย
5. ช่องเดียวมีหลายค่า (1309: จำนวน ``1 2`` ยอด ``115.00 3,860.00``) แยกเป็นหลายรายการเมื่อทุกคอลัมน์ตัวเลขมีค่าเท่ากัน
   คำอธิบายแยกตามรายการไม่ได้ (ไม่รู้ว่าตัดตรงไหน) จึงใช้ข้อความทั้งช่องกับทุกรายการ
"""

from __future__ import annotations

import html
import re
from decimal import Decimal

from ..readers.lines import normalize_thai
from ..schema import LineItem
from .patterns import PERCENT_RE, find_amounts, fix_ocr_numbers

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", re.S | re.I)
_COLSPAN_RE = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.I)

# บทบาทคอลัมน์จากคำในหัว ลำดับสำคัญ (ตัวแรกที่ตรงชนะ) — คำทุกตัวเห็นบนหัวตารางของสแกนจริง:
# จำนวนเงิน (1306, 1308, 1320) ก่อนจำนวน · ราคา/หน่วย, Price/Unit, หน่วยละ (1317) ก่อนหน่วย/Unit (1310: ด้าม, ขวด)
# รหัสสินค้า / Article Number ก่อนคำอธิบาย (มีคำว่า "สินค้า"/"Article" ด้วย) · ราคา, รวม (1312) · Net Charges, Charges Description (1313)
COLUMN_ROLES: list[tuple[str, re.Pattern]] = [
    ("amount", re.compile(r"จำนวนเงิน|amount|net\s*charges?|^\s*รวม\s*$", re.I)),
    ("unit_price", re.compile(r"ราคา|price|หน่วยละ", re.I)),
    ("uom", re.compile(r"หน่วย|\bunit\b", re.I)),
    ("qty", re.compile(r"จำนวน|\bqty\b|quantity", re.I)),
    ("discount", re.compile(r"ส่วนลด|discount", re.I)),
    ("code", re.compile(r"รหัส|\bcode\b|article\s*number", re.I)),
    ("no", re.compile(r"ลำดับ|^\s*item\b|^\s*เลขที่", re.I)),
    ("description", re.compile(r"รายการ|รายละเอียด|description|ชื่อสินค้า", re.I)),
]
# จำนวนเงินตัวอักษร ("สามพันเก้าร้อยสี่สิบเก้าบาทถ้วน" 1312, "(หนึ่งร้อย...บาทสิบหกสตางค์.)" 1316) = แถวสรุป
_WORDS_AMOUNT_RE = re.compile(r"บาท\s*ถ้วน|บาท.*สตางค์")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SATANG_RE = re.compile(r"^\.?(\d{2})$")


def _text(cell_html: str) -> str:
    t = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<br\s*/?>", " ", cell_html, flags=re.I)))
    return re.sub(r"\s+", " ", t).strip()


def parse_tables(raw: str) -> list[list[list[tuple[str, int]]]]:
    """ตาราง HTML ทุกตาราง → แถว → ช่อง ``(ข้อความ, ส่วนที่)`` ช่องที่ colspan=n ขยายเป็น n ช่อง ส่วนที่ 0..n-1."""
    tables = []
    for t in _TABLE_RE.findall(raw):
        rows = []
        for tr in _ROW_RE.findall(t):
            row = []
            for attrs, body in _CELL_RE.findall(tr):
                span = int(m.group(1)) if (m := _COLSPAN_RE.search(attrs)) else 1
                text = _text(body)
                row += [(text, k) for k in range(max(span, 1))]
            rows.append(row)
        tables.append(rows)
    return tables


def _role(text: str) -> str | None:
    for role, rx in COLUMN_ROLES:
        if rx.search(text):
            return role
    return None


def _header_roles(row: list[tuple[str, int]]) -> list[str | None] | None:
    """แถวหัวตาราง → บทบาทของแต่ละช่อง (ช่องต่อของหัว colspan ที่เป็นยอดเงิน = สตางค์) หรือ None ถ้าไม่ใช่หัวตารางรายการ."""
    if any(find_amounts(t) for t, _k in row):
        return None
    roles = [_role(t) if k == 0 else None for t, k in row]
    for j, (_t, k) in enumerate(row):
        if k:
            roles[j] = "satang" if roles[j - k] == "amount" else roles[j - k]
    return roles if "description" in roles and "amount" in roles else None


def _is_summary(row: list[tuple[str, int]], max_label: int) -> bool:
    from .rule_extractor import _cell_label  # ป้ายยอดเงินชุดเดียวกับตัวแยกยอด (import ในฟังก์ชันกันวนกัน)

    for text, _k in row:
        if not text:
            continue
        if _WORDS_AMOUNT_RE.search(text) and not re.search(r"\d", text):
            return True
        label = re.sub(r"[\d,.\s|/%]+", " ", text).strip()
        if len(label) <= max_label and _cell_label(text, True):
            return True
    return False


def _numbers(text: str) -> list[Decimal]:
    return [Decimal(n.replace(",", "")) for n in _NUMBER_RE.findall(PERCENT_RE.sub(" ", text))]


def _items_of_row(cells: dict[str, str]) -> list[LineItem]:
    amounts = find_amounts(PERCENT_RE.sub(" ", cells.get("amount", "")))
    if not amounts and (m := _NUMBER_RE.fullmatch(cells.get("amount", "").strip())):
        amounts = [Decimal(m.group().replace(",", ""))]  # บาทที่ไม่มีทศนิยม ต่อสตางค์ด้านล่าง (1320: 103,062 | 50)
    if not amounts:
        return []
    if len(amounts) == 1 and (s := _SATANG_RE.match(cells.get("satang", "").strip())):
        amounts = [(amounts[0] + Decimal(s.group(1)) / 100).quantize(Decimal("0.01"))]
    desc = cells.get("description") or cells.get("code") or "-"
    qty = _numbers(cells.get("qty", ""))
    price = find_amounts(cells.get("unit_price", ""))
    k = len(amounts)
    code_n = len(cells.get("code", "").split())
    if k == 1:
        return [LineItem(description=desc, quantity=qty[0] if len(qty) == 1 else None,
                         unit_price=price[0] if len(price) == 1 else None, amount=amounts[0])]
    aligned = all(n in (0, k) for n in (len(qty), len(price))) and code_n in (0, k)
    return [LineItem(description=desc, quantity=qty[i] if aligned and qty else None,
                     unit_price=price[i] if aligned and price else None, amount=amounts[i]) for i in range(k)]


def table_items(raw: str, max_label: int) -> list[LineItem] | None:
    """รายการจากตารางรายการในข้อความดิบ ``None`` = ข้อความไม่มีตาราง HTML (ให้ใช้ตัวแยกทีละบรรทัด)
    ``[]`` = มีตารางแต่ไม่มีตารางรายการ (ไม่เดาจากบรรทัด: 50 ทวิ ใบศุลกากร ได้รายการขยะจากทางนั้น)."""
    tables = parse_tables(fix_ocr_numbers(normalize_thai(raw)))
    if not tables:
        return None
    items: list[LineItem] = []
    for rows in tables:
        roles = None
        for row in rows:
            if (head := _header_roles(row)) is not None:
                roles = head
                continue
            if roles is None:
                continue
            if _is_summary(row, max_label):
                roles = None  # แถวสรุปปิดตาราง แถวหลังจากนี้ไม่ใช่รายการ (จนกว่าจะเจอหัวใหม่)
                continue
            cells: dict[str, str] = {}
            for (text, _k), role in zip(row, roles):
                if role and text:
                    cells[role] = f"{cells[role]} {text}" if role in cells else text
            items += _items_of_row(cells)
    return items
