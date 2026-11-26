"""สกัด field ด้วย keyword + regex จากข้อความ OCR ทีละบรรทัด.

ไม่ต้องใช้ model เพิ่ม เร็ว อธิบายได้ แต่เปราะกับ layout แปลก ๆ —
ใช้เป็น baseline หรือ fallback เมื่อ LLM ใช้ไม่ได้
"""

from __future__ import annotations

import html
import re
from decimal import Decimal

from ..readers.lines import normalize_thai
from ..schema import InvoiceData, LineItem
from ..validation.tax_id import is_valid_thai_tax_id
from .item_table import table_items
from .patterns import AMOUNT_RE, PERCENT_RE, find_amounts, find_tax_ids, fix_ocr_numbers, parse_date

DOC_TYPES = [
    (r"ใบกำกับภาษี|tax\s*invoice", "tax_invoice"),
    (r"ใบเสร็จ|receipt", "receipt"),
    (r"ใบแจ้งหนี้|invoice", "invoice"),
    (r"ใบเสนอราคา|quotation", "quotation"),
]
COMPANY_RE = re.compile(r"บริษัท|ห้างหุ้นส่วน|หจก\.?|co\.?,?\s*ltd|company|limited|corporation|จำกัด", re.I)
BUYER_RE = re.compile(r"ลูกค้า|ผู้ซื้อ|customer|buyer|bill\s*to|sold\s*to", re.I)
# ป้ายระบุสำเนา "สำหรับลูกค้า / For Customer" พิมพ์ต่อท้ายชื่อผู้ขายบรรทัดแรก (001308, 001309) ไม่ใช่การเริ่มส่วนผู้ซื้อ
# ถ้านับเป็นส่วนผู้ซื้อ ชื่อและเลขภาษีของผู้ขายจะตกไปช่องผู้ซื้อ
COPY_MARK_RE = re.compile(r"สำหรับ\s*(?:ลูกค้า|ผู้ซื้อ)(?:\s*/\s*for\s*customer)?", re.I)
DATE_LABEL_RE = re.compile(r"วันที่|date", re.I)
INVOICE_NO_RE = re.compile(
    r"(?:เลขที่(?:ใบกำกับ(?:ภาษี)?|เอกสาร)?|invoice\s*no\.?|receipt\s*no\.?|doc(?:ument)?\s*no\.?|\bno\.)"
    r"[\s:/#.]*(?:no\.?)?[\s:#]*([A-Za-z0-9][A-Za-z0-9\-/]{2,})",
    re.I,
)
ITEM_HEADER_RE = re.compile(r"รายการ|description|item|สินค้า", re.I)
# ชื่อคอลัมน์ของหัวตารางรายการที่ OCR แยกเป็นหลายบรรทัด (บรรทัดละช่อง) คำมาจากหัวตารางบนสแกน: 1310 (Item ลำดับ · Article Number รหัสสินค้า ·
# Qty จำนวน · Unit หน่วย · Price/Unit ราคา/หน่วย · Discount ส่วนลด · Vat ภาษี · Amount จำนวนเงิน), 1308/1309 (Product Code · Unit price), 1306 (No. เลขที่)
COLUMN_HEADER_RE = re.compile(r"รายการ|description|item|สินค้า|article|product|code|รหัส|ลำดับ|\bno\b|เลขที่|qty|quantity|จำนวน|unit|หน่วย|"
                              r"price|ราคา|discount|ส่วนลด|vat|ภาษี|amount|จำนวนเงิน", re.I)
ITEM_END_RE = re.compile(r"หมายเหตุ|remark|note", re.I)

# ลำดับสำคัญ: เช็คแบบเจาะจงก่อน ("Grand Total (VAT Excluded)" ต้องไม่ถูกจับเป็น total)
# ป้ายที่เพิ่มมีตัวอย่างจริงจากสแกน 18 ใบ: มูลค่าสินค้าก่อนคิดภาษีมูลค่าเพิ่ม (1310) · รวมมูลค่าสินค้า = ยอดก่อน VAT (1312: 3,690.65 + 258.35 = 3,949.00)
# รวมเงินทั้งสิ้น (1316) · ยอดเงินสุทธิ (1317–1319) · V.A.T. (1317–1319)
AMOUNT_FIELDS: list[tuple[str, re.Pattern]] = [
    ("amount_before_vat", re.compile(
        r"ก่อน(?:รวม|คิด)?ภาษี|ไม่รวมภาษี|มูลค่าสินค้า|before\s*vat|vat\s*excl|excl(?:uding|\.)?\s*vat|pre[-\s]?vat", re.I)),
    ("total_amount", re.compile(
        r"รวมภาษี|รวม(?:เงิน)?ทั้งสิ้น|ยอด(?:เงิน)?สุทธิ|ยอดชำระ|จำนวนเงินรวม|vat\s*incl|incl(?:uding|\.)?\s*vat|"
        r"grand\s*total|net\s*total|total\s*amount|amount\s*due", re.I)),
    ("vat_amount", re.compile(r"ภาษีมูลค่าเพิ่ม|\bvat\b|\bv\.\s*a\.\s*t\b", re.I)),
    ("discount", re.compile(r"ส่วนลด|discount", re.I)),
    ("subtotal", re.compile(r"รวมเงิน|^\s*รวม|sub\s*-?\s*total|\btotal\b", re.I)),
]


def markdown_to_lines(markdown: str) -> list[str]:
    """แปลงข้อความจากตัวอ่าน (Markdown/HTML table หรือบรรทัดล้วน) ให้เป็นบรรทัดข้อความ (1 แถวตาราง = 1 บรรทัด)."""
    # บางตัวอ่าน (OpenThai) พิมพ์ "ำ" เป็น นิคหิต+า (U+0E4D U+0E32) ทำให้ป้ายอย่าง "นำส่ง" จับไม่เจอ
    text = fix_ocr_numbers(normalize_thai(markdown))
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.I)
    text = re.sub(r"</tr>|<br\s*/?>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\*\*|__", "", text)  # ตัวหนาใน markdown
    lines = [re.sub(r"[ \t]+", " ", ln).strip(" |#*") for ln in text.splitlines()]
    return [ln for ln in lines if ln]


_CHUNK_RE = re.compile(r"<!--\s*(s\d+)(?=\s)([^>]*)-->")
REPEAT_HEADER = "repeat-header"  # คำในคอมเมนต์ของท่อนที่ต่อแถวหัวคอลัมน์ซ้ำไว้บนภาพ (regions: ตารางที่ถูกตัดตามความสูง)
SUPPLEMENT = "supplement"  # คำในคอมเมนต์ของท่อนเสริมจากชั้นตรวจความครบ ตามด้วยชื่อตัวอ่านเสริม (sections/coverage.py)


def lines_with_chunks(markdown: str) -> list[tuple[str, str | None]]:
    """``markdown_to_lines`` + รหัส section (``sNN``) ของแต่ละบรรทัด จากคอมเมนต์ ``<!-- sNN kind -->`` ที่ขั้น OCR รายส่วนใส่นำหน้าแต่ละท่อน
    บรรทัดก่อนคอมเมนต์แรก (หรือข้อความที่ไม่มีคอมเมนต์ เช่นอ่านทั้งหน้า) ได้ None
    ท่อนที่คอมเมนต์มี ``repeat-header`` (ภาพมีแถวหัวคอลัมน์ต่อไว้ให้ OCR เห็นบริบท): บรรทัดต้นท่อนที่ซ้ำตรงตัวกับบรรทัดก่อนหน้าคือหัวที่อ่านซ้ำ
    ตัดทิ้งจนถึงบรรทัดแรกที่ไม่ซ้ำ (ไม่ให้ rules เห็นหัวตารางกลางรายการ) ที่เหลือเท่ากับ ``markdown_to_lines`` ทุกประการ."""
    return [(ln, c) for ln, c, _s in chunk_lines(markdown)]


def supplement_of(markdown: str) -> list[str | None]:
    """ขนานกับ ``lines_with_chunks``: ชื่อตัวอ่านเสริมของบรรทัดที่มาจากท่อนเสริม (``<!-- sNN kind supplement <engine> -->``) อื่น ๆ None."""
    return [s for _ln, _c, s in chunk_lines(markdown)]


def chunk_lines(markdown: str) -> list[tuple[str, str | None, str | None]]:
    out: list[tuple[str, str | None, str | None]] = []
    chunk: str | None = None
    supplement: str | None = None
    repeated = False
    pos = 0
    for m in [*_CHUNK_RE.finditer(markdown), None]:
        end = m.start() if m else len(markdown)
        seg = markdown_to_lines(markdown[pos:end])
        if repeated:
            seen = {ln for ln, _c, _s in out}
            k = 0
            while k < len(seg) and seg[k] in seen:
                k += 1
            seg = seg[k:]
        out.extend((ln, chunk, supplement) for ln in seg)
        if m:
            words = m.group(2).split()
            chunk, repeated, pos = m.group(1), REPEAT_HEADER in words, m.end()
            k = words.index(SUPPLEMENT) if SUPPLEMENT in words else -1
            supplement = (words[k + 1] if k + 1 < len(words) else SUPPLEMENT) if k >= 0 else None
    return out


AMOUNT_ONLY_RE = re.compile(r"^[\s|฿]*-?[\d,]+\.\d{2}[\s|บาทTHB]*$")
AMOUNTS_ONLY_RE = re.compile(r"^(?:[\s|฿]*-?[\d,]+\.\d{2}[\s|บาทTHB]*)+$")  # บรรทัดที่มีแต่ยอดเงิน (หนึ่งตัวหรือหลายตัว)
MAX_SUBTOTAL_LABEL = 30  # "Total" เป็นคำทั่วไป เช่น "Total THB withdrawal fee ..." ในรายการ
# ป้ายที่บอกชัดว่าเป็นยอดย่อย — ต่างจาก "Total" / "รวม" เปล่า ๆ ที่เป็นได้ทั้งยอดย่อยและยอดรวม
EXPLICIT_SUBTOTAL_RE = re.compile(r"รวมเงิน|sub\s*-?\s*total", re.I)


# ยอดที่ไม่ใช่ฐานภาษี: "มูลค่าสินค้าที่ยกเว้นภาษีมูลค่าเพิ่ม" (1310) ไม่ใช่ VAT · "Non-Taxable Charges" (1313) — ไม่จัดเป็นช่องใด
EXEMPT_RE = re.compile(r"ยกเว้นภาษี|non[-\s]?taxable|tax\s*exempt|vat\s*exempt", re.I)
SATANG_RE = re.compile(r"^\.?\d{2}$")  # ช่องสตางค์ของตารางบาท|สตางค์ ("103,062 | 50" 1320 · "3,187 | .50" 1321)


def _cell_label(cell: str, seen_subtotal_or_vat: bool) -> str | None:
    """ช่องเดียว (ไม่ใช่ทั้งบรรทัด) → ช่องยอดเงิน

    "Total" / "รวม" เปล่า ๆ ตีความตามตำแหน่ง: ถ้ามาก่อนยอดย่อยและ VAT คือยอดย่อย
    แต่ถ้ามาหลังจากนั้นคือยอดรวม — ใบแจ้งหนี้ทั่วไปเรียง Subtotal → VAT → TOTAL
    เดิมจัดเป็นยอดย่อยเสมอ บรรทัด TOTAL จึงถูกทิ้ง (ช่องยอดย่อยมีค่าแล้ว) และ
    เอกสารได้ ``failed`` เพราะไม่มียอดรวม ทั้งที่ OCR อ่านได้ครบ (VATGB-001)
    ความยาวป้ายนับจากจุดที่เจอป้ายถึงท้ายช่อง (ข้อความหน้าป้ายในช่องเดียวกัน เช่นจำนวนเงินตัวอักษร ไม่นับ)
    """
    if EXEMPT_RE.search(cell):
        return None
    for field, regex in AMOUNT_FIELDS:
        if m := regex.search(cell):
            if field == "subtotal":
                if len(AMOUNT_RE.sub("", cell[m.start():]).strip()) > MAX_SUBTOTAL_LABEL:
                    return None
                if seen_subtotal_or_vat and not EXPLICIT_SUBTOTAL_RE.search(cell):
                    return "total_amount"
            return field
    return None


def _cell_amounts(cell: str) -> list[Decimal]:
    return find_amounts(PERCENT_RE.sub(" ", cell))  # "7.00 %" ไม่ใช่ยอดเงิน


def _line_amounts(lines: list[str], i: int, seen_subtotal_or_vat: bool) -> list[tuple[str, Decimal, str]]:
    """ยอดเงินในบรรทัดสรุป แยกตามช่อง (`` | `` จากตาราง) → [(ช่อง, ยอด, ข้อความป้ายถึงค่า)]
    ช่องที่มีป้าย: ค่าในช่องเดียวกัน (ตัวท้าย) หรือค่าเดียวในช่องถัดไปก่อนถึงช่องป้ายถัดไป หรือบรรทัดถัดไปที่มีแต่ตัวเลข (ช่องป้ายสุดท้ายเท่านั้น)
    ช่องเดียวมีหลายป้ายเรียงกัน + ช่องถัดไปช่องเดียวมีตัวเลขเท่าจำนวนป้าย (1312) → จับคู่ตามลำดับ."""
    cells = [c.strip() for c in lines[i].split(" | ")]
    out: list[tuple[str, Decimal, str]] = []
    labels = [_cell_label(c, seen_subtotal_or_vat) for c in cells]
    labelled = [j for j, f in enumerate(labels) if f]
    for j in labelled:
        field, cell = _cell_label(cells[j], seen_subtotal_or_vat), cells[j]  # จัดใหม่: "Total" หลัง VAT ในบรรทัดเดียวกันเป็นยอดรวม
        stop = next((k for k in labelled if k > j), len(cells))
        own = _cell_amounts(cell)
        nxt = [(k, a) for k in range(j + 1, stop) for a in _cell_amounts(cells[k])]
        if not own and not nxt and stop == len(cells) and i + 1 < len(lines) and AMOUNTS_ONLY_RE.match(lines[i + 1]):
            nxt = [(len(cells), a) for a in find_amounts(lines[i + 1])]  # ค่าอยู่บรรทัดถัดไป (OCR แยกช่องเป็นคนละบรรทัด 1312)
        words = [w for w in (_cell_label(w, False) for w in cell.split()) if w]
        if not own and len(nxt) > 1 and len({k for k, _a in nxt}) == 1 and len(words) == len(nxt) == len(set(words)):
            out += [(w, a, cell) for w, (_k, a) in zip(words, nxt)]
        elif own:
            out.append((field, own[-1], cell))
        elif len({a for _k, a in nxt}) > 1:
            pass  # ป้ายเดียวตามด้วยหลายค่าที่จับคู่ไม่ได้ (50 ทวิ: "รวมเงินที่จ่ายและภาษีที่หักนำส่ง | 106,250.00 | 3,187.50") ไม่รู้ว่าค่าไหน ไม่เดา
        elif nxt:
            k, a = nxt[-1]
            if k < len(cells) and "." not in cells[k] and k + 1 < len(cells) and SATANG_RE.match(cells[k + 1]):
                a = (a + Decimal(cells[k + 1].lstrip(".")) / 100).quantize(Decimal("0.01"))
            out.append((field, a, " ".join(cells[j:k + 1])))
        seen_subtotal_or_vat = seen_subtotal_or_vat or any(f in ("subtotal", "vat_amount") for f, _a, _t in out)
    return out


# ป้ายเลขที่เอกสาร ลำดับความเจาะจง (0 ชนะ): ป้ายของใบกำกับ > ใบเสร็จ/เอกสาร > "เลขที่" / "No." เปล่า ๆ
# "No." เปล่า ๆ นับเฉพาะต้นช่อง: "Account No." (1313), "Hawb No." (1316) ไม่ใช่เลขที่เอกสาร
DOC_NO_LABELS: list[tuple[int, re.Pattern]] = [
    (0, re.compile(r"เลขที่ใบกำกับ(?:ภาษี)?|tax\s*invoice\s*no\.?|invoice\s*(?:no|number)\.?", re.I)),
    (1, re.compile(r"เลขที่(?:ใบเสร็จ(?:รับเงิน)?|เอกสาร)|receipt\s*no\.?|doc(?:ument)?\s*no\.?", re.I)),
    (2, re.compile(r"เลขที่|^\s*no\.", re.I)),
]
DOC_NO_VALUE_RE = re.compile(r"^[\s:/#.]*(?:no\.?)?[\s:#]*([A-Za-z0-9][A-Za-z0-9\-/]{2,})", re.I)
# ค่าหลัง "เลขที่" ที่เป็นที่อยู่ ("เลขที่ 49/12 ซอย 6 ถนน ..." 1308/1309, "เลขที่ 6 ซอย รามอินทรา" 1317)
ADDRESS_RE = re.compile(r"ซอย|ถนน|ถ\.|หมู่|ตำบล|แขวง|อำเภอ|เขต|จังหวัด|\b(?:road|soi|moo)\b", re.I)
DATE_LIKE_RE = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")  # 1322: OCR ใส่วันที่ไว้ในบรรทัดเลขที่


def _doc_no_value(val: str | None) -> str | None:
    if not val or not re.search(r"\d", val) or find_tax_ids(val) or DATE_LIKE_RE.match(val):
        return None
    return val


def _find_invoice_no(lines: list[str]) -> str | None:
    """เลขที่เอกสารจากป้าย เลือกป้ายที่เจาะจงที่สุด (ที่เจอก่อนชนะเมื่อเจาะจงเท่ากัน) อ่านค่าได้สี่แบบ:
    ค่าหลังป้าย (``เลขที่ : AABZ...``) · ค่าก่อนป้ายอังกฤษในช่องเดียวกัน (``ใบวางบิล : IN-691005003 Invoice No`` 1306 — ป้ายไทย ค่า ป้ายอังกฤษ)
    · ช่องถัดไปในแถวเดียวกัน (``เลขที่ใบกำกับภาษี : Invoice No. | BAUNS...`` 1308 เมื่อ OCR อ่านกล่องเป็นตาราง)
    · ป้ายเป็นหัวตาราง ค่าอยู่ช่องเดียวกันของแถวถัดไป (``เลขที่ | ออกเมื่อ`` → ``BI8461-101069 | ...`` 1312)
    สามแบบหลังใช้เมื่อหลังป้ายในช่องนั้นไม่มีตัวเลข (มีแค่ป้ายอังกฤษกำกับ เช่น ``Invoice No.``, ``Order No.``)."""
    best: tuple[int, str] | None = None
    for i, line in enumerate(lines):
        cells = line.split(" | ")
        nxt = lines[i + 1].split(" | ") if i + 1 < len(lines) else []
        for j, cell in enumerate(cells):
            for rank, rx in DOC_NO_LABELS:
                if best is not None and rank >= best[0]:
                    break
                m = rx.search(cell)
                if not m:
                    continue
                after = cell[m.end():]
                val = None if ADDRESS_RE.search(after) else _doc_no_value((DOC_NO_VALUE_RE.match(after) or [None, None])[1])
                bare = not re.search(r"\d", after)  # ช่องนี้มีแค่ป้าย ค่าอยู่ที่อื่น
                if val is None and bare and rank < 2:
                    before = re.search(r"[:：]\s*([A-Za-z0-9][A-Za-z0-9\-/]{2,})\s*$", cell[:m.start()])
                    val = _doc_no_value(before and before.group(1))
                if val is None and bare and j + 1 < len(cells) and not ADDRESS_RE.search(cells[j + 1]):
                    beside = DOC_NO_VALUE_RE.match(cells[j + 1])
                    val = _doc_no_value(beside and beside.group(1))
                if val is None and bare and len(nxt) == len(cells) > 1:
                    head = DOC_NO_VALUE_RE.match(nxt[j])
                    val = _doc_no_value(head and head.group(1))
                if val is not None:
                    best = (rank, val)
                    break
    return best[1] if best else None


def _clean_company_name(line: str) -> str:
    return re.sub(r"^(?:ผู้ขาย|vendor|seller|from)\s*[:/]?\s*", "", line, flags=re.I).strip()


def _extract_items(lines: list[str], start: int, end: int) -> list[LineItem]:
    items: list[LineItem] = []
    pending: list[str] = []
    while start < end and not re.search(r"\d", lines[start]) and COLUMN_HEADER_RE.search(lines[start]):
        start += 1  # หัวตารางที่เหลือ (ชื่อคอลัมน์ไม่มีตัวเลข บรรทัดละช่อง) ไม่ใช่คำอธิบายของรายการแรก
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


def extract_fields(lines: list[str], raw: str | None = None) -> InvoiceData:
    """``raw`` = ข้อความดิบจากตัวอ่าน ถ้ามีตาราง HTML รายการสินค้าอ่านจากตารางตามหัวคอลัมน์ (``item_table``) แทนทีละบรรทัด."""
    data = InvoiceData()
    full_text = "\n".join(lines)

    for pattern, doc_type in DOC_TYPES:
        if re.search(pattern, full_text, re.I):
            data.document_type = doc_type
            break

    in_buyer = False
    item_start = summary_start = None
    for i, line in enumerate(lines):
        if BUYER_RE.search(COPY_MARK_RE.sub("", line)):
            in_buyer = True
        party = data.buyer if in_buyer else data.vendor

        if not data.vendor.name and not in_buyer and COMPANY_RE.search(line):
            data.vendor.name = _clean_company_name(COPY_MARK_RE.sub("", line))

        for tid in find_tax_ids(line):
            if not party.tax_id and is_valid_thai_tax_id(tid):
                party.tax_id = tid
        if "สำนักงานใหญ่" in line or re.search(r"head\s*office", line, re.I):
            party.branch = party.branch or "สำนักงานใหญ่"

        if not data.date and DATE_LABEL_RE.search(line):
            data.date = parse_date(line)

        if item_start is None and ITEM_HEADER_RE.search(line) and not find_amounts(line):
            item_start = i + 1

        seen = data.subtotal is not None or data.vat_amount is not None
        found = _line_amounts(lines, i, seen)
        ends_items = bool(found) or ITEM_END_RE.search(line)
        if ends_items and item_start is not None and summary_start is None and i >= item_start:
            summary_start = i
        for field, amount, text in found:
            if getattr(data, field) is None:
                setattr(data, field, amount)
                if field == "vat_amount" and (p := PERCENT_RE.search(text)):
                    data.vat_rate = Decimal(p.group(1)) / 100

    data.invoice_no = _find_invoice_no(lines)
    data.date = data.date or parse_date(full_text)
    if data.total_amount is None and data.vat_amount is None and data.discount is None:
        data.total_amount = data.subtotal  # เอกสารที่มีแค่ "รวม/Total" บรรทัดเดียว
    table = table_items(raw, MAX_SUBTOTAL_LABEL) if raw else None
    if table is not None:
        data.items = table
    elif item_start is not None:
        data.items = _extract_items(lines, item_start, summary_start or len(lines))
    return data

