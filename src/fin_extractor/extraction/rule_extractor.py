"""สกัด field ด้วย keyword + regex จากข้อความ OCR ทีละบรรทัด.

ไม่ต้องใช้ model เพิ่ม เร็ว อธิบายได้ แต่เปราะกับ layout แปลก ๆ —
ใช้เป็น baseline หรือ fallback เมื่อ LLM ใช้ไม่ได้
"""

from __future__ import annotations

import html
import re
from decimal import Decimal
from functools import lru_cache

from ..config import load_config
from ..doctypes import kind_of, load_kinds
from ..readers.lines import normalize_thai
from ..schema import PARTY_KEYS, InvoiceData, LineItem, Party
from ..validation.tax_id import is_valid_thai_tax_id
from .item_table import table_items
from .patterns import AMOUNT_RE, PERCENT_RE, find_amounts, find_tax_ids, fix_ocr_numbers, parse_date



@lru_cache(maxsize=1)
def default_kinds() -> list[tuple[str, re.Pattern]]:
    """ชนิดเอกสารจาก ``document_kinds`` ใน configs/default.yaml (ที่เดียวกับที่ pipeline ใช้ ไม่มีสำเนาในโค้ด) ใช้เมื่อเรียก ``extract_fields`` โดยไม่ส่ง ``kinds``."""
    return load_kinds(load_config().get("document_kinds"))


# "สำนักงานใหญ่" รวมรูปที่ OCR อ่านเพี้ยน (ส/ล · ใ/โ)
HEAD_OFFICE_RE = re.compile(r"[สล][ำา]นักงาน[ใโ]หญ่|head\s*office", re.I)
# "สาขาที่ออกใบกำกับภาษี : สาขาที่ 00054" / "...คือ : สำนักงานใหญ่ (00000)" → เลขสาขา 5 หลัก หรือสำนักงานใหญ่
ISSUING_BRANCH_RE = re.compile(r"สาขาที่ออกใบกำกับภาษี(?:คือ)?\s*[:：]?\s*(?:สาขาที่\s*(\d{5})|[สล][ำา]นักงาน[ใโ]หญ่)")
COMPANY_RE = re.compile(r"บริษัท|ห้างหุ้นส่วน|หจก\.?|co\.?,?\s*ltd|company|limited|corporation|จำกัด", re.I)
BUYER_RE = re.compile(r"ลูกค้า|ผู้ซื้อ|customer|buyer|bill\s*to|sold\s*to", re.I)
# ป้ายระบุสำเนา "สำหรับลูกค้า / For Customer" พิมพ์ต่อท้ายชื่อผู้ขายบรรทัดแรก ไม่ใช่การเริ่มส่วนผู้ซื้อ
# ถ้านับเป็นส่วนผู้ซื้อ ชื่อและเลขภาษีของผู้ขายจะตกไปช่องผู้ซื้อ
COPY_MARK_RE = re.compile(r"สำหรับ\s*(?:ลูกค้า|ผู้ซื้อ)(?:\s*/\s*for\s*customer)?", re.I)
DATE_LABEL_RE = re.compile(r"วันที่|date", re.I)
INVOICE_NO_RE = re.compile(
    r"(?:เลขที่(?:ใบกำกับ(?:ภาษี)?|เอกสาร)?|invoice\s*no\.?|receipt\s*no\.?|doc(?:ument)?\s*no\.?|\bno\.)"
    r"[\s:/#.]*(?:no\.?)?[\s:#]*([A-Za-z0-9][A-Za-z0-9\-/]{2,})",
    re.I,
)
ITEM_HEADER_RE = re.compile(r"รายการ|description|item|สินค้า", re.I)
# ชื่อคอลัมน์ของหัวตารางรายการที่ OCR แยกเป็นหลายบรรทัด (บรรทัดละช่อง) ตัวอย่างหัวตาราง: Item ลำดับ · Article Number รหัสสินค้า ·
# Qty จำนวน · Unit หน่วย · Price/Unit ราคา/หน่วย · Discount ส่วนลด · Vat ภาษี · Amount จำนวนเงิน · Product Code · Unit price · No. เลขที่
COLUMN_HEADER_RE = re.compile(r"รายการ|description|item|สินค้า|article|product|code|รหัส|ลำดับ|\bno\b|เลขที่|qty|quantity|จำนวน|unit|หน่วย|"
                              r"price|ราคา|discount|ส่วนลด|vat|ภาษี|amount|จำนวนเงิน", re.I)
ITEM_END_RE = re.compile(r"หมายเหตุ|remark|note", re.I)

# ลำดับสำคัญ: เช็คแบบเจาะจงก่อน ("Grand Total (VAT Excluded)" ต้องไม่ถูกจับเป็น total)
# ป้ายที่เพิ่มมีตัวอย่างจริงจากชุดสแกนทดสอบ: มูลค่าสินค้าก่อนคิดภาษีมูลค่าเพิ่ม · รวมมูลค่าสินค้า = ยอดก่อน VAT
# รวมเงินทั้งสิ้น · ยอดเงินสุทธิ · V.A.T.
AMOUNT_FIELDS: list[tuple[str, re.Pattern]] = [
    ("amount_before_vat", re.compile(
        r"ก่อน(?:รวม|คิด)?ภาษี|ไม่รวมภาษี|มูลค่าสินค้า|before\s*vat|vat\s*excl|excl(?:uding|\.)?\s*vat|pre[-\s]?vat|(?<![-\w])taxable\s*(?:charges?|amount|value)", re.I)),
    ("total_amount", re.compile(
        r"รวมภาษี|รวม(?:เงิน)?ทั้งสิ้น|ยอด(?:เงิน)?สุทธิ|ยอดชำระ|จำนวนเงินรวม|vat\s*incl|incl(?:uding|\.)?\s*vat|"
        r"grand\s*total|net\s*total|total\s*amount|amount\s*due", re.I)),
    ("vat_amount", re.compile(r"ภาษีมูลค่าเพิ่ม|\bvat\b|\bv\.\s*a\.\s*t\b", re.I)),
    ("discount", re.compile(r"ส่วนลด|discount", re.I)),
    ("subtotal", re.compile(r"รวมเงิน|^\s*รวม|sub\s*-?\s*total|\btotal\b", re.I)),
]


# \ นำหน้าเครื่องหมายวรรคตอนของ markdown (ไม่รวม | ที่เป็นตัวคั่นช่องของตาราง)
_MD_ESCAPE_RE = re.compile(r"\\([!-/:-@\[\]^_`{}~])")


def markdown_to_lines(markdown: str) -> list[str]:
    """แปลงข้อความจากตัวอ่าน (Markdown/HTML table หรือบรรทัดล้วน) ให้เป็นบรรทัดข้อความ (1 แถวตาราง = 1 บรรทัด)."""
    # บางตัวอ่าน (OpenThai) พิมพ์ "ำ" เป็น นิคหิต+า (U+0E4D U+0E32) ทำให้ป้ายอย่าง "นำส่ง" จับไม่เจอ
    text = fix_ocr_numbers(normalize_thai(markdown))
    text = _MD_ESCAPE_RE.sub(r"\1", text)  # "7\% VAT" "(13หลัก)\*" ตัวอ่านใส่ \ นำหน้าเครื่องหมาย ทำให้จับ "7%" ไม่ได้
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


def chunk_lines(markdown: str) -> list[tuple[str, str | None, str | None]]:
    """(บรรทัด, รหัส section, ชื่อตัวอ่านเสริมถ้าบรรทัดมาจากท่อน supplement) ของทุกบรรทัด ``lines_with_chunks`` คือสองช่องแรก."""
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


# ยอดที่ไม่ใช่ฐานภาษี: "มูลค่าสินค้าที่ยกเว้นภาษีมูลค่าเพิ่ม" · "Non-Taxable Charges" ไม่ใช่ VAT และไม่อยู่ในฐาน VAT
# บรรทัดสรุปเก็บเป็นช่อง non_taxable_amount (ยอดรวม = ฐาน + VAT + ค่านี้) ส่วนหัวตารางรายการยังไม่ถือเป็นป้ายยอด (``_cell_label``)
EXEMPT_RE = re.compile(r"ยกเว้นภาษี|non[-\s]?taxable|tax\s*exempt|vat\s*exempt", re.I)
SATANG_RE = re.compile(r"^\.?\d{2}$")  # ช่องสตางค์ของตารางบาท|สตางค์ ("1,234 | 50" · "1,234 | .50")


def _cell_label(cell: str, seen_subtotal_or_vat: bool) -> str | None:
    """ช่องเดียว (ไม่ใช่ทั้งบรรทัด) → ช่องยอดเงิน

    "Total" / "รวม" เปล่า ๆ ตีความตามตำแหน่ง: ถ้ามาก่อนยอดย่อยและ VAT คือยอดย่อย
    แต่ถ้ามาหลังจากนั้นคือยอดรวม — ใบแจ้งหนี้ทั่วไปเรียง Subtotal → VAT → TOTAL
    เดิมจัดเป็นยอดย่อยเสมอ บรรทัด TOTAL จึงถูกทิ้ง (ช่องยอดย่อยมีค่าแล้ว) และ
    เอกสารได้ ``failed`` เพราะไม่มียอดรวม ทั้งที่ OCR อ่านได้ครบ
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


def _summary_label(cell: str, seen_subtotal_or_vat: bool) -> str | None:
    return "non_taxable_amount" if EXEMPT_RE.search(cell) else _cell_label(cell, seen_subtotal_or_vat)


def _cell_amounts(cell: str) -> list[Decimal]:
    return find_amounts(PERCENT_RE.sub(" ", cell))  # "7.00 %" ไม่ใช่ยอดเงิน


def _line_amounts(lines: list[str], i: int, seen_subtotal_or_vat: bool) -> list[tuple[str, Decimal, str]]:
    """ยอดเงินในบรรทัดสรุป แยกตามช่อง (`` | `` จากตาราง) → [(ช่อง, ยอด, ข้อความป้ายถึงค่า)]
    ช่องที่มีป้าย: ค่าในช่องเดียวกัน (ตัวท้าย) หรือค่าเดียวในช่องถัดไปก่อนถึงช่องป้ายถัดไป หรือบรรทัดถัดไปที่มีแต่ตัวเลข (ช่องป้ายสุดท้ายเท่านั้น)
    ช่องเดียวมีหลายป้ายเรียงกัน + ช่องถัดไปช่องเดียวมีตัวเลขเท่าจำนวนป้าย → จับคู่ตามลำดับ."""
    cells = [c.strip() for c in lines[i].split(" | ")]
    out: list[tuple[str, Decimal, str]] = []
    labels = [_summary_label(c, seen_subtotal_or_vat) for c in cells]
    labelled = [j for j, f in enumerate(labels) if f]
    for j in labelled:
        field, cell = _summary_label(cells[j], seen_subtotal_or_vat), cells[j]  # จัดใหม่: "Total" หลัง VAT ในบรรทัดเดียวกันเป็นยอดรวม
        stop = next((k for k in labelled if k > j), len(cells))
        own = _cell_amounts(cell)
        nxt = [(k, a) for k in range(j + 1, stop) for a in _cell_amounts(cells[k])]
        if not own and not nxt and stop == len(cells) and i + 1 < len(lines) and AMOUNTS_ONLY_RE.match(lines[i + 1]):
            nxt = [(len(cells), a) for a in find_amounts(lines[i + 1])]  # ค่าอยู่บรรทัดถัดไป (OCR แยกช่องเป็นคนละบรรทัด)
        words = [w for w in (_summary_label(w, False) for w in cell.split()) if w]
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
# "No." เปล่า ๆ นับเฉพาะต้นช่อง: "Account No.", "Hawb No." ไม่ใช่เลขที่เอกสาร
DOC_NO_LABELS: list[tuple[int, re.Pattern]] = [
    (0, re.compile(r"เลขที่ใบกำกับ(?:ภาษี)?|tax\s*invoice\s*no\.?|invoice\s*(?:no|number)\.?", re.I)),
    (1, re.compile(r"เลขที่(?:ใบเสร็จ(?:รับเงิน)?|เอกสาร)|receipt\s*no\.?|doc(?:ument)?\s*no\.?", re.I)),
    (2, re.compile(r"เลขที่|^\s*no\.", re.I)),
]
DOC_NO_VALUE_RE = re.compile(r"^[\s:/#.]*(?:no\.?)?[\s:#]*([A-Za-z0-9][A-Za-z0-9\-/]{2,})", re.I)
# ค่าหลัง "เลขที่" ที่เป็นที่อยู่ ("เลขที่ 49/12 ซอย 6 ถนน ...", "เลขที่ 6 ซอย ตัวอย่าง")
ADDRESS_RE = re.compile(r"ซอย|ถนน|ถ\.|หมู่|ตำบล|แขวง|อำเภอ|เขต|จังหวัด|\b(?:road|soi|moo)\b", re.I)
DATE_LIKE_RE = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")  # OCR ใส่วันที่ไว้ในบรรทัดเลขที่


def _doc_no_value(val: str | None) -> str | None:
    if not val or not re.search(r"\d", val) or find_tax_ids(val) or DATE_LIKE_RE.match(val):
        return None
    return val


# ที่อยู่: ป้าย "ที่อยู่" ต้นช่อง (OCR สะกด "ที่อยุ่" ได้ · "ที่อยู่ตามบัตรประชาชน" ในใบสำคัญรับเงิน) ตามด้วย "Address" ได้
# ค่า = ข้อความหลังป้ายในช่องเดียวกัน หรือช่องถัดไป · บรรทัดถัดไปที่ช่องแรกเป็นป้าย "Address" อย่างเดียว = ที่อยู่บรรทัดที่สอง
# (ฟอร์มสองภาษาพิมพ์ ที่อยู่ / Address สองบรรทัด) · ต้องมีตัวเลข (เลขที่บ้าน/รหัสไปรษณีย์) และไม่ขึ้นต้นด้วย "("
# (คำแนะนำของฟอร์ม 50 ทวิ "ที่อยู่:... (ให้ระบุ ...)") · ตัดที่ Tel./โทร
ADDRESS_LABEL_RE = re.compile(r"^\s*ที่อย[ูุ]่(?:ตามบัตรประชาชน)?\s*[:：]?\s*(?:address\b)?\s*[:：]?", re.I)
ADDRESS_CONT_RE = re.compile(r"^\s*address\s*[:：]?\s*$", re.I)
ADDRESS_STOP_RE = re.compile(r"\s*(?:tel\b|โทร)", re.I)


def _find_address(lines: list[str], i: int) -> str | None:
    cells = [c.strip() for c in lines[i].split("|")]
    m = ADDRESS_LABEL_RE.match(cells[0])
    if not m:
        return None
    value = cells[0][m.end():].strip(" :：.…") or (cells[1] if len(cells) > 1 else "")
    nxt = [c.strip() for c in lines[i + 1].split("|")] if i + 1 < len(lines) else []
    if value and len(nxt) > 1 and ADDRESS_CONT_RE.match(nxt[0]):
        value = f"{value}\n{nxt[1]}"  # คงการขึ้นบรรทัด: ตรวจว่าค่าอยู่ในข้อความได้ทีละบรรทัด
    value = value.strip(" :：.…")
    if (cut := ADDRESS_STOP_RE.search(value)):
        value = value[:cut.start()]
    value = value.strip(" :：.…")
    return value if value and not value.startswith("(") and re.search(r"\d", value) else None


NAME_LINE_RE = re.compile(r"ชื่อ|ซื่อ|ข้าพเจ้า|\bname\b", re.I)


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def assign_addresses(lines: list[str], data: InvoiceData) -> None:
    """เติมที่อยู่ผู้ขาย/ผู้ซื้อ (ต้องรู้ชื่อทั้งสองฝ่ายก่อน: เรียกหลังอ่านชื่อจากป้ายของโปรไฟล์)
    ที่อยู่เป็นของฝ่ายที่ชื่อ (ที่สกัดได้) อยู่ในบรรทัดชื่อที่ใกล้ที่สุดเหนือมัน (บรรทัดที่มีป้ายชื่อหรือชื่อบริษัท)
    บรรทัดชื่อนั้นไม่ตรงกับชื่อฝ่ายไหน = ไม่เติม (ไม่เดาฝ่าย: บล็อกผู้ซื้อไม่มีป้าย "ลูกค้า" · บรรทัดที่ขึ้นต้นด้วย "(" เป็นคำแนะนำของฟอร์ม ข้ามไป)."""
    for i in range(len(lines)):
        addr = _find_address(lines, i)
        if not addr:
            continue
        j = next((k for k in range(i - 1, -1, -1) if not lines[k].lstrip().startswith("(")  # "(ให้ระบุ...)" = คำแนะนำของฟอร์ม
                  and (NAME_LINE_RE.search(lines[k]) or COMPANY_RE.search(lines[k]))), None)
        if j is None:
            continue
        near = _squash(lines[j])
        for party in (data.buyer, data.vendor):
            if party.name and not party.address and _squash(party.name) in near:
                party.address = addr
                break


POSTCODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")
_THAI_RE = re.compile(r"[฀-๿]")
# ถ้าเจอบรรทัดพวกนี้ก่อนเริ่มที่อยู่ = ไม่มีที่อยู่ที่ไม่มีป้ายใต้ชื่อ (ป้ายที่อยู่จริงมีตัวอ่านอีกทางแล้ว)
_ADDRESS_ABORT_RE = re.compile(r"เลขประจำตัว|tax\s*id|ใบเสร็จ|ใบกำกับ|invoice|receipt|ที่อยู่|address|^\s*(?:tel|fax|โทร)", re.I)
_LEADING_HEAD_OFFICE_RE = re.compile(r"^\s*(?:[สล][ำา]นักงาน[ใโ]หญ่|head\s*office)\s*", re.I)


def strip_contact(value: str) -> str:
    """ตัดเบอร์โทร/Tel ที่ติดท้ายที่อยู่ (ทั้งค่าจาก rules และจาก LLM)."""
    cut = ADDRESS_STOP_RE.search(value)
    return (value[:cut.start()] if cut else value).strip(" :：.…\n")


def _address_below(lines: list[str], i: int) -> str | None:
    """ที่อยู่ที่พิมพ์ต่อใต้บรรทัดชื่อบริษัท (``lines[i]``) โดยไม่มีป้าย "ที่อยู่": ข้ามบรรทัดชื่อซ้ำ (ชื่ออังกฤษ ป้ายสำเนา) แล้วเก็บบรรทัดที่หน้าตาเป็นที่อยู่
    (มีคำที่อยู่หรือรหัสไปรษณีย์) ต่อกัน หยุดที่บรรทัดอื่น / บรรทัดที่ไม่ใช่ภาษาเดียวกับบรรทัดแรก (คำแปลอังกฤษของที่อยู่ไทย) / เบอร์โทร."""
    block: list[str] = []
    for line in lines[i + 1:]:
        text = line.strip()
        looks = bool(ADDRESS_RE.search(text) or POSTCODE_RE.search(text))
        if not block:
            if COMPANY_RE.search(text) or COPY_MARK_RE.search(text) or text.startswith("("):
                continue
            if _ADDRESS_ABORT_RE.search(text) or not looks:
                return None
        elif not looks or (_THAI_RE.search(block[0]) and not _THAI_RE.search(text)):
            break
        block.append(text)
        if ADDRESS_STOP_RE.search(text):
            break
    if not block:
        return None
    block[0] = _LEADING_HEAD_OFFICE_RE.sub("", block[0])
    return strip_contact("\n".join(block)) or None


def fill_vendor_address(lines: list[str], data: InvoiceData) -> None:
    """ที่อยู่ผู้ขายจากบรรทัดใต้ชื่อบริษัทที่ไม่มีป้าย (ใช้เมื่อยังไม่ได้ที่อยู่จากป้าย "ที่อยู่" เท่านั้น)."""
    name = data.vendor.name
    if data.vendor.address or not name:
        return
    key = _squash(name)
    i = next((k for k, ln in enumerate(lines) if key and key in _squash(ln)), None)
    if i is not None and (addr := _address_below(lines, i)):
        data.vendor.address = addr


def unmix_parties(data: InvoiceData) -> None:
    """ชื่อที่ได้จากป้ายของฝ่ายซื้อ (``buyer.name``) ซ้ำกับชื่อที่ rules ตามตำแหน่งใส่ให้ผู้ขาย = บริษัทเดียวกัน ผู้ขายที่ได้ตามตำแหน่งเป็นของฝ่ายซื้อ
    ย้ายเลขภาษี/สาขา/ที่อยู่ที่ติดมากับมันไปฝั่งซื้อด้วย แล้วล้างฝั่งขาย (ไม่เดาว่าผู้ขายคือใคร)."""
    v, b = data.vendor, data.buyer
    if not (v.name and b.name):
        return
    vk, bk = _squash(v.name).casefold(), _squash(b.name).casefold()
    if len(bk) >= 4 and (bk in vk or vk in bk):
        for attr in (k for k in PARTY_KEYS if k != "name"):
            if not getattr(b, attr):
                setattr(b, attr, getattr(v, attr))
        data.vendor = Party()


def derive_total(data: InvoiceData, tol: Decimal) -> None:
    """ไม่พบยอดรวม แต่ "รวมเงิน" = ก่อน VAT + VAT (+ ไม่เสียภาษี) ตามตัวเลขในเอกสารเอง: ยอดนั้นคือยอดรวม (เอกสารที่ใช้ป้าย "รวมเป็นเงิน" เป็นยอดสุทธิ)."""
    if data.total_amount is None and None not in (data.subtotal, data.amount_before_vat, data.vat_amount):
        if abs(data.subtotal - (data.amount_before_vat + data.vat_amount + (data.non_taxable_amount or 0))) <= tol:
            data.total_amount = data.subtotal


def _find_invoice_no(lines: list[str]) -> str | None:
    """เลขที่เอกสารจากป้าย เลือกป้ายที่เจาะจงที่สุด (ที่เจอก่อนชนะเมื่อเจาะจงเท่ากัน) อ่านค่าได้สี่แบบ:
    ค่าหลังป้าย (``เลขที่ : AB-0001``) · ค่าก่อนป้ายอังกฤษในช่องเดียวกัน (``ใบวางบิล : AB-0001 Invoice No`` — ป้ายไทย ค่า ป้ายอังกฤษ)
    · ช่องถัดไปในแถวเดียวกัน (``เลขที่ใบกำกับภาษี : Invoice No. | AB-0001`` เมื่อ OCR อ่านกล่องเป็นตาราง)
    · ป้ายเป็นหัวตาราง ค่าอยู่ช่องเดียวกันของแถวถัดไป (``เลขที่ | ออกเมื่อ`` → ``AB-0001 | ...``)
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


# เล่มที่ (บิลเล่ม) ค่าสั้นได้ ("เล่มที่ 5") จึงไม่บังคับความยาวขั้นต่ำเหมือนเลขที่เอกสาร แต่ต้องมีตัวเลข
BOOK_NO_LABEL_RE = re.compile(r"เล่มที่|book\s*no\.?", re.I)
BOOK_NO_VALUE_RE = re.compile(r"^[\s:/#.]*(?:no\.?)?[\s:#]*([A-Za-z0-9][A-Za-z0-9\-/]*)", re.I)


def _find_book_no(lines: list[str]) -> str | None:
    """เล่มที่จากป้าย ``เล่มที่`` / ``Book No.`` ค่าที่เจอก่อนชนะ อ่านได้สามแบบเหมือนเลขที่เอกสาร:
    ค่าหลังป้ายในช่องเดียวกัน (``เล่มที่ 12 เลขที่ 0345``) · ช่องถัดไปในแถวเดียวกัน · ป้ายเป็นหัวตาราง ค่าอยู่ช่องเดียวกันของแถวถัดไป
    สองแบบหลังใช้เมื่อหลังป้ายในช่องนั้นไม่มีตัวเลข (``เล่มที่ เลขที่ 345`` ไม่หยิบ 345 เป็นเล่มที่)."""
    for i, line in enumerate(lines):
        cells = line.split(" | ")
        nxt = lines[i + 1].split(" | ") if i + 1 < len(lines) else []
        for j, cell in enumerate(cells):
            m = BOOK_NO_LABEL_RE.search(cell)
            if not m:
                continue
            after = cell[m.end():]
            val = _doc_no_value((BOOK_NO_VALUE_RE.match(after) or [None, None])[1])
            bare = not re.search(r"\d", after)
            if val is None and bare and j + 1 < len(cells) and not BOOK_NO_LABEL_RE.search(cells[j + 1]):
                beside = BOOK_NO_VALUE_RE.match(cells[j + 1])
                val = _doc_no_value(beside and beside.group(1))
            if val is None and bare and len(nxt) == len(cells) > 1:
                head = BOOK_NO_VALUE_RE.match(nxt[j])
                val = _doc_no_value(head and head.group(1))
            if val is not None:
                return val
    return None


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


def extract_fields(lines: list[str], raw: str | None = None, kinds: list[tuple[str, re.Pattern]] | None = None) -> InvoiceData:
    """``raw`` = ข้อความดิบจากตัวอ่าน ถ้ามีตาราง HTML รายการสินค้าอ่านจากตารางตามหัวคอลัมน์ (``item_table``) แทนทีละบรรทัด
    ``kinds`` = ชนิดเอกสารจาก ``load_kinds`` (ไม่ส่ง = ใช้ค่าจาก config)."""
    data = InvoiceData()
    full_text = "\n".join(lines)

    data.document_type = kind_of(default_kinds() if kinds is None else kinds, full_text)

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
        if HEAD_OFFICE_RE.search(line):
            party.branch = party.branch or "สำนักงานใหญ่"
        if (m := ISSUING_BRANCH_RE.search(line)) and not in_buyer:  # ป้ายสาขาที่ออกใบกำกับ ชนะคำว่าสำนักงานใหญ่ที่หัวกระดาษ
            data.vendor.branch = m.group(1) or "สำนักงานใหญ่"

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
    data.book_no = _find_book_no(lines)
    data.date = data.date or parse_date(full_text)
    if data.total_amount is None and data.vat_amount is None and data.discount is None:
        data.total_amount = data.subtotal  # เอกสารที่มีแค่ "รวม/Total" บรรทัดเดียว
    table = table_items(raw, MAX_SUBTOTAL_LABEL) if raw else None
    if table is not None:
        data.items = table
    elif item_start is not None:
        data.items = _extract_items(lines, item_start, summary_start or len(lines))
    return data

