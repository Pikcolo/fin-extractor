"""เลขภาษีที่ OCR อ่านได้ไม่ครบในบล็อกของฝ่ายหนึ่ง: ไม่เติมค่าลงฟิลด์ แต่ใส่คำใบ้ในเหตุผลของฟิลด์ · บล็อกที่ OCR อ่านซ้ำใช้อ่านเลขภาษีได้ (เลขที่ใช้ในเทสต์สร้างจากสูตร checksum)."""

from fin_extractor.config import load_config
from fin_extractor.doctypes import load_profiles
from fin_extractor.extraction.party_blocks import fill_party_blocks, tax_id_hints
from fin_extractor.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from fin_extractor.schema import InvoiceData
from fin_extractor.validation.evidence import build_field_evidence, review_from_evidence
from fin_extractor.validation.tax_id import is_valid_thai_tax_id

PROFILES = {p.name: p for p in load_profiles(load_config()["document_types"])}
WH = PROFILES["withholding_certificate"]
BLOCKS = WH.party_blocks


def make_id(first12: str) -> str:
    check = (11 - sum(int(d) * (13 - i) for i, d in enumerate(first12)) % 11) % 10
    return first12 + str(check)


COMPANY_ID = make_id("010000000001")   # นิติบุคคล: ขึ้นต้นด้วย 0
PERSON_ID = make_id("390000000001")    # บุคคลธรรมดา: ไม่ขึ้นต้นด้วย 0


def spaced(digits: str) -> str:
    return " ".join(digits)


FORM_HINT = "(ให้ระบุว่าเป็น บุคคล นิติบุคคล บริษัท สมาคม หรือคณะบุคคล)"  # ฟอร์มพิมพ์บรรทัดนี้ในทุกบล็อก (มีคำว่า บริษัท แม้ฝ่ายนั้นเป็นบุคคลธรรมดา)


def payee_block(name: str, digits: str) -> list[str]:
    return ["ผู้ถูกหักภาษี ณ ที่จ่าย : - เลขประจำตัวผู้เสียภาษีอากร (13หลัก)* " + spaced(digits), f"ชื่อ {name} เลขประจำตัวผู้เสียภาษีอากร",
            FORM_HINT, "ประเภทเงินได้พึงประเมินจ่าย"]


def hints_for(lines):
    d = InvoiceData()
    fill_party_blocks(lines, d, BLOCKS)
    return d, tax_id_hints(lines, d, BLOCKS)


def test_ids_made_for_the_tests_are_valid():
    assert is_valid_thai_tax_id(COMPANY_ID) and is_valid_thai_tax_id(PERSON_ID)


def test_a_12_digit_read_is_hinted_and_not_filled():
    digits = PERSON_ID[:5] + PERSON_ID[6:]  # หลุดหนึ่งหลักกลางตัว
    d, hints = hints_for(payee_block("นายสมชาย ตัวอย่าง", digits))
    assert d.vendor.tax_id is None
    assert digits in hints["vendor.tax_id"] and "12 หลัก" in hints["vendor.tax_id"] and "ไม่เติมเลขเอง" in hints["vendor.tax_id"]
    assert "เติม 0" not in hints["vendor.tax_id"]


def test_leading_zero_is_suggested_only_for_a_company_and_only_when_it_passes_the_checksum():
    dropped = COMPANY_ID[1:]  # ตัว 0 นำหน้าหาย
    _d, hints = hints_for(payee_block("บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่)", dropped))
    assert f"0{dropped}" in hints["vendor.tax_id"] and "เติม 0 นำหน้า" in hints["vendor.tax_id"]
    _d, hints = hints_for(payee_block("นายสมชาย ตัวอย่าง", dropped))  # ไม่ใช่นิติบุคคล ไม่เสนอ
    assert "เติม 0" not in hints["vendor.tax_id"]
    _d, hints = hints_for(payee_block("บริษัท ตัวอย่าง จำกัด", PERSON_ID[:5] + PERSON_ID[6:]))  # เติม 0 แล้วไม่ผ่าน checksum ไม่เสนอ
    assert "เติม 0" not in hints["vendor.tax_id"]


def test_no_hint_when_the_id_is_valid_or_absent():
    d, hints = hints_for(payee_block("บริษัท ตัวอย่าง จำกัด", COMPANY_ID))
    assert d.vendor.tax_id == COMPANY_ID and hints == {}
    _d, hints = hints_for(["ผู้ถูกหักภาษี ณ ที่จ่าย : - เลขประจำตัวผู้เสียภาษีอากร (13หลัก)", "ชื่อ นายสมชาย ตัวอย่าง", "ประเภทเงินได้พึงประเมินจ่าย"])
    assert hints == {}  # OCR ไม่มีตัวเลขเลย ไม่มีอะไรให้ใบ้


def test_a_human_entered_id_removes_the_hint():
    lines = payee_block("บริษัท ตัวอย่าง จำกัด", COMPANY_ID[1:])
    d, hints = hints_for(lines)
    assert "vendor.tax_id" in hints
    d.vendor.tax_id = COMPANY_ID
    assert tax_id_hints(lines, d, BLOCKS) == {}


def test_hint_becomes_a_reason_on_the_field_without_forcing_missing():
    ev = build_field_evidence(InvoiceData(), [], "", required=(), hints={"vendor.tax_id": "คำใบ้"})
    assert ev["vendor.tax_id"].level == "unverified"  # ไม่บังคับ: ไม่ใช่ missing ไม่เปลี่ยนสถานะ
    assert "คำใบ้" in review_from_evidence(ev, [])["vendor.tax_id"].reasons
    ev = build_field_evidence(InvoiceData(), [], "", required=("vendor.tax_id",), hints={"vendor.tax_id": "คำใบ้"})
    review = review_from_evidence(ev, [])["vendor.tax_id"]
    assert review.level == "missing" and "คำใบ้" in review.reasons  # บังคับและไม่มีค่า: ยัง missing พร้อมคำใบ้


# ---------- บล็อกที่ OCR อ่านซ้ำหลังหัวตาราง
def payer_block(name: str, digits: str) -> list[str]:
    return ["ผู้มีหน้าที่หักภาษี ณ ที่จ่าย : - เลขประจำตัวผู้เสียภาษีอากร (13หลัก)* " + spaced(digits), f"ชื่อ {name} เลขประจำตัวผู้เสียภาษีอากร"]


def test_a_repeated_block_label_line_supplies_the_id_the_first_block_lost():
    first = payer_block("บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่)", COMPANY_ID[:4] + COMPANY_ID[5:])  # 12 หลัก
    lines = [*first, "ประเภทเงินได้พึงประเมินจ่าย", *payer_block("บริษัท ชื่ออื่น จำกัด", COMPANY_ID), "1. เงินเดือน | 01/01/2026 | 100.00 | 3.00"]
    d = InvoiceData()
    fill_party_blocks(lines, d, BLOCKS)
    assert d.buyer.tax_id == COMPANY_ID
    assert d.buyer.name == "บริษัท ตัวอย่าง จำกัด"  # ชื่อมาจากบล็อกแรกเท่านั้น บล็อกซ้ำไม่ถูกอ่านชื่อ
    assert tax_id_hints(lines, d, BLOCKS) == {}


def test_a_valid_id_in_the_first_block_is_kept_over_a_later_one():
    lines = [*payer_block("บริษัท ตัวอย่าง จำกัด", COMPANY_ID), "ประเภทเงินได้พึงประเมินจ่าย", *payer_block("บริษัท ก จำกัด", PERSON_ID)]
    d = InvoiceData()
    fill_party_blocks(lines, d, BLOCKS)
    assert d.buyer.tax_id == COMPANY_ID


def test_llm_is_told_not_to_complete_a_short_id():
    assert "ห้ามเติมหลักเอง" in EXTRACTION_SYSTEM_PROMPT
