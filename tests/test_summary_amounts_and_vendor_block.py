"""ยอดไม่เสียภาษี/ฐาน VAT · สาขา · ที่อยู่ผู้ขายใต้ชื่อบริษัท · ตัดบทบาทผู้ซื้อ/ผู้ขายที่ปนกัน (ข้อความตัวอย่างแต่งขึ้นตามรูปแบบที่ OCR ให้จริง)."""

from decimal import Decimal

from fin_extractor.config import load_config
from fin_extractor.doctypes import load_profiles
from fin_extractor.extraction import extract_fields, markdown_to_lines
from fin_extractor.extraction.core_labels import extract_party_name
from fin_extractor.extraction.llm_extractor import normalize_llm_output
from fin_extractor.extraction.rule_extractor import derive_total, fill_vendor_address, strip_contact, unmix_parties
from fin_extractor.schema import InvoiceData, LineItem
from fin_extractor.validation import run_checks
from fin_extractor.validation.thai_amount import words_to_amount

BS = chr(92)
NL = chr(10)
D = Decimal
PROFILES = {p.name: p for p in load_profiles(load_config()["document_types"])}


def checks(d):
    return {c.name: c.passed for c in run_checks(d)}


# ---------- markdown escape
def test_backslash_before_punctuation_is_removed_but_not_before_a_table_pipe():
    assert markdown_to_lines("7" + BS + "% VAT | 14.00") == ["7% VAT | 14.00"]
    assert markdown_to_lines("(13หลัก)" + BS + "* 1 2") == ["(13หลัก)* 1 2"]
    assert markdown_to_lines("a " + BS + "| b") == ["a " + BS + "| b"]


# ---------- ยอดไม่เสียภาษี + ฐาน VAT
SUMMARY = ["Non-Taxable Charges | 100.00", "Taxable Charges | 200.00", "7% VAT | 14.00", "TOTAL Amount Due (Baht) | 314.00"]


def test_taxable_is_the_vat_base_and_non_taxable_is_its_own_field():
    d = extract_fields(SUMMARY)
    assert (d.amount_before_vat, d.non_taxable_amount, d.vat_amount, d.total_amount) == (D("200.00"), D("100.00"), D("14.00"), D("314.00"))
    c = checks(d)
    assert c["vat_rate"] is True and c["before_vat_plus_vat"] is True  # VAT = 7% ของส่วนที่เสียภาษีเท่านั้น · ยอดรวม = ฐาน + VAT + ไม่เสียภาษี


def test_total_must_include_the_non_taxable_part():
    d = extract_fields(SUMMARY)
    d.total_amount = D("214.00")  # ลืมบวกยอดไม่เสียภาษี
    assert checks(d)["before_vat_plus_vat"] is False


def test_items_sum_matches_taxable_plus_non_taxable():
    d = extract_fields(SUMMARY)
    d.items = [LineItem(description="ค่าบริการ ก", amount=D("100.00")), LineItem(description="ค่าบริการ ข", amount=D("200.00"))]
    assert checks(d)["items_sum"] is True
    d.items[1].amount = D("250.00")
    assert checks(d)["items_sum"] is False


def test_exempt_value_line_with_zero_amount_is_kept_as_zero():
    d = extract_fields(["รวมเงิน | 107.00", "มูลค่าสินค้าที่ยกเว้นภาษีมูลค่าเพิ่ม | 0.00", "มูลค่าสินค้าก่อนคิดภาษีมูลค่าเพิ่ม | 100.00",
                        "ภาษีมูลค่าเพิ่ม 7% | 7.00", "ยอดสุทธิ | 107.00"])
    assert (d.non_taxable_amount, d.amount_before_vat, d.vat_amount, d.total_amount) == (D("0.00"), D("100.00"), D("7.00"), D("107.00"))
    assert checks(d)["before_vat_plus_vat"] is True


def test_total_is_derived_only_when_the_figures_agree():
    d = InvoiceData(subtotal=D("107.00"), amount_before_vat=D("100.00"), vat_amount=D("7.00"))
    derive_total(d, D("0.02"))
    assert d.total_amount == D("107.00")
    d2 = InvoiceData(subtotal=D("110.00"), amount_before_vat=D("100.00"), vat_amount=D("7.00"))
    derive_total(d2, D("0.02"))
    assert d2.total_amount is None  # ไม่ตรงกัน ไม่เดา
    d3 = InvoiceData(subtotal=D("107.00"), amount_before_vat=D("100.00"), vat_amount=D("7.00"), total_amount=D("999.00"))
    derive_total(d3, D("0.02"))
    assert d3.total_amount == D("999.00")  # มียอดรวมอยู่แล้ว ไม่ทับ


# ---------- สาขา
def test_issuing_branch_label_beats_the_head_office_mention_in_the_header():
    d = extract_fields(["บริษัท ตัวอย่าง จำกัด", "สำนักงานใหญ่ 1 หมู่ที่ 2", "สาขาที่ออกใบกำกับภาษี : สาขาที่ 00012 เลขที่ 5"])
    assert d.vendor.branch == "00012"
    d = extract_fields(["บริษัท ตัวอย่าง จำกัด", "สาขาที่ออกใบกำกับภาษีคือ : สำนักงานใหญ่ (00000)"])
    assert d.vendor.branch == "สำนักงานใหญ่"


def test_head_office_spelled_wrongly_by_ocr_is_still_head_office():
    for spelled in ("ลำนักงานใหญ่", "ลำนักงานโหญ่", "สำนักงานใหญ่", "Head Office"):
        d = extract_fields(["บริษัท ตัวอย่าง จำกัด", f"เลขประจำตัวผู้เสียภาษี 0105559117128 {spelled}"])
        assert d.vendor.branch == "สำนักงานใหญ่", spelled


# ---------- ที่อยู่ผู้ขายใต้ชื่อ
def vendor_address(lines):
    d = extract_fields(lines)
    fill_vendor_address(lines, d)
    return d.vendor.address


def test_address_under_the_name_skips_the_echoed_name_and_the_copy_notice_and_drops_the_english_twin():
    lines = ["บริษัท ตัวอย่าง จำกัด", "Example Co.,Ltd. (ต้นฉบับ)", "สำหรับลูกค้า / For Customer",
             "สำนักงานใหญ่ 12 หมู่ที่ 3 ตำบลทดสอบ อำเภอเมือง จังหวัดตัวอย่าง", "Head Office 12 Moo 3, Test, Muang 11111", "Tel. 02-000-0000"]
    assert vendor_address(lines) == "12 หมู่ที่ 3 ตำบลทดสอบ อำเภอเมือง จังหวัดตัวอย่าง"


def test_address_over_several_lines_stops_at_the_phone():
    lines = ["บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่)", "37/1 หมู่ที่ 5 ตำบลทดสอบ", "อำเภอเมือง จังหวัดตัวอย่าง", "รหัสไปรษณีย์ 11111 โทร. 02-000-0000",
             "เลขประจำตัวผู้เสียภาษี 0105559117128"]
    assert vendor_address(lines) == "37/1 หมู่ที่ 5 ตำบลทดสอบ" + NL + "อำเภอเมือง จังหวัดตัวอย่าง" + NL + "รหัสไปรษณีย์ 11111"


def test_no_address_when_the_line_under_the_name_is_not_one():
    assert vendor_address(["บริษัท ตัวอย่าง จำกัด", "Tel. 02-000-0000", "12 หมู่ที่ 3"]) is None
    assert vendor_address(["บริษัท ตัวอย่าง จำกัด", "เลขประจำตัวผู้เสียภาษี 0105559117128", "12 หมู่ที่ 3"]) is None


def test_a_labelled_address_is_not_overwritten():
    lines = ["บริษัท ตัวอย่าง จำกัด", "12 หมู่ที่ 3 ตำบลทดสอบ"]
    d = extract_fields(lines)
    d.vendor.address = "ที่อยู่จากป้าย"
    fill_vendor_address(lines, d)
    assert d.vendor.address == "ที่อยู่จากป้าย"


def test_contact_is_cut_from_rule_and_llm_addresses_alike():
    assert strip_contact("49/12 ซอย 6 90110 Tel.0800000000") == "49/12 ซอย 6 90110"
    out = normalize_llm_output({"buyer": {"name": "x", "address": "49/12 ซอย 6 90110 Tel.0800000000"}})
    assert out["buyer"]["address"] == "49/12 ซอย 6 90110"


# ---------- ผู้ซื้อ/ผู้ขายที่ปนกัน
def test_vendor_by_position_that_is_the_labelled_buyer_moves_to_the_buyer_side():
    d = InvoiceData(vendor={"name": "ชื่อผู้นำเข้า X CO., LTD.", "tax_id": "0105559117128", "branch": "สำนักงานใหญ่"}, buyer={"name": "X CO., LTD."})
    unmix_parties(d)
    assert d.vendor.name is None and d.vendor.tax_id is None
    assert (d.buyer.name, d.buyer.tax_id, d.buyer.branch) == ("X CO., LTD.", "0105559117128", "สำนักงานใหญ่")


def test_different_parties_are_left_alone():
    d = InvoiceData(vendor={"name": "บริษัท ก จำกัด", "tax_id": "0105559117128"}, buyer={"name": "บริษัท ข จำกัด"})
    unmix_parties(d)
    assert d.vendor.name == "บริษัท ก จำกัด" and d.buyer.tax_id is None


# ---------- ใบเสร็จศุลกากร
def test_customs_importer_label_is_read_even_when_ocr_changes_the_prefix():
    specs = PROFILES["customs_receipt"].core_labels
    for prefix in ("ชื่อ", "ซื่อ"):
        assert extract_party_name([f"{prefix}ผู้นำของเข้า/ผู้ส่งของออก X CO., LTD."], specs, "buyer.name")[0] == "X CO., LTD."


def test_customs_total_is_checked_against_the_amount_in_words():
    prof = PROFILES["customs_receipt"]
    assert "amount_in_words" in prof.checks and prof.words == (("document_fields.total_charges_text", "document_fields.total_charges"),)


def test_satang_misread_by_ocr_is_still_read():
    assert words_to_amount("(หนึ่งร้อยบาทสิบหกสตางค์.)") == D("100.16")
    assert words_to_amount("(หนึ่งร้อยบาทสิบหกสดางค์.)") == D("100.16")


def test_schema_has_the_non_taxable_field():
    assert "non_taxable_amount" in InvoiceData.model_fields
