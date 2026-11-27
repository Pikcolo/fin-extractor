"""ชื่อลูกค้า (buyer.name) จากป้าย — ใช้บรรทัด OCR รูปแบบเดียวกับที่ OCR ให้จริง

ชื่อบุคคลและบริษัทเป็นค่าสมมติ (โครงบรรทัด ป้ายสองภาษาติดกัน คำนำหน้าที่ OCR ต่อมา และป้ายหลอก คงตามของจริง)
"""

from types import SimpleNamespace

import pytest
from test_extraction_validation import BITKUB_LINES

from fin_extractor.config import load_config
from fin_extractor.doctypes import (
    CORE_LABEL_KEYS, GENERIC_PROFILE, FieldSpec, core_labels_for, load_core_labels, load_profiles)
from fin_extractor.extraction import extract_fields
from fin_extractor.extraction.core_labels import extract_party_name, split_branch
from fin_extractor.extraction.llm_extractor import normalize_llm_output
from fin_extractor.schema import InvoiceData

CFG = load_config()
DEFAULTS = load_core_labels(CFG["core_labels"])
PROFILES = {p.name: p for p in load_profiles(CFG["document_types"])}


def buyer(lines, profile=GENERIC_PROFILE):
    return extract_party_name(lines, core_labels_for(profile, DEFAULTS), "buyer.name")


# ---------- รูปแบบป้ายที่เห็นจริง (ค่าตั้งต้น) ----------

@pytest.mark.parametrize("lines,expected", [
    # ค่าอยู่หน้าป้ายอื่นในบรรทัดเดียวกัน ตัดที่ "วันที่"
    (["ชื่อลูกค้า บริษัท ตัวอย่าง จำกัด วันที่Issue Date : 05/10/2569 Salesmanพนักงานขาย", "Customer Name"],
     ("บริษัท ตัวอย่าง จำกัด", None)),
    # ป้ายสองภาษาติดกัน, OCR "นามลูกคา", ต่อด้วยป้ายถัดไป, สาขาในวงเล็บ
    (["Customer codeนามลูกคา : บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่) Invoice No.เลขที่ใบกำกับภาษี"],
     ("บริษัท ตัวอย่าง จำกัด", "สำนักงานใหญ่")),
    # ป้ายสะกดเพี้ยน ไม่มีตัวคั่น
    (["ซื่อลูกค้าสมชาย ทดสอบนามสกุล"], ("สมชาย ทดสอบนามสกุล", None)),
    # คำนำหน้า "Address" ติดมา และเลขที่บ้านติดท้ายชื่อ
    (["Nameท่อยู่Codeนามลูกค้าAddress บริษัท ตัวอย่าง จำกัด49/12 ซอย 6 ถนน ก"], ("บริษัท ตัวอย่าง จำกัด", None)),
    # "Name" นำหน้า, "จำกัต" (OCR) ติดเลขที่บ้าน ห้ามพึ่งการหา "จำกัด" ว่าชื่อจบตรงไหน
    (["ที่อยู่Addressนามลูกค้าName บริษัท ตัวอย่าง จำกัต49/12 ชออ 6"], ("บริษัท ตัวอย่าง จำกัต", None)),
    # เลขภาษี 13 หลักตามหลังชื่อทันที
    (["Codeนามลูกค้าบริษัท ตัวอย่าง จำกัด 0105561234501 เลยที่ใบกำกับOrder No. ST6910"], ("บริษัท ตัวอย่าง จำกัด", None)),
    # ตัวคั่นตารางจาก markdown/HTML ของตัวอ่าน (Typhoon/OpenThai)
    (["นามลูกค้า : บริษัท ตัวอย่าง จำกัด | เลขที่ใบกำกับภาษี : A1"], ("บริษัท ตัวอย่าง จำกัด", None)),
])
def test_buyer_name_from_observed_label_shapes(lines, expected):
    assert buyer(lines) == expected


@pytest.mark.parametrize("lines", [
    ["บริษัท ผู้ขาย จำกัด สำหรับลูกค้า / For Customer"],        # ป้ายระบุสำเนา ไม่ใช่ป้ายชื่อ
    ["รหัสลูกค้า C00964 เลขประจำตัวผู้เสียภาษี วันที่ 15/10/2569"],  # รหัสลูกค้า
    ["อัตราราคา 1-10 บิล ค่าบริการ 1,000 บาท ราคาสำหรับลูกค้าพิเศษ"],  # บรรทัดรายการ
    ["SONGKHLA 90110 Customer Tax ID.:"],                        # ไม่มีค่า
    ["Shipper:- Consignee:-", "SOMECO LTD. TESTCO LTD."],        # สองชื่อในบรรทัดเดียว แยกคอลัมน์ไม่ได้
    ["Customer Name", "เลขที่ผู้เสียภาษี0-1055-61234-50-1 (สำนักงานใหญ่)"],  # ป้ายล้วน บรรทัดถัดไปเป็นป้ายอื่น
    ["นามลูกค้า : C00964"],                                       # ค่าเป็นรหัส
    ["นามลูกค้า : 0105561234501"],                                # ค่าเป็นเลขภาษี
    ["นามลูกค้า : "],
    BITKUB_LINES,
])
def test_no_buyer_name_when_label_is_a_decoy_or_value_is_not_a_name(lines):
    assert buyer(lines) == (None, None)


def test_earliest_line_with_a_valid_value_wins():
    lines = ["ชื่อลูกค้า : ", "ชื่อลูกค้า บริษัท ก จำกัด", "นามลูกค้า : บริษัท ข จำกัด"]
    assert buyer(lines)[0] == "บริษัท ก จำกัด"


def test_trailing_period_of_an_abbreviation_is_kept():
    assert buyer(["นามลูกค้า : IMPORT TEST CO., LTD."])[0] == "IMPORT TEST CO., LTD."


# ---------- ต่อชนิดเอกสาร ----------

def test_next_line_is_not_used_unless_the_label_asks_for_it():
    assert buyer(["ชื่อลูกค้า", "บริษัท ตัวอย่าง จำกัด"]) == (None, None)


def test_customs_and_voucher_labels():
    assert buyer(["ชื่อผู้น่าของเข้า/ผู้ส่งของออก IMPORT TEST CO., LTD.", "C/O ตัวอย่างขนส่ง"],
                 PROFILES["customs_receipt"]) == ("IMPORT TEST CO., LTD.", None)
    pv = PROFILES["payment_voucher"]
    assert buyer(["ด ป๒ได้รับเงินจาก บริษัท ตัวอย่าง จำกัด (ผู้ซื้อ/ผู้รับบริการ) ดังรายการต่อไปนี้"], pv)[0] == "บริษัท ตัวอย่าง จำกัด"
    assert buyer(["1 ป๒ได้รับเงินจาก (ผู้ซื้อ/ผู้รับบริการ) ดังรายการต่อไปนี้ป 9"], pv) == (None, None)       # ชื่อเป็นลายมือ ไม่มีค่า


def test_profile_labels_override_defaults_at_the_same_key_only():
    pv = PROFILES["payment_voucher"]
    keys = [(f.key, f.aliases) for f in core_labels_for(pv, DEFAULTS)]
    assert keys == [("buyer.name", ("ได้รับเงินจาก",)), ("vendor.name", ("ข้าพเจ้า",))]  # ของโปรไฟล์ทับค่าตั้งต้น (ไม่รวมกัน)
    assert core_labels_for(GENERIC_PROFILE, DEFAULTS) == DEFAULTS


def test_payment_voucher_vendor_is_the_name_after_kha_pha_chao_001322():
    """ใบสำคัญรับเงิน: ผู้ขายเขียนชื่อหลัง "ข้าพเจ้า" คำกำกับ (ผู้ขายสินค้า/ให้บริการ) ไม่ใช่ส่วนของชื่อ."""
    lines = ["ใบสำคัญรับเงิน", "ข้าพเจ้า อห์นดา ศิลปโชติ (ผู้ขายสินค้า/ให้บริการ)",
             "ได้รับเงินจาก บริษัท ฮิมพอร์ต รถทร์ จำกัด (ผู้ซื้อ/ผู้รับบริการ) คังรายการต่อไปนี้"]
    specs = core_labels_for(PROFILES["payment_voucher"], DEFAULTS)
    assert extract_party_name(lines, specs, "vendor.name") == ("อห์นดา ศิลปโชติ", None)
    assert extract_party_name(lines, specs, "buyer.name")[0] == "บริษัท ฮิมพอร์ต รถทร์ จำกัด"


def test_unsupported_core_label_key_is_rejected_at_load():
    assert CORE_LABEL_KEYS == ("buyer.name", "vendor.name")
    with pytest.raises(ValueError, match="ยังไม่รองรับ"):
        load_core_labels([{"key": "invoice_no", "aliases": ["เลขที่"]}])


# ---------- branch และการรวมกับ LLM ----------

def test_split_branch():
    assert split_branch("บริษัท ก จำกัด (สำนักงานใหญ่)") == ("บริษัท ก จำกัด", "สำนักงานใหญ่")
    assert split_branch("A Co. (Head Office)") == ("A Co.", "Head Office")
    assert split_branch("บริษัท ก จำกัด") == ("บริษัท ก จำกัด", None)


def test_llm_buyer_branch_is_split_like_rules_so_sources_agree():
    out = normalize_llm_output({"buyer": {"name": "บริษัท ก จำกัด (สำนักงานใหญ่)"}})
    assert out["buyer"]["name"] == "บริษัท ก จำกัด" and out["buyer"]["branch"] == "สำนักงานใหญ่"
    kept = normalize_llm_output({"buyer": {"name": "บริษัท ก จำกัด (สำนักงานใหญ่)", "branch": "00001"}})
    assert kept["buyer"]["branch"] == "00001"  # ไม่ทับสาขาที่ LLM ตอบเอง


# ---------- pipeline: เติมเฉพาะที่ว่าง ไม่แตะ required/สถานะ ----------

def _pipe():
    from fin_extractor.pipeline import FinanceOCRPipeline
    from fin_extractor.extraction.rule_extractor import default_kinds

    p = object.__new__(FinanceOCRPipeline)
    p.kinds = default_kinds()
    p.cfg = {"validation": {"amount_tolerance": 0.02, "default_vat_rate": 0.07}}
    p.llm, p.profiles, p.core_labels = None, list(PROFILES.values()), DEFAULTS
    p.reader = SimpleNamespace(name="test")  # engine_name ใช้ตอนสร้างผล
    return p


def test_pipeline_fills_buyer_name_and_branch_only_when_empty():
    p = _pipe()
    lines = ["นามลูกค้า : บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่) Invoice No.A1"]
    data, sources, _ = p.extract_traced(lines, "")
    assert data.buyer.name == "บริษัท ตัวอย่าง จำกัด" and data.buyer.branch == "สำนักงานใหญ่"
    assert sources["buyer.name"] == "rules"
    # ถ้า rules เดิมได้ชื่อไว้แล้ว ไม่ทับ (ตรรกะเดิมไม่เคยกรอก buyer.name แต่กันไว้เผื่อวันหน้า)
    assert extract_fields(lines).buyer.name is None


def test_buyer_name_never_changes_required_fields_or_status():
    p = _pipe()
    base = ["ใบกำกับภาษี", "บริษัท ผู้ขาย จำกัด", "รวมทั้งสิ้น 107.00"]
    r1 = p.build_result("x", 1, base, "", None, [])
    r2 = p.build_result("x", 1, [*base, "ชื่อลูกค้า บริษัท ตัวอย่าง จำกัด"], "", None, [])
    assert r2.data.buyer.name == "บริษัท ตัวอย่าง จำกัด" and r1.data.buyer.name is None
    assert (r1.status, r1.missing_fields, r1.checks) == (r2.status, r2.missing_fields, r2.checks)
    assert "buyer.name" in r2.review and r2.review["buyer.name"].level != "conflict"  # ค่าอยู่ในข้อความที่อ่านได้ (ไม่ขัดแย้ง)


def test_field_spec_next_line_flag_round_trips():
    spec = load_core_labels([{"key": "buyer.name", "aliases": ["x"], "next_line": True}])[0]
    assert isinstance(spec, FieldSpec) and spec.next_line and spec.type == "text"
    assert InvoiceData().buyer.name is None


# ---------- ป้ายสำเนา "สำหรับลูกค้า / For Customer" ไม่ใช่การเริ่มส่วนผู้ซื้อ ----------

COPY_LINES = [
    "บริษัท ผู้ขายตัวอย่าง จำกัด สำหรับลูกค้า / For Customer",
    "เลขประจำตัวผู้เสียภาษี 0105559117128 สำนักงานใหญ่",
    "Customer codeนามลูกค้า : บริษัท ผู้ซื้อตัวอย่าง จำกัด Invoice No.เลขที่ใบกำกับภาษี : A1",
    "เลขประจำตัวผู้เสียภาษี :0105561234501",
]


def test_copy_mark_does_not_start_the_buyer_section():
    d = extract_fields(COPY_LINES)
    assert d.vendor.name == "บริษัท ผู้ขายตัวอย่าง จำกัด"       # ป้ายสำเนาไม่ติดท้ายชื่อ
    assert d.vendor.tax_id == "0105559117128" and d.vendor.branch == "สำนักงานใหญ่"
    assert d.buyer.tax_id == "0105561234501"                    # เลขหลังป้าย "นามลูกค้า" เป็นของผู้ซื้อ


def test_real_buyer_label_still_starts_the_buyer_section():
    d = extract_fields(["บริษัท ผู้ขาย จำกัด", "ลูกค้า : บริษัท ผู้ซื้อ จำกัด", "เลขประจำตัวผู้เสียภาษี 0105561234501"])
    assert d.vendor.name == "บริษัท ผู้ขาย จำกัด" and d.buyer.tax_id == "0105561234501" and d.vendor.tax_id is None


# ---------- ข้อความแบบตารางจาก Typhoon (markdown → markdown_to_lines) ----------

@pytest.mark.parametrize("lines,expected", [
    # ป้ายไทย : ป้ายอังกฤษ | ค่า | ป้ายถัดไป  — ห้ามได้ "Customer name" เป็นชื่อ
    (["นามลูกค้า : Customer name | บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่) | เลขที่ใบกำกับภาษี : A1"],
     ("บริษัท ตัวอย่าง จำกัด", "สำนักงานใหญ่")),
    # ป้ายไทยอยู่บรรทัดเดียว ค่าอยู่บรรทัดถัดไปหลังป้ายอังกฤษ
    (["ชื่อลูกค้า", "Customer Name: บริษัท ตัวอย่าง จำกัด"], ("บริษัท ตัวอย่าง จำกัด", None)),
    # "Name | ค่า"
    (["นามลูกค้า", "Name | บริษัท ตัวอย่าง จำกัด"], ("บริษัท ตัวอย่าง จำกัด", None)),
])
def test_buyer_name_from_typhoon_table_shapes(lines, expected):
    assert buyer(lines) == expected


@pytest.mark.parametrize("lines", [
    ["นามลูกค้า : Customer name"],                          # มีแต่ป้ายอังกฤษ
    ["นามลูกค้า : Customer name | Address | เลขที่ 1"],      # เซลล์ป้ายล้วนสองเซลล์แล้วเป็นเลขที่/ป้ายอื่น
    ["ชื่อลูกค้า", "บริษัท อะไรก็ได้ จำกัด"],               # บรรทัดถัดไปไม่ขึ้นด้วยป้ายกำกับชื่อ ไม่เดา
    ["ชื่อลูกค้า", "Name |"],                              # ป้ายกำกับแต่ไม่มีค่า
])
def test_table_shapes_that_must_not_yield_a_name(lines):
    assert buyer(lines) == (None, None)
