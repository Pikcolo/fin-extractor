"""50 ทวิ: ผู้มีหน้าที่หักภาษี (ผู้จ่าย) = buyer · ผู้ถูกหักภาษี (ผู้รับเงิน) = vendor อ่านจากบล็อกป้ายของแต่ละฝ่าย ไม่ใช่ตามตำแหน่ง."""

import pytest

from fin_extractor.config import load_config
from fin_extractor.doctypes import PartyBlocks, load_profiles
from fin_extractor.extraction.llm_extractor import profile_schema
from fin_extractor.extraction.party_blocks import fill_party_blocks, split_blocks
from fin_extractor.schema import InvoiceData

CFG = load_config()
PROFILES = {p.name: p for p in load_profiles(CFG["document_types"])}
WH = PROFILES["withholding_certificate"]
BLOCKS = WH.party_blocks

TAX_PAYER = "0 9 0 5 5 6 4 0 0 2 7 9 4"  # 0905564002794 ผ่าน checksum
TAX_12 = "3 9 0 0 3 0 2 4 7 0 9 1"        # OCR ทำเลขหลุดเป็น 12 หลัก (001307)

# รูปแบบเดียวกับ 001307: ป้ายบล็อกพร้อมเลขภาษี แล้วบรรทัดชื่อติดป้าย "ชื่อ" ตามด้วยจุดไข่ปลาและป้ายถัดไป
SAMPLE = [
    "ฉบับที่ 1 (สำหรับผู้ถูกหักภาษี ณ ที่จ่าย ใช้แนบพร้อมกับแบบแสดงรายการภาษี)",
    "หนังสือรับรองการหักภาษี ณ ที่จ่าย",
    f"ผู้มีหน้าที่หักภาษี ณ ที่จ่าย : - เลขประจำตัวผู้เสียภาษีอากร (13หลัก)\* {TAX_PAYER}",
    "ชื่อ บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่) ... เลขประจำตัวผู้เสียภาษีอากร ... (ให้ระบุว่าเป็น บุคคล นิติบุคคล บริษัท)",
    "(ให้ระบุ ชื่ออาคาร/หมู่บ้าน ห้องเลขที่ ชั้นที่ เลขที่ ตรอก/ซอย)",
    f"ผู้ถูกหักภาษี ณ ที่จ่าย : - เลขประจำตัวผู้เสียภาษีอากร (13หลัก)* {TAX_12}",
    "ชื่อ เพอร์เฟคทดสอบ (สมหญิง ทดลอง) เลขประจำตัวผู้เสียภาษีอากร",
    "ประเภทเงินได้พึงประเมินจ่าย | วัน เดือนหรือปีภาษี ที่จ่าย | จำนวนเงินที่จ่าย | ภาษีที่หักและนำส่งไว้",
    "ชื่อ ไม่ใช่ชื่อของฝ่ายใด 0 1 0 5 5 5 9 1 1 7 1 2 8",  # หลังหัวตาราง: อยู่นอกบล็อก
]


def filled(lines, **given) -> InvoiceData:
    d = InvoiceData(**given)
    fill_party_blocks(lines, d, BLOCKS)
    return d


def test_payer_is_buyer_and_payee_is_vendor():
    d = filled(SAMPLE)
    assert (d.buyer.name, d.buyer.branch, d.buyer.tax_id) == ("บริษัท ตัวอย่าง จำกัด", "สำนักงานใหญ่", "0905564002794")
    assert (d.vendor.name, d.vendor.branch) == ("เพอร์เฟคทดสอบ (สมหญิง ทดลอง)", None)  # จุดไข่ปลาท้ายชื่อถูกตัด


def test_tax_id_that_fails_checksum_stays_empty_instead_of_being_guessed():
    assert filled(SAMPLE).vendor.tax_id is None  # 12 หลัก


def test_nothing_after_the_table_header_belongs_to_a_party():
    got = split_blocks(SAMPLE, BLOCKS)
    assert not any("ไม่ใช่ชื่อของฝ่ายใด" in ln for block in got.values() for ln in block)


def test_name_on_the_label_line_and_on_the_line_below_the_label():
    same = ["ผู้ถูกหักภาษี ณ ที่จ่าย :-ชือซี บริษัท ตัวอย่าง จำกัด (สำนักงานใหญ่)(ให้ระบุว่าเป็น บุคคล นิติบุคคล บริษัท)"]  # ชื่อติดป้ายบรรทัดเดียว
    d = filled(same)
    assert (d.vendor.name, d.vendor.branch) == ("บริษัท ตัวอย่าง จำกัด", "สำนักงานใหญ่")
    below = ["ผู้มีหน้าที่หักภาษี ณ ที่จ่าย :-", "ชื่อ", "บริษัท โกลเรียซ ทดสอบ จำกัด", "เลขประจำตัวผู้เสียภาษีอากร", "0-90-5-560-00217-0",
             "(ให้ระบุว่าเป็น บุคคล นิติบุคคล บริษัท)", "ประเภทเงินได้พึงประเมินที่จ่าย"]  # 001311: ป้ายอยู่ลำพัง ค่าอยู่บรรทัดถัดไป
    d = filled(below)
    assert (d.buyer.name, d.buyer.tax_id) == ("บริษัท โกลเรียซ ทดสอบ จำกัด", "0905560002170")


def test_form_hint_lines_are_never_taken_as_a_name():
    lines = ["ผู้ถูกหักภาษี ณ ที่จ่าย : -", "(ให้ระบุ ชื่ออาคาร/หมู่บ้าน ห้องเลขที่ ชั้นที่)", "ประเภทเงินได้พึงประเมินจ่าย"]
    assert filled(lines).vendor.name is None


def test_copy_notice_lines_do_not_start_a_block():
    notice = ["ฉบับที่ 1 (สำหรับผู้ถูกหักภาษี ณ ที่จ่าย ใช้แนบพร้อมกับแบบแสดงรายการภาษี)", "ชื่อ ไม่ใช่ชื่อ"]
    d = filled(notice)
    assert d.vendor.name is None and d.buyer.name is None


def test_values_the_positional_rules_put_on_the_wrong_side_are_reset_not_kept():
    # 001323: เลขภาษีของผู้ถูกหักไปอยู่ในช่องผู้หักเพราะ rules ตามตำแหน่ง ต้องถูกล้างแล้วอ่านใหม่จากบล็อก
    d = filled(SAMPLE, vendor={"name": "ชื่อผิดฝั่ง", "tax_id": "0105559117128", "address": "ที่อยู่ผิดฝั่ง"},
               buyer={"name": "ชื่อผิดฝั่ง 2"})
    assert d.vendor.tax_id is None and d.vendor.address is None and d.buyer.name == "บริษัท ตัวอย่าง จำกัด"


def test_missing_block_leaves_that_party_empty():
    d = filled(["ผู้ถูกหักภาษี ณ ที่จ่าย : -", "ชื่อ นายทดสอบ ตัวอย่าง"], buyer={"name": "x"})
    assert d.vendor.name == "นายทดสอบ ตัวอย่าง" and d.buyer.name is None


def test_only_the_withholding_profile_uses_blocks():
    assert [n for n, p in PROFILES.items() if p.party_blocks] == ["withholding_certificate"]


def test_llm_schema_states_what_vendor_and_buyer_mean_for_this_profile():
    props = profile_schema(WH)["properties"]
    assert "ผู้ถูกหักภาษี" in props["vendor"]["description"] and "ผู้ขาย" in props["vendor"]["description"]
    assert "ผู้มีหน้าที่หักภาษี" in props["buyer"]["description"]
    assert "description" not in profile_schema(PROFILES["payment_voucher"])["properties"]["vendor"]  # ชนิดอื่นไม่เปลี่ยน


def test_required_party_is_the_withholder_as_before_the_swap():
    assert "buyer.name" in WH.required and "buyer.tax_id" in WH.required and "vendor.name" not in WH.required


@pytest.mark.parametrize("bad", [{"payer": ["x"]}, {"vendor": ()}])
def test_blocks_reject_unknown_party_or_empty_aliases(bad):
    with pytest.raises(ValueError):
        PartyBlocks(parties=bad)
