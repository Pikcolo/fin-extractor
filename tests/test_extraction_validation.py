"""ทดสอบ rules + validation ด้วยข้อความที่ถอดจาก data/raw/BITKUB-001.jpg."""

from datetime import date
from decimal import Decimal

from fin_extractor.extraction import extract_fields, markdown_to_lines
from fin_extractor.validation import decide_status, run_checks

BITKUB_LINES = """ใบเสร็จรับเงิน / ใบกำกับภาษี
RECEIPT / TAX INVOICE
บริษัท บิทคับ ออนไลน์ จำกัด (สำนักงานใหญ่)
Bitkub Online Co., Ltd (Head Office)
เลขประจำตัวผู้เสียภาษี / Tax ID: 0105559117128
โทร / Tel. 02-032-9555
ลูกค้า / Customer เลขที่ / No.
วันที่ / Date
ลำดับ / No. รายการ / Description จำนวนเงิน (บาท) / Amount (THB)
1 ค่าธรรมเนียมการถอนเงินบาท ณ วันที่ 2023-05-18 20.00
Total THB withdrawal fee on 2023-05-18
2 ค่าธรรมเนียมการขายสินทรัพย์ดิจิทัล ณ วันที่ 2023-05-18 8.05
3 ค่าธรรมเนียมการซื้อสินทรัพย์ดิจิทัล ณ วันที่ 2023-05-18 10.02
4 ค่าธรรมเนียมการถอนสินทรัพย์ดิจิทัล ณ วันที่ 2023-05-18 5.29
5 Fee credit ที่ได้รับจากการแลกเหรียญ KUB ณ วันที่ 2023-05-18 4.62
หมายเหตุ / Remark
รวม / Total 47.98
ส่วนลด / Discount 0.42
มูลค่ารวมภาษีมูลค่าเพิ่ม / Grand Total (VAT Included) 47.56
มูลค่าก่อนรวมภาษีมูลค่าเพิ่ม / Grand Total (VAT Excluded) 44.45
ภาษีมูลค่าเพิ่ม 7% / VAT 7% 3.11""".splitlines()


def test_rule_extractor_bitkub():
    d = extract_fields(BITKUB_LINES)
    assert d.document_type == "tax_invoice"
    assert d.vendor.name.startswith("บริษัท บิทคับ")
    assert d.vendor.tax_id == "0105559117128"
    assert d.vendor.branch == "สำนักงานใหญ่"
    assert d.invoice_no is None  # ช่องเลขที่ถูกปิดไว้ในภาพ
    assert (d.subtotal, d.discount, d.total_amount) == (Decimal("47.98"), Decimal("0.42"), Decimal("47.56"))
    assert (d.amount_before_vat, d.vat_amount, d.vat_rate) == (Decimal("44.45"), Decimal("3.11"), Decimal("0.07"))
    assert [it.amount for it in d.items] == [Decimal("20.00"), Decimal("8.05"), Decimal("10.02"), Decimal("5.29"), Decimal("4.62")]
    assert d.date == date(2023, 5, 18)


def test_checks_pass_and_catch_ocr_error():
    d = extract_fields(BITKUB_LINES)
    checks = {c.name: c.passed for c in run_checks(d)}
    assert checks["vendor_tax_id_checksum"]
    assert checks["items_sum"] and checks["discount"]
    assert checks["before_vat_plus_vat"] and checks["vat_rate"]

    # จำลอง OCR อ่าน 44.45 เป็น 44.46 → cross-check ต้องจับได้
    d.amount_before_vat = Decimal("44.46")
    d.vat_amount = Decimal("3.17")
    checks = {c.name: c.passed for c in run_checks(d)}
    assert checks["before_vat_plus_vat"] is False
    _, status = decide_status(d, run_checks(d))
    assert status == "check"


def test_markdown_table_to_lines():
    md = "<table><tr><td>1</td><td>ค่าธรรมเนียม</td><td>20.00</td></tr></table>\n**รวม / Total** 47.98"
    assert markdown_to_lines(md) == ["1 | ค่าธรรมเนียม | 20.00", "รวม / Total 47.98"]


def test_fill_missing_keeps_primary_values():
    from fin_extractor.pipeline import fill_missing
    from fin_extractor.schema import InvoiceData

    primary = InvoiceData(total_amount=Decimal("47.56"), vendor={"name": "A"})
    secondary = InvoiceData(total_amount=Decimal("1.0"), vat_amount=Decimal("3.11"), vendor={"name": "B", "tax_id": "0105559117128"})
    merged = fill_missing(primary, secondary)
    assert merged.total_amount == Decimal("47.56") and merged.vat_amount == Decimal("3.11")
    assert merged.vendor.name == "A" and merged.vendor.tax_id == "0105559117128"


# ใบแจ้งหนี้อังกฤษที่เรียง Subtotal → VAT → TOTAL (ข้อความจาก VATGB-001.pdf)
VATGB_LINES = """Severn Industrial Supply Ltd TAX INVOICE
VAT Reg No: GB 933 6364 83
Invoice Number: F-26-6724
Date: 9 September 2025
Description Qty Unit Price VAT % Amount Excl
Air filter element, heavy duty 6 884.37 20% 5,306.22
Subtotal £16,681.31
VAT 20% £3,336.26
TOTAL £20,017.57""".splitlines()


def test_bare_total_after_vat_is_the_grand_total():
    """"TOTAL" ที่มาหลัง Subtotal กับ VAT คือยอดรวม ไม่ใช่ยอดย่อยซ้ำ

    บั๊กจริง: เดิมจัด "Total" เป็นยอดย่อยเสมอ บรรทัด TOTAL จึงถูกทิ้ง เพราะช่องยอดย่อยมีค่าแล้ว
    เอกสารได้ failed (ไม่มียอดรวม) ทั้งที่ OCR อ่านได้ครบทุก engine
    """
    d = extract_fields(VATGB_LINES)
    assert d.subtotal == Decimal("16681.31")
    assert d.vat_amount == Decimal("3336.26")
    assert d.total_amount == Decimal("20017.57")


def test_bare_total_before_vat_is_still_the_subtotal():
    """"รวม / Total" ที่มาก่อนส่วนลดและ VAT ยังเป็นยอดย่อยเหมือนเดิม (ผังของ BITKUB)"""
    d = extract_fields(BITKUB_LINES)
    assert d.subtotal == Decimal("47.98")
    assert d.total_amount == Decimal("47.56")


def test_status_comes_from_missing_fields_and_checks():
    d = extract_fields(BITKUB_LINES)  # ช่องเลขที่เอกสารว่างในข้อความนี้
    missing, status = decide_status(d, run_checks(d))
    assert missing == ["invoice_no"] and status == "check"

    d.invoice_no = "INV-001"
    missing, status = decide_status(d, run_checks(d))
    assert missing == [] and status == "ready"

    # field ครบแต่ไม่มี check ไหนยืนยันตัวเลขได้ → ยังไม่อนุมัติอัตโนมัติ
    _, status = decide_status(d, [])
    assert status == "check"

    d.total_amount = None
    _, status = decide_status(d, run_checks(d))
    assert status == "incomplete"


def test_handwritten_field_needs_corroboration_to_auto_approve():
    d = extract_fields(BITKUB_LINES)
    d.invoice_no = "INV-001"
    checks = run_checks(d)
    assert decide_status(d, checks)[1] == "ready"

    # ยอดรวมจากลายมือ + ผลรวมลงตัว → ยังอนุมัติได้
    assert decide_status(d, checks, ["total_amount"])[1] == "ready"
    # เลขที่เอกสารจากลายมือ ไม่มี check ไหนยืนยันได้ → ต้องคนตรวจ
    assert decide_status(d, checks, ["invoice_no"])[1] == "check"

    # ยอดรวมจากลายมือแต่ผลรวมไม่ลงตัว → ต้องคนตรวจ
    d.total_amount += 1
    assert decide_status(d, run_checks(d), ["total_amount"])[1] == "check"


def test_markdown_to_lines_joins_decimals_split_by_a_space():
    """ฟอร์ม 50 ทวิแบ่งช่องบาท/สตางค์ OpenThai จึงพิมพ์ "2,000 .00" ตัวจับยอดเงินต้องเห็นเป็น 2,000.00 (เคยเป็นกฎที่ไม่ทำงานเพราะมีอักขระ backspace ติดอยู่ใน regex)"""
    md = '<td>รวมเงินที่จ่ายและภาษีที่หักนำส่ง</td><td colspan="1" rowspan="1">2,000 .00</td><td>60 .00</td></tr>'
    assert markdown_to_lines(md) == ["รวมเงินที่จ่ายและภาษีที่หักนำส่ง | 2,000.00 | 60.00"]
    assert markdown_to_lines("106,250  \u00a0.00") == ["106,250.00"]  # ช่องว่างหลายตัว / nbsp ในบรรทัดเดียวกันก็นับ
    assert markdown_to_lines("1 .25\n.50") == ["1.25", ".50"]  # "1 .25" รวมเป็น 1.25 แต่ ".50" ของบรรทัดถัดไปต้องไม่ถูกดึงมาต่อ (เดิม \s+ ข้ามบรรทัดจนได้ 1.25.50)


def test_markdown_to_lines_leaves_other_dots_alone():
    assert markdown_to_lines("ข้อ 5 .500 กรัม") == ["ข้อ 5 .500 กรัม"]  # ไม่ใช่ทศนิยมสองหลัก
    assert markdown_to_lines("มาตรา 40 (4)(ข) ... .00") == ["มาตรา 40 (4)(ข) ... .00"]  # ไม่มีตัวเลขนำหน้าจุด


def test_address_goes_to_the_party_named_on_the_nearest_name_line_above():
    from fin_extractor.extraction.rule_extractor import assign_addresses
    from fin_extractor.schema import InvoiceData

    lines = ["ชื่อลูกค้า Customer Name | บริษัท อิมพอร์ต สตาร์ จำกัด | วันที่ : 05/10/2569",
             "ที่อยุ่ Address | 49/12 ซอย 6 ถนน นิพัทธ์สงเคราะห์ 1 ต.หาดใหญ่ | สำนักงานใหญ่",
             "Address | อ.หาดใหญ่ จ.สงขลา 90110"]
    d = InvoiceData(buyer={"name": "บริษัท อิมพอร์ต สตาร์ จำกัด"})
    assign_addresses(lines, d)
    assert d.buyer.address == "49/12 ซอย 6 ถนน นิพัทธ์สงเคราะห์ 1 ต.หาดใหญ่\nอ.หาดใหญ่ จ.สงขลา 90110" and d.vendor.address is None


def test_address_is_left_empty_when_the_name_line_above_matches_no_party():
    from fin_extractor.extraction.rule_extractor import assign_addresses
    from fin_extractor.schema import InvoiceData

    # 1310: บล็อกผู้ซื้อไม่มีป้าย "ลูกค้า" และ rules ไม่ได้ชื่อผู้ซื้อ → ไม่เดาฝ่าย
    lines = ["บริษัท ทีเอ็นแอส หาดใหญ่ จำกัด", "ชื่อ : บริษัท อิมพอร์ต สตาร์ จำกัด",
             "ที่อยู่ : 49/12 ซอย 6 ต.หาดใหญ่ จ.สงขลา 90110 Tel.0808317128 | รหัสพนักงานขาย : HY01"]
    d = InvoiceData(vendor={"name": "บริษัท ทีเอ็นแอส หาดใหญ่ จำกัด"})
    assign_addresses(lines, d)
    assert d.vendor.address is None and d.buyer.address is None
    # คำแนะนำของฟอร์ม "(ให้ระบุ ...)" ไม่ใช่ค่า
    d2 = InvoiceData(vendor={"name": "บริษัท อิมพอร์ต สตาร์ จำกัด"})
    assign_addresses(["ชื่อ บริษัท อิมพอร์ต สตาร์ จำกัด", "ที่อยู่:... (ให้ระบุ ชื่ออาคาร/หมู่บ้าน ห้องเลขที่ 1)"], d2)
    assert d2.vendor.address is None



def test_book_no_from_label():
    assert extract_fields(["ใบกำกับภาษี", "เล่มที่ 12 เลขที่ 0345"]).book_no == "12"
    assert extract_fields(["Book No. : 7 | Invoice No. : INV-0345"]).book_no == "7"
    assert extract_fields(["เล่มที่ | 5 | เลขที่ | 0345"]).book_no == "5"  # ค่าอยู่ช่องถัดไป
    assert extract_fields(["เล่มที่ | เลขที่", "5 | 0345"]).book_no == "5"  # ป้ายเป็นหัวตาราง
    d = extract_fields(["เล่มที่ 12 เลขที่ 0345"])
    assert d.invoice_no == "0345"  # เล่มที่ไม่ทำให้เลขที่เอกสารเพี้ยน


def test_book_no_not_guessed():
    assert extract_fields(["เล่มที่ เลขที่ 0345"]).book_no is None  # ช่องเล่มที่ว่าง ไม่หยิบเลขที่มาแทน
    assert extract_fields(["เล่มที่ 01/02/2566"]).book_no is None  # วันที่ ไม่ใช่เล่มที่
    assert extract_fields(BITKUB_LINES).book_no is None

