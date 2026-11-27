"""check ใหม่: จำนวนเงินตัวอักษร ↔ ตัวเลข และ 50 ทวิ ยอดรวม = ผลรวมแถวรายการ."""

from decimal import Decimal

import pytest

from fin_extractor.schema import InvoiceData
from fin_extractor.validation.checks import amount_in_words, income_rows_sum
from fin_extractor.validation.thai_amount import words_to_amount


@pytest.mark.parametrize("text, value", [
    ("(สามพันหนึ่งร้อยแปดสิบเจ็ดบาทห้าสิบสตางค์)", "3187.50"),
    ("(หกสิบบาทถ้วน)", "60.00"),                                
    ("เจ็ดพันเก้าบาทสามสิบห้าสตางค์", "7009.35"),                   # ถ้า OCR อ่านถูก
    ("หนึ่งแสนหกพันสองร้อยห้าสิบบาทถ้วน", "106250.00"),
    ("สิบเอ็ดบาท", "11.00"), ("ยี่สิบเอ็ดล้านบาทถ้วน", "21000000.00"), ("หนึ่งร้อยบาทห้าสตางค์", "100.05"),
])
def test_words_to_amount(text, value):
    assert words_to_amount(text) == Decimal(value)


@pytest.mark.parametrize("text", ["(หกสืบข้าพถ้วน)", "หนึ่งและสามพันทุกสิบของบาทน้ำสืบสถานต์", "บาท", "สามพัน", "", None])
def test_unreadable_words_are_none_not_a_guess(text):
    assert words_to_amount(text) is None


PAIR = [("document_fields.total_tax_withheld_text", "document_fields.total_tax_withheld")]


def test_amount_in_words_pass_fail_and_unreadable():
    ok = amount_in_words(InvoiceData(document_fields={"total_tax_withheld_text": "(หกสิบบาทถ้วน)", "total_tax_withheld": "60.00"}), PAIR)
    assert ok.passed is True and ok.fields == [PAIR[0][0], PAIR[0][1]]
    bad = amount_in_words(InvoiceData(document_fields={"total_tax_withheld_text": "สามพันหนึ่งร้อยแปดสิบเจ็ดบาทห้าสิบสตางค์",
                                                       "total_tax_withheld": "3187.00"}), PAIR)  # template เคยอ่านผิดเป็น 3,187.00
    assert bad.passed is False
    unreadable = amount_in_words(InvoiceData(document_fields={"total_tax_withheld_text": "(หกสืบข้าพถ้วน)", "total_tax_withheld": "60.00"}), PAIR)
    assert unreadable.passed is None and unreadable.fields == []
    assert amount_in_words(InvoiceData(), ()).passed is None


def test_income_rows_sum_uses_dated_two_amount_rows_once():
    lines = ["1. เงินเดือน ค่าจ้าง ฯลฯ ตามมาตรา 40 (1) | 12/10/2026 | 106,250.00 | 3,187.50",
             "1. เงินเดือน ค่าจ้าง ฯลฯ ตามมาตรา 40 (1) | 12/10/2026 | 106,250.00 | 3,187.50",  # ข้อความเสริมอ่านแถวเดิมซ้ำ
             "รวมเงินที่จ่ายและภาษีที่หักนำส่ง | 106,250.00 | 3,187.50"]
    d = InvoiceData(document_fields={"total_paid": "106250.00", "total_tax_withheld": "3187.50"})
    assert income_rows_sum(lines, d).passed is True
    assert income_rows_sum(lines, InvoiceData(document_fields={"total_paid": "106250.00", "total_tax_withheld": "3187.00"})).passed is False
    missing = income_rows_sum(["8 ต.ค. 69 | 233,644.86 | 7,009.35"], InvoiceData())  # แถวรวมหาย
    assert missing.passed is None and "233644.86" in missing.detail
    assert income_rows_sum(["ไม่มีแถว"], d).passed is None
