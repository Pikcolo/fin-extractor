from datetime import date
from decimal import Decimal

import pytest


from fin_extractor.extraction.patterns import find_amounts, find_tax_ids, parse_date
from fin_extractor.validation import is_valid_thai_tax_id

TODAY = date(2026, 1, 1)


@pytest.mark.parametrize("tax_id,ok", [
    ("0105559117128", True),   # Bitkub
    ("0105559117123", False),  # หลักสุดท้ายผิด
    ("010555911712", False),   # 12 หลัก
])
def test_tax_id_checksum(tax_id, ok):
    assert is_valid_thai_tax_id(tax_id) is ok


def test_find_tax_ids_with_separators():
    text = "เลขประจำตัวผู้เสียภาษี / Tax ID: 0-1055-59117-12-8 โทร 02-032-9555"
    assert find_tax_ids(text) == ["0105559117128"]


def test_find_amounts_ignores_percent_and_index():
    assert find_amounts("ภาษีมูลค่าเพิ่ม 7% / VAT 7% 3.11") == [Decimal("3.11")]
    assert find_amounts("1 Total THB withdrawal fee 1,234.50") == [Decimal("1234.50")]


@pytest.mark.parametrize("text,expected", [
    ("วันที่ 2023-05-18", date(2023, 5, 18)),
    ("วันที่ 18/05/2566", date(2023, 5, 18)),
    ("18/05/66", date(2023, 5, 18)),
    ("18 พฤษภาคม 2566", date(2023, 5, 18)),
    ("18 พ.ค. 66", date(2023, 5, 18)),
    ("Date: 18 May 2023", date(2023, 5, 18)),
    ("ไม่มีวันที่", None),
])
def test_parse_date(text, expected):
    assert parse_date(text, today=TODAY) == expected
