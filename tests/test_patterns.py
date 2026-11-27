from datetime import date
from decimal import Decimal

import pytest


from fin_extractor.extraction.patterns import find_amounts, find_tax_ids, parse_date
from fin_extractor.validation import is_valid_thai_tax_id

TODAY = date(2026, 1, 1)


@pytest.mark.parametrize("tax_id,ok", [
    ("0105559117128", True),
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


def test_repeated_column_header_is_dropped_only_at_the_start_of_a_marked_piece():
    from fin_extractor.extraction import lines_with_chunks

    md = ("<!-- s02 strip_02-03 1/2 -->\nItem ลำดับ\nArticle รายการ\n1 ปากกา 19.00\n\n"
          "<!-- s03 strip_02-03 2/2 repeat-header -->\nItem ลำดับ\nArticle รายการ\n5 แฟ้ม 82.00\nItem ลำดับ\n\n"
          "<!-- s04 strip_04 -->\nItem ลำดับ\n")
    pairs = lines_with_chunks(md)
    assert [ln for ln, c in pairs if c == "s03"] == ["5 แฟ้ม 82.00", "Item ลำดับ"]  # ตัดแค่หัวที่ซ้ำต้นท่อน หลังจากนั้นไม่แตะ
    assert [ln for ln, c in pairs if c == "s04"] == ["Item ลำดับ"]  # ท่อนที่ไม่มีธง: ไม่ตัด
    from fin_extractor.pipeline import FinanceOCRPipeline
    assert FinanceOCRPipeline.lines_from_raw(md) == [ln for ln, _c in pairs]  # บรรทัดของขั้นสกัดชุดเดียวกัน (ที่มาของบรรทัดยังตรง)
