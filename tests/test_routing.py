"""routing/relevance.py: router คัดภาพย่อยก่อน OCR (ไม่เรียก OCR/GPU: ส่งข้อความที่ "อ่านได้" เข้าไปตรง ๆ)."""

import pytest

from fin_extractor.routing import build_aliases, route, signals
from fin_extractor.routing import relevance as R

PAGE = (1000, 1400)


def sec(i, x, y, w, h, table=None, kind="cell"):
    return {"id": f"s{i:02d}", "kind": kind, "bbox": [x, y, w, h], "table": table}


def decide(sections, texts, contrib=None):
    return {d.id: d for d in route(sections, {f"s{i:02d}": t for i, t in texts.items()}, None, PAGE, contrib=contrib)}


LONG_PROSE = "1. เมื่อลูกค้าได้รับสินค้า กรุณาตรวจสอบสภาพสินค้าและจำนวนก่อนทุกครั้ง หากพบว่าสินค้าอยู่ในสภาพไม่พร้อมรับ กรุณาแจ้งกลับบริษัททันที"


def test_signals_use_the_extractor_vocabulary_and_value_patterns():
    g = signals("รวมทั้งสิ้น 3,975.00 บาท")
    assert "total_amount" in g["labels"] and g["values"] == ["amount"]
    assert signals("0905564002794")["values"] == ["tax_id"]
    assert signals("วันที่ 31/10/2026")["values"] == ["date"]


def test_html_table_markup_is_stripped_before_looking_for_values():
    assert "date" in signals('<table><tr><td>วันที่ :</td><td>31/10/2026</td></tr></table>')["values"]


def test_amount_pieces_in_split_cells_count_as_values_only_when_short():
    # 50 ทวิ แบ่งช่องบาท/สตางค์: "60" กับ ".00" คนละช่อง
    for piece in ("60", "00", ".00"):  # ยอดภาษี 60.00 ที่แตกเป็น "60" กับ ".00" ต้องไม่ถูกทิ้ง
        assert signals(piece)["values"] == ["number"], piece
    assert signals("5")["values"] == []  # เลขหลักเดียวไม่พอ
    assert signals("เลข 40 ในประโยคยาว " + "ก" * 80)["values"] == []  # ข้อความยาวที่มีเลขปนเป็นแค่ประโยค


def test_long_prose_without_values_or_fields_is_dropped_but_short_text_is_always_kept():
    secs = [sec(0, 0, 0, 900, 300), sec(1, 0, 300, 400, 60), sec(2, 0, 400, 900, 300)]
    d = decide(secs, {0: LONG_PROSE, 1: "อลกค้า กฤตวิทย์ กว้างสกุล", 2: "3,975.00"})
    assert not d["s00"].keep and d["s00"].role == "drop"      # เงื่อนไขท้ายใบ: มีคำว่า "ลูกค้า/สินค้า" แต่เป็นประโยค
    assert d["s01"].keep and d["s01"].role == "short"          # ชื่อที่ป้ายอ่านเพี้ยน ต้องไม่ถูกทิ้ง
    assert d["s02"].keep and d["s02"].role == "data"


def test_a_section_the_extractor_can_use_is_kept_even_if_long():
    secs = [sec(0, 0, 0, 900, 300)]
    d = decide(secs, {0: LONG_PROSE}, contrib={"s00": ["buyer.name"]})
    assert d["s00"].keep and d["s00"].role == "contrib"


def test_small_crops_with_long_text_are_kept_because_the_read_is_probably_garbled():
    small = sec(0, 0, 0, 120, 90)  # 10,800 px² < SMALL_AREA
    assert R.SMALL_AREA > 120 * 90
    d = decide([small], {0: LONG_PROSE})
    assert d["s00"].keep and d["s00"].role == "small"


def test_address_led_long_text_is_kept():
    d = decide([sec(0, 0, 0, 900, 300)], {0: "ที่อยู่ 49/12 ซอย 6 ถนนนิพัทธ์สงเคราะห์ 1 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา " + "ก" * 30})
    assert d["s00"].keep and d["s00"].role == "label"


def test_duplicate_sections_are_dropped_in_favour_of_the_first():
    a, b = sec(0, 0, 0, 500, 100), sec(1, 2, 1, 500, 100)
    d = decide([a, b], {0: "3,975.00", 1: "3,975.00"})
    assert d["s00"].keep and d["s01"].role == "duplicate" and not d["s01"].keep


def test_container_is_kept_as_residual_not_dropped():
    box = sec(0, 0, 0, 1000, 1000)
    kids = [sec(1, 100, 100, 300, 200), sec(2, 500, 100, 300, 200)]
    d = decide([box, *kids], {0: "ทั้งฟอร์ม", 1: "3,975.00", 2: "260.05"})
    assert d["s00"].keep and d["s00"].role == "residual" and "ส่วนย่อย" in d["s00"].reason
    assert d["s01"].role == "data" and d["s02"].role == "data"


def test_item_columns_under_a_short_item_header_are_kept_even_without_numbers():
    header = sec(0, 0, 0, 300, 60, table=0)
    body = sec(1, 0, 70, 300, 500, table=0)  # รายการสินค้า ไม่มีตัวเลข และยาวพอจะถูกตัดถ้าไม่มีกฎนี้
    d = decide([header, body], {0: "รายละเอียด Description", 1: "Micro SD Card 256GB SANDISK ULTRA SDSQUAC-256G-GN6MN (150MB) " + "ก" * 40})
    assert d["s00"].keep and d["s01"].keep and d["s01"].role == "companion"


def test_build_aliases_reads_core_and_document_type_fields():
    al = dict(build_aliases({"core_labels": [{"key": "buyer.name", "aliases": ["นามลูกค้า"]}],
                             "document_types": {"x": {"fields": [{"key": "f", "aliases": ["เลขที่"]}]}}}))
    assert al["buyer.name"] == ["นามลูกค้า"] and al["x:f"] == ["เลขที่"]


@pytest.mark.parametrize("n", [0, 1])
def test_route_handles_trivial_inputs(n):
    secs = [sec(i, 0, 100 * i, 100, 50) for i in range(n)]
    assert len(route(secs, {}, None, PAGE)) == n
