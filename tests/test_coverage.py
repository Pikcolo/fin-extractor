"""ชั้นตรวจความครบ (sections/coverage.py): ชี้ส่วนที่ตัวอ่านหลักอ่านไม่ครบ แนบข้อความเสริมท้าย .md และบันทึกที่มาของค่าที่ได้จากข้อความเสริม
ข้อความแต่ละกรณีมาจากผลจริงของ OpenThai (หลัก) และ OpenThai Text (เสริม)."""

import json

from fin_extractor.config import load_config
from fin_extractor.extraction.rule_extractor import lines_with_chunks, supplement_of
from fin_extractor.pipeline import FinanceOCRPipeline
from fin_extractor.sections.coverage import missing_values, normalize_digits
from fin_extractor.sections.read import markdown_of, write_markdown

VLM_1317 = "<table><tr><td>รหัสลูกค้า Code</td><td>C00964</td></tr><tr><td>นามลูกค้า Name</td><td>บริษัท อิมพอร์ต สตาร์ จำกัด</td></tr></table>"
TEXT_1317 = "รหัสลูกค้า CO0964 เลขประจำตัวผู้เสียภาษี วันที่ 15/10/2569\nนามลูกค้า บริษัท อิมพอร์ตสตารั จำกัด 0905564002794 เลขที่ใบกำกับ ST6910-O036"


def test_zero_read_as_letter_o_next_to_digits_is_fixed():
    assert normalize_digits("Total 2,O00.O0 · เลขที่ใบกำกับ ST6910-O036 · CO0964") == "Total 2,000.00 · เลขที่ใบกำกับ ST6910-0036 · C00964"


def test_dates_and_amounts_only_the_supplement_reader_saw_flag_the_section():
    assert missing_values(VLM_1317, TEXT_1317) == {"money": [], "dates": ["2026-10-15"], "codes": ["0905564002794", "ST69100036"]}
    assert missing_values("ภาษีมูลค่าเพิ่ม 7% ยอดสุทธิ 788.90", "ภาษีมูลค่าเพิ่ม 7% 51.61\nยอดสุทธิ 788.90")["money"] == ["51.61"]
    assert missing_values("ยอดสุทธิ 788.90", "ยอดสุทธิ 788.90") == {"money": [], "dates": [], "codes": []}


def test_long_codes_compare_lookalike_letters_and_skip_plain_words():
    """1308 s01: เลขที่ใบกำกับที่ตัวอ่านหลักทำหาย · รหัสเดียวกันที่อ่าน O/0 ต่างกันไม่นับ · คำยาวที่ไม่มีตัวเลข (COLLECTOR) ไม่ใช่รหัส."""
    m = missing_values("Order No. SO26040857", "Invoice No. BAUNS00054022610000155 Order No. S026040857 COLLECTOR")
    assert m["codes"] == ["BAUNS00054022610000155"]


def test_split_column_falls_between_two_boxes_and_ignores_rules():
    import numpy as np
    from fin_extractor.sections.coverage import split_column
    from fin_extractor.sections.regions.run import RegionConfig

    img = np.full((200, 1000, 3), 255, np.uint8)
    img[40:60, 50:400] = 0      # ข้อความกล่องซ้าย
    img[40:60, 600:950] = 0     # ข้อความกล่องขวา
    img[10:190, 498:501] = 0    # เส้นคั่นกล่อง (เส้นตั้ง ไม่นับเป็นเนื้อหา)
    x = split_column(img, RegionConfig().cfg)
    assert 400 < x < 600
    img[:, 400:600] = 0          # ไม่มีแนวว่าง = ไม่แยก
    assert split_column(img, RegionConfig().cfg) is None


def test_supplement_block_goes_after_every_main_block_and_is_marked():
    doc = {"sections": [{"id": "s00", "kind": "strip_00", "text": "บริษัท อี พาวเวอร์ เซอร์วิส จำกัด"},
                        {"id": "s01", "kind": "strip_01", "text": VLM_1317, "supplement": {"engine": "openthai-text", "text": "วันที่ 15/10/2569"}},
                        {"id": "s02", "kind": "strip_02", "text": "รายการ"}]}
    md = markdown_of(doc)
    assert md.index("<!-- s02") < md.index("<!-- s01 strip_01 supplement openthai-text -->")
    rows = list(zip(lines_with_chunks(md), supplement_of(md), strict=True))
    assert rows[-1] == (("วันที่ 15/10/2569", "s01"), "openthai-text")
    assert all(s is None for (_ln, c), s in rows[:-1])


def test_write_markdown_attaches_only_flagged_sections(tmp_path):
    src = tmp_path / "x_p1.sections.json"
    src.write_text(json.dumps({"source": "x", "sections": [{"id": "s00", "kind": "k", "bbox": [0, 0, 1, 1], "text": "ยอดสุทธิ 788.90"},
                                                          {"id": "s01", "kind": "k", "bbox": [0, 0, 1, 1], "text": VLM_1317}]}), encoding="utf-8")
    (tmp_path / "x_p1.coverage.json").write_text(json.dumps({"engine": "openthai-text", "flagged": ["s01"], "sections": [
        {"id": "s00", "text": "ยอดสุทธิ 788.90"}, {"id": "s01", "text": TEXT_1317}]}), encoding="utf-8")
    md = write_markdown(src).read_text(encoding="utf-8")
    assert md.count("supplement") == 1 and "ST6910-0036" in md


def test_values_found_only_in_the_supplement_are_attributed_and_need_review():
    md = markdown_of({"sections": [
        {"id": "s00", "kind": "k", "text": "บริษัท อี พาวเวอร์ เซอร์วิส จำกัด\nใบกำกับภาษี"},
        {"id": "s01", "kind": "k", "text": VLM_1317, "supplement": {"engine": "openthai-text", "text": normalize_digits(TEXT_1317)}},
        {"id": "s02", "kind": "k", "text": "รวมทั้งสิ้น 100.00"}]})
    pipe = FinanceOCRPipeline(load_config(), engine="openthai", use_llm=False, with_reader=False)
    res = pipe.build_result("x", 1, pipe.lines_from_raw(md), md, None, [])
    assert res.data.invoice_no == "ST6910-0036"
    assert res.review["invoice_no"].source == "openthai-text" and res.review["date"].source == "openthai-text"
    assert any("ตัวอ่านเสริม" in r for r in res.review["invoice_no"].reasons)
    assert res.review["vendor.name"].source is None  # ค่าที่ตัวอ่านหลักให้ ยังเป็นของ rules
    assert res.status != "auto_approved"


def test_word_coverage_marks_words_the_primary_reader_missed_1306():
    from fin_extractor.sections.coverage import word_coverage

    lines = [{"text": "จำนวน รวมเป็นเงิน 2,O00.O0", "box": [[0, 0], [900, 0], [900, 40], [0, 40]],
              "words": [{"text": "จำนวน", "box": [[0, 0], [100, 0], [100, 40], [0, 40]]},
                        {"text": "รวมเป็นเงิน", "box": [[500, 0], [700, 0], [700, 40], [500, 40]]},
                        {"text": "2,O00.O0", "box": [[750, 0], [900, 0], [900, 40], [750, 40]]},
                        {"text": "7", "box": [[950, 0], [960, 0], [960, 40], [950, 40]]}]}]
    w = word_coverage(lines, "จำนวน เงิน Amount สองพันบาทถ้วน")
    assert [x["found"] for x in w] == [True, False, False, None]
    assert w[1]["box"] == [500, 0, 200, 40]
    assert [x["found"] for x in word_coverage(lines, "จำนวน | รวมเป็นเงิน Subtotal | 2,000.00")][:3] == [True, True, True]
