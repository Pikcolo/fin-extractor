"""ตัวแบ่ง section ของผังแบบเดิม (sections/layout.py): ตารางต้องเป็นก้อนเดียวเป็นค่าตั้งต้น ไม่แยกหัวคอลัมน์ออกจากแถว."""

import cv2
import numpy as np
import pytest

from fin_extractor.preprocessing import binarize as bin_ops
from fin_extractor.sections import layout


def _page() -> np.ndarray:
    """หน้าขาว 800×600 มีข้อความหัว ตาราง 3 คอลัมน์ × 3 แถว (หัว + 2 แถว) และข้อความท้าย."""
    img = np.full((800, 600), 255, np.uint8)
    cv2.putText(img, "INVOICE HEADER", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    x0, y0, w, h = 40, 150, 520, 300
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), 0, 2)
    for i in (1, 2):
        cv2.line(img, (x0, y0 + i * h // 3), (x0 + w, y0 + i * h // 3), 0, 2)
        cv2.line(img, (x0 + i * w // 3, y0), (x0 + i * w // 3, y0 + h), 0, 2)
    for r, row in enumerate([("Qty", "Unit price", "Amount"), ("1", "2000", "2000"), ("2", "500", "1000")]):
        for c, text in enumerate(row):
            cv2.putText(img, text, (x0 + 15 + c * w // 3, y0 + 60 + r * h // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "TOTAL 3000", (40, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    return img


def test_table_is_one_section_by_default():
    secs = layout.layout_sections(_page(), table_mode="whole", max_piece_frac=None)
    tables = [s for s in secs if s["kind"] == "table"]
    assert len(tables) == 1 and not any(s["kind"] == "cell" for s in secs)
    x, y, w, h = tables[0]["bbox"]
    assert y <= 150 and y + h >= 450  # ครอบทั้งตาราง (หัวคอลัมน์ + แถว)
    assert tables[0]["keep"]


def test_cells_mode_splits_table_into_cells():
    secs = layout.layout_sections(_page(), table_mode="cells")
    assert sum(s["kind"] == "cell" for s in secs) >= 9  # 3×3 ช่อง


def test_text_outside_table_stays_separate_strips_in_reading_order():
    secs = [s for s in layout.layout_sections(_page(), table_mode="whole", max_piece_frac=None) if s["keep"]]  # ขอบว่างซ้าย/ขวาติดธง keep=False
    assert [s["kind"] for s in secs] == ["text", "table", "text"]
    assert [s["bbox"][1] for s in secs] == sorted(s["bbox"][1] for s in secs)  # ลำดับอ่านบนลงล่าง


def test_unknown_table_mode_rejected():
    with pytest.raises(ValueError):
        layout.layout_sections(_page(), table_mode="diagonal")


def test_drop_frames_removes_form_frame_but_keeps_plain_tables():
    frame, a, b = (10, 10, 500, 700), (30, 50, 400, 100), (30, 300, 400, 300)
    assert layout.drop_frames([frame, a, b]) == [a, b]  # กรอบครอบตารางสองก้อน
    assert layout.drop_frames([a, b]) == [a, b]  # ตารางที่ไม่มีอะไรข้างใน
    twin = (12, 12, 496, 696)
    assert layout.drop_frames([frame, twin]) == [twin]
    assert layout.drop_frames([frame, a]) == [a]  # ตารางที่มีตารางอื่นข้างในคือกรอบ (ซ้อนกันทุกคู่เป็นกรอบฟอร์ม)
    assert layout.drop_frames([a, (30, 400, 400, 100)]) == [a, (30, 400, 400, 100)]  # ตารางแยกกันไม่ถูกแตะ


def test_form_frame_around_two_tables_does_not_swallow_the_page():
    img = _page()
    cv2.rectangle(img, (40, 520), (560, 580), 0, 2)  # ตารางที่สอง (มีเส้นตั้ง)
    cv2.line(img, (300, 520), (300, 580), 0, 2)
    cv2.rectangle(img, (20, 120), (580, 620), 0, 2)  # กรอบฟอร์มครอบทั้งสองตาราง
    secs = layout.layout_sections(img, table_mode="whole", max_piece_frac=None)
    tables = [s for s in secs if s["kind"] == "table"]
    assert len(tables) == 2
    assert all(t["bbox"][3] < 400 for t in tables)  # ไม่มีก้อนไหนคลุมทั้งฟอร์ม






def test_text_beside_a_table_is_its_own_section_not_lost():
    """ป้ายยอดรวมอยู่ซ้ายของกล่องตัวเลข (เจอจริงในสแกน): ต้องมี section ข้อความของมัน ไม่ใช่ตกหล่น."""
    img = np.full((800, 700), 255, np.uint8)
    x0, y0, w, h = 400, 100, 250, 150  # ตารางชิดขวา
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), 0, 2)
    cv2.line(img, (x0, y0 + h // 2), (x0 + w, y0 + h // 2), 0, 2)
    cv2.line(img, (x0 + w // 2, y0), (x0 + w // 2, y0 + h), 0, 2)
    cv2.putText(img, "Subtotal", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)  # ป้ายอยู่ซ้ายของตาราง
    cv2.putText(img, "2000", (x0 + 20, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    secs = [s for s in layout.layout_sections(img, table_mode="whole") if s["keep"]]
    label = [s for s in secs if s["kind"] == "text" and s["bbox"][0] < 100 and s["bbox"][0] + s["bbox"][2] <= x0]
    assert label, secs
    lx, ly, lw, lh = label[0]["bbox"]
    assert ly <= 160 - 20 and ly + lh >= 160  # ครอบบรรทัดป้าย
    ink = layout.text_ink(bin_ops_otsu(img)) > 0
    covered = np.zeros_like(ink)
    for s in secs:
        x, y, w, h = s["bbox"]
        covered[y:y + h, x:x + w] = True
    assert not (ink & ~covered).any()  # หมึกทุกจุด (นอกเส้นตาราง) อยู่ใน section


def bin_ops_otsu(gray):
    from fin_extractor.preprocessing import binarize

    return binarize.otsu(gray)


def _tall_table_page() -> np.ndarray:
    """หน้า 1000 สูง: ตารางสูง 800 มี 10 แถวข้อความ (แถวละ 80) ห่างกันด้วยช่องว่าง."""
    img = np.full((1000, 600), 255, np.uint8)
    x0, y0, w, h = 30, 100, 540, 800
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), 0, 2)
    cv2.line(img, (x0 + w // 2, y0), (x0 + w // 2, y0 + h), 0, 2)
    for r in range(10):
        cv2.putText(img, f"row {r} item", (x0 + 20, y0 + 50 + r * 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    return img


def test_split_rows_cuts_in_gaps_between_text_lines_and_covers_the_box_exactly():
    img = _tall_table_page()
    ink = layout.text_ink(bin_ops.otsu(img))
    box = (30, 100, 540, 800)
    pieces = layout.split_rows(ink, box, max_h=500)
    assert len(pieces) == 2  # ceil(800 / 500)
    assert pieces[0][1] == 100 and pieces[-1][1] + pieces[-1][3] == 900  # คลุมทั้งกล่อง
    assert pieces[0][1] + pieces[0][3] == pieces[1][1]  # ติดกัน ไม่ซ้อน ไม่เว้น
    cut = pieces[1][1]
    assert not (ink[cut, 30:570] > 0).any()  # แถวที่ตัดไม่มีหมึก (ไม่ตัดกลางบรรทัด)
    assert layout.split_rows(ink, box, max_h=900) == [box]  # ไม่สูงเกินเกณฑ์ ไม่ตัด


def test_tall_table_is_split_into_row_groups_with_part_numbers():
    secs = [s for s in layout.layout_sections(_tall_table_page(), table_mode="whole", max_piece_frac=0.5) if s["keep"] and s["kind"] == "table"]
    assert len(secs) == 2 and [s["part"] for s in secs] == [1, 2] and all(s["parts"] == 2 for s in secs)
    assert all(s["bbox"][3] <= 500 for s in secs)  # ไม่สูงเกินครึ่งหน้า (1000 × 0.5)
    whole = [s for s in layout.layout_sections(_tall_table_page(), table_mode="whole", max_piece_frac=None) if s["keep"] and s["kind"] == "table"]
    assert len(whole) == 1 and "part" not in whole[0]


def test_run_sections_refuses_stale_or_missing_clean_images(tmp_path):
    import json

    from fin_extractor.sections import read as run_sections

    layout = [{"id": "s00", "kind": "text", "bbox": [0, 0, 10, 10], "keep": True},
              {"id": "s01", "kind": "text", "bbox": [0, 10, 10, 10], "keep": False}]
    with pytest.raises(SystemExit, match="--from images"):
        run_sections.clean_index(tmp_path, layout, "doc1")  # ไม่มี index.json
    (tmp_path / "index.json").write_text(json.dumps({"sections": [{"id": "s00", "bbox": [0, 0, 10, 10]}]}), encoding="utf-8")
    assert list(run_sections.clean_index(tmp_path, layout, "doc1")) == ["s00"]  # ตรงกับส่วนที่ keep
    (tmp_path / "index.json").write_text(json.dumps({"sections": [{"id": "s00", "bbox": [0, 0, 10, 99]}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="ไม่ตรงกับผัง"):
        run_sections.clean_index(tmp_path, layout, "doc1")  # สร้างจากผังเก่า


def test_split_rows_does_not_cut_through_text_when_no_gap_is_left():
    ink = np.ones((1000, 100), np.uint8)
    ink[595:605] = 0  # ช่องว่างเดียว
    pieces = layout.split_rows(ink, (0, 0, 100, 1000), 250)
    assert [p[3] for p in pieces] == [600, 400]  # ตัดที่ช่องว่างเดียวแล้วหยุด ไม่ตัดผ่ากลางหมึกที่แถว 500
    assert all(not ink[p[1], 50] for p in pieces[1:])
    assert layout.split_rows(np.ones((1000, 100), np.uint8), (0, 0, 100, 1000), 250) == [(0, 0, 100, 1000)]  # ไม่มีช่องว่างเลย = ไม่ตัด


def test_default_splits_a_table_taller_than_the_default_fraction():
    tall = [s for s in layout.layout_sections(_page(), table_mode="whole") if s["keep"] and s["kind"] == "table"]  # ตาราง 300/800 = 37.5% > MAX_PIECE_FRAC
    assert layout.MAX_PIECE_FRAC == 0.3 and len(tall) == 2 and [s["part"] for s in tall] == [1, 2]


def _totals_box_page() -> np.ndarray:
    """หน้าแบบหนึ่ง: ป้ายยอดรวมอยู่ซ้าย ตัวเลขอยู่ในตารางชิดขวา แถวละ 3 บรรทัด (ป้ายกับเส้นแนวนอนของตารางอยู่แถวเดียวกัน)."""
    img = np.full((600, 700), 255, np.uint8)
    cv2.putText(img, "HEADER", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    x0, y0, w, h = 450, 150, 220, 240
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), 0, 2)
    for i in range(1, 4):
        cv2.line(img, (x0, y0 + i * 60), (x0 + w, y0 + i * 60), 0, 2)
    cv2.line(img, (x0 + 150, y0), (x0 + 150, y0 + h), 0, 2)
    for r, (label, value) in enumerate([("Subtotal", "2000"), ("Discount", "0.00"), ("Total", "2000"), ("Paid", "1940")]):
        cv2.putText(img, label, (60, y0 + 40 + r * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        cv2.putText(img, value, (x0 + 20, y0 + 40 + r * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return img


def _ink_in(secs, img, x0, x1, y0, y1):
    """ส่วนที่เก็บทั้งหมดที่ครอบหมึกของกรอบ (x0..x1, y0..y1) แต่ละจุด."""
    ink = layout.text_ink(bin_ops.otsu(img)) > 0
    pts = np.argwhere(ink[y0:y1, x0:x1]) + [y0, x0]
    return [{s["id"] for s in secs if s["bbox"][0] <= x < s["bbox"][0] + s["bbox"][2] and s["bbox"][1] <= y < s["bbox"][1] + s["bbox"][3]}
            for y, x in pts[:: max(1, len(pts) // 200)]]


def test_bands_keep_a_label_and_its_value_in_the_same_full_width_section():
    img = _totals_box_page()
    secs = [s for s in layout.layout_sections(img, table_mode="bands", max_piece_frac=None) if s["keep"]]
    assert all(s["bbox"][0] == 0 and s["bbox"][2] == 700 for s in secs)  # เต็มความกว้างทุกส่วน
    label = _ink_in(secs, img, 55, 200, 170, 200)  # ป้าย "Subtotal"
    value = _ink_in(secs, img, 460, 590, 170, 200)  # ตัวเลขแถวเดียวกัน
    assert label and value
    assert set.intersection(*label) & set.intersection(*value)  # มีส่วนเดียวที่ครอบทั้งป้ายและค่า


def test_rows_mode_cuts_at_table_rules_only_where_the_whole_row_is_empty():
    img = _totals_box_page()
    rows = [s for s in layout.layout_sections(img, table_mode="rows", max_piece_frac=None) if s["keep"] and s["kind"] == "table"]
    bands = [s for s in layout.layout_sections(img, table_mode="bands", max_piece_frac=None) if s["keep"] and s["kind"] == "table"]
    assert len(bands) == 1 and len(rows) == 4  # 4 แถวป้าย/ค่า แต่ละแถวเป็นแถบเต็มความกว้าง
    assert [r["bbox"][1] for r in rows] == sorted(r["bbox"][1] for r in rows)
    ink = layout.text_ink(bin_ops.otsu(img)) > 0
    assert not any(ink[r["bbox"][1]].any() for r in rows[1:])  # รอยตัดไม่ผ่านหมึก


def test_rows_mode_refuses_a_cut_through_side_text():
    img = _totals_box_page()
    cv2.putText(img, "Side note", (60, 150 + 60 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)  # ข้อความซ้ายคร่อมเส้นแนวนอนเส้นแรก (y=210)
    rows = [s for s in layout.layout_sections(img, table_mode="rows", max_piece_frac=None) if s["keep"] and s["kind"] == "table"]
    assert 210 not in [r["bbox"][1] for r in rows]  # แถวที่ y=210 มีหมึกข้างตาราง ตัดไม่ได้


def test_default_mode_is_rows_with_full_width_bands():
    import inspect

    assert inspect.signature(layout.layout_sections).parameters["table_mode"].default == "rows"
    secs = [s for s in layout.layout_sections(_totals_box_page(), max_piece_frac=None) if s["keep"]]
    assert all(s["bbox"][0] == 0 and s["bbox"][2] == 700 for s in secs)  # ทุกส่วนเต็มความกว้าง


def test_chunks_map_each_md_chunk_to_its_section_image():
    from fin_extractor.sections import read as run_sections

    doc = {"source": "doc1", "engine": "e", "variant": "luma", "sections": [
        {"id": "s00", "kind": "text", "bbox": [0, 0, 10, 5], "text": "บริษัท ABC 123\nบรรทัดสอง"},
        {"id": "s01", "kind": "table", "bbox": [0, 5, 10, 5], "skipped": "blank"},
        {"id": "s02", "kind": "table", "bbox": [0, 10, 10, 5], "part": 1, "parts": 2, "text": "<table><td>รวม 5.00</td></table>", "truncated": True}]}
    chunks_doc = run_sections.chunks_of(doc, lambda s: f"{s['id']}.png")
    md = run_sections.markdown_of(doc).split("\n")
    assert [(c["section"], c["md_lines"]) for c in chunks_doc["chunks"]] == [("s00", [1, 3]), ("s02", [5, 6])]
    for c in chunks_doc["chunks"]:  # บรรทัดแรกของช่วงคือคอมเมนต์ของส่วนนั้นจริง
        assert md[c["md_lines"][0] - 1].startswith(f"<!-- {c['section']} ")
    assert chunks_doc["chunks"][0]["image"] == "s00.png" and chunks_doc["chunks"][0]["text"] == "บริษัท ABC 123\nบรรทัดสอง" and chunks_doc["chunks"][1]["truncated"] and chunks_doc["chunks"][1]["part"] == 1
    assert chunks_doc["not_read"] == [{"section": "s01", "kind": "table", "reason": "blank"}]
    mix = chunks_doc["chunks"][0]["script_mix"]
    assert mix["thai"] > 0 and mix["latin"] == 3 and mix["digit"] == 3  # แท็ก HTML ไม่ถูกนับ
    assert chunks_doc["chunks"][1]["script_mix"]["latin"] == 0  # "table/td" ในแท็กไม่นับเป็นอักษรละติน

