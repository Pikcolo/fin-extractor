"""โหมดแบ่ง region/ตาราง (sections/regions/) — test ที่ย้ายมาจาก learning/preprocess_finance/tests + ลำดับงานต่อหน้า (run_page)."""

import json

import cv2
import numpy as np

from fin_extractor.sections.regions import doc_types, geometry, lines, split, tables
from fin_extractor.sections.regions.run import RegionConfig, run_page


def test_classify_exact_pattern_and_default():
    cfg = {"default_type": "generic",
           "documents": {"A": "wht_50tawi", "B": {"type": "invoice", "clean": "max_channel"}},
           "patterns": {"UPS_*": "invoice"}}
    assert doc_types.classify("A", cfg) == ("wht_50tawi", {})
    assert doc_types.classify("B", cfg) == ("invoice", {"clean": "max_channel"})
    assert doc_types.classify("UPS_001", cfg) == ("invoice", {})
    assert doc_types.classify("zzz", cfg) == ("generic", {})


def test_trim_box_ignores_rule_slivers():
    text = np.zeros((300, 300), np.uint8)
    text[100:120, 100:180] = 255                  # the text
    text[280:282, 0:300] = 255                    # 2 px residue of a removed table rule
    assert geometry.trim_box(text, (0, 0, 300, 300), pad_px=5) == (95, 95, 185, 125)
    assert geometry.trim_box(text, (0, 200, 300, 300), pad_px=5) is None


def test_detect_hlines_ignores_dotted_leaders():
    img = np.full((600, 1000, 3), 255, np.uint8)
    img[100:103, 50:950] = 0                 # real rule
    img[300:302, 100:900:12] = 0             # dotted leader "......"
    assert [ln["y"] for ln in lines.detect_hlines(img)] == [101]


def test_snap_moves_edge_onto_nearby_rule_only():
    rules = [{"y": 200, "x0": 0, "x1": 1000}]
    assert lines.snap_y(190, 0, 1000, rules, tol_px=15) == (200, True)
    assert lines.snap_y(150, 0, 1000, rules, tol_px=15) == (150, False)
    assert lines.snap_y(195, 0, 1000, [{"y": 200, "x0": 900, "x1": 1000}], tol_px=15) == (195, False)


def test_safe_cut_avoids_text_beside_short_rule():
    text = np.zeros((400, 1000), np.uint8)
    text[190:215, 50:300] = 255              # text on the left, crossing the rule's row
    rule = {"y": 200, "x0": 600, "x1": 1000}  # short rule on the right (totals box)
    c = lines.safe_cut(rule, text, search_px=30)
    assert c is not None and (c <= 187 or c >= 217)
    assert lines.safe_cut(rule, text, search_px=5) is None
    assert lines.safe_cut({"y": 300, "x0": 600, "x1": 1000}, text, search_px=5) == 300


def test_ratio_to_px_uses_frame_offset():
    assert split.ratio_to_px([0, 0, 1, 1], (10, 20, 110, 220)) == (10, 20, 110, 220)
    assert split.ratio_to_px([0.5, 0.25, 1, 0.5], (0, 0, 200, 400)) == (100, 100, 200, 200)


def test_whitespace_bands_and_merge():
    mask = np.zeros((100, 50), np.uint8)
    mask[10:20] = 255
    mask[40:50] = 255
    mask[52:60] = 255
    bands = split.whitespace_bands(mask, min_gap=5, min_band=3)
    assert bands == [(10, 20), (40, 60)]
    assert split.merge_bands(bands, 1) == [(10, 60)]


def test_split_rows_keeps_wrapped_lines_and_offset_numbers():
    text = np.zeros((400, 600), np.uint8)
    seps = [0, 100, 400, 600]                     # item no. | description | amount
    for y in (20, 150):
        text[y:y + 20, 30:60] = 255
    text[20:40, 130:350] = 255
    text[60:80, 130:250] = 255                    # item 1 wrapped line (no anchor entry)
    text[150:170, 130:350] = 255
    text[5:25, 450:550] = 255                     # amounts printed ~15 px higher than the code
    text[135:155, 450:550] = 255
    items = tables.split_rows(text, (0, 0, 600, 400), seps, min_gap=6, min_row=12)
    assert len(items) == 2
    assert items[0][1][1] >= 80 and items[1][1][0] >= 140
    assert items[0][2][1] < 40 and items[1][2][0] > 120


def test_text_crosses_tells_rule_from_text():
    band = np.zeros((60, 200), bool)
    band[:, 99:102] = True
    assert not tables._text_crosses(band, 100)
    band[20:40, 60:140] = True
    assert tables._text_crosses(band, 100)


def _invoice_page() -> np.ndarray:
    """หน้าสี 300 dpi จำลอง: หัวเอกสาร + ตาราง 3 คอลัมน์ (หัว + 2 แถว) มีเส้นนอน/ตั้ง + ข้อความท้าย."""
    img = np.full((3000, 2100, 3), 255, np.uint8)
    put = lambda t, x, y: cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    put("INVOICE NO 42", 150, 250)
    xs, ys = [150, 750, 1350, 1950], [600, 800, 1100, 1400]
    for y in ys:
        cv2.line(img, (xs[0], y), (xs[-1], y), (0, 0, 0), 5)
    for x in xs:
        cv2.line(img, (x, ys[0]), (x, ys[-1]), (0, 0, 0), 5)
    for r, row in enumerate([("QTY", "PRICE", "AMOUNT"), ("1", "2000", "2000"), ("2", "500", "1000")]):
        for c, t in enumerate(row):
            put(t, xs[c] + 60, ys[r] + 120)
    put("TOTAL 3000", 150, 2000)
    return img


def test_run_page_writes_stage_images_crops_tables_and_page_boxes(tmp_path):
    full = _invoice_page()
    m = run_page(full, "X_0001", 300.0, RegionConfig(), tmp_path / "X_0001")
    page_dir = tmp_path / "X_0001"
    assert json.loads((page_dir / "manifest.json").read_text(encoding="utf-8"))["doc_type"] == "generic"
    assert (page_dir / "01_normalize.png").exists() and (page_dir / "04_regions.png").exists()
    assert m["source"]["scale"] == round(200 / 300, 6)
    tb = m["stages"]["tables"]["tables"]
    assert len(tb) == 1 and tb[0]["rows"][0]["kind"] == "header" and len(tb[0]["rows"]) == 3
    files = m["stages"]["export"]["files"]
    cells = [f for f in files if "table" in f]
    assert {(f["row"], f["col"]) for f in cells} == {(r, c) for r in range(3) for c in range(3)}
    assert all((page_dir / f["path"]).exists() for f in files)
    # กล่องบนภาพ final ต้องครอบข้อความจริงของช่องนั้น: ช่อง (1, 2) = "2000" อยู่ที่ x 1350–1950, y 800–1100 ของภาพเต็ม
    x0, y0, x1, y1 = next(f["box_page"] for f in cells if (f["row"], f["col"]) == (1, 2))
    assert 1350 <= x0 < x1 <= 1950 and 800 <= y0 < y1 <= 1100
    kv = m["stages"]["export"]["kv_pairs"]
    assert any(k["row"] == 1 and k["col"] == 2 and k["header_key_crop"].endswith("t0_r00_c02__luma.png") for k in kv)


def test_run_page_clears_previous_results(tmp_path):
    page_dir = tmp_path / "X_0001"
    (page_dir / "crops").mkdir(parents=True)
    (page_dir / "crops" / "99_old__luma.png").write_bytes(b"x")
    (page_dir / "07_old.png").write_bytes(b"x")
    run_page(_invoice_page(), "X_0001", 300.0, RegionConfig(), page_dir, debug_images=False)
    assert not (page_dir / "crops" / "99_old__luma.png").exists() and not (page_dir / "07_old.png").exists()
    assert not (page_dir / "01_normalize.png").exists()  # --no-debug-images


def test_run_page_writes_a_layout_the_section_reader_accepts(tmp_path):
    from fin_extractor.sections.read import clean_index, kept_layout

    page_dir = tmp_path / "X_0001"
    run_page(_invoice_page(), "X_0001", 300.0, RegionConfig(), page_dir, debug_images=False)
    layout = json.loads((page_dir / "sections.json").read_text(encoding="utf-8"))["layout_sections"]
    entries = clean_index(page_dir / "crops", layout, "X_0001")  # หยุด (sys.exit) ถ้า index ไม่ตรงผัง
    assert [i for i, _b in kept_layout(layout)] == list(entries)
    assert all((page_dir / "crops" / e["file"]).exists() and e["file"].endswith("__luma.png") for e in entries.values())
    assert not any("tables/" in e["file"] for e in entries.values())  # อ่านเฉพาะ region ไม่อ่านช่องตารางซ้ำ


def test_run_page_does_not_send_pieces_without_text_to_ocr(tmp_path):
    """แถวตารางที่มีแต่เส้นขอบช่อง (ไม่มีข้อความ: แบบ 1320/1322) ไม่ถูกส่ง OCR และถูกบันทึกใน skipped_regions."""
    full = _invoice_page()
    y0, y1 = 2400, 2460  # สูง 60 px: เส้นขอบช่อง (40 px ที่ 200 dpi) สั้นกว่าเกณฑ์เส้นตั้งของหน้า (~97 px) จึงเหลือในหน้ากากระดับหน้า แบบ 1320/1322
    for y in (y0, y1):
        cv2.line(full, (150, y), (1950, y), (0, 0, 0), 5)
    for x in (150, 750, 1350, 1950):  # เส้นขอบช่องสั้น: สั้นกว่าเกณฑ์เส้นของหน้า จึงไม่ถูกลบในหน้ากากระดับหน้า
        cv2.line(full, (x, y0), (x, y1), (0, 0, 0), 5)
    m = run_page(full, "X_0001", 300.0, RegionConfig(), tmp_path / "X_0001", debug_images=False)
    regions = [f for f in m["stages"]["export"]["files"] if "region" in f]
    assert regions and not any(f["box_page"][1] < y1 and f["box_page"][3] > y0 for f in regions)  # ไม่มีท่อนไหนทับแถวว่าง
    assert any(s.get("reason") == "no text" for s in m["stages"]["export"]["skipped_regions"])
    assert any(f["box_page"][1] < 300 for f in regions) and any(f["box_page"][1] <= 2000 <= f["box_page"][3] for f in regions)  # ข้อความยังอยู่ครบ


def _text_page(rows, w=600, h=1000):
    """หน้ากากข้อความจำลอง: rows = [(y, สูง, x0, x1)] แต่ละแถวเป็นแท่งหมึก (ตัวอักษร)."""
    t = np.zeros((h, w), np.uint8)
    for y, hh, x0, x1 in rows:
        t[y:y + hh, x0:x1] = 255
    return t


def test_cut_region_trims_to_ink_and_keeps_short_tables_whole():
    from fin_extractor.sections.regions import pieces

    t = _text_page([(100, 20, 50, 400), (140, 20, 50, 300), (998, 1, 0, 600)])  # 1 px เศษเส้นล่าง = ไม่ใช่หมึก
    cut = pieces.cut_region(t, (0, 0, 600, 1000), [], max_h=300, min_line_h=12, pad=6)
    assert cut["method"] == "whole" and cut["pieces"] == [(94, 166)]


def test_cut_region_cuts_at_the_widest_gap_not_inside_an_item():
    from fin_extractor.sections.regions import pieces

    # สองย่อหน้า บรรทัดห่าง 10 px ภายในย่อหน้า ระหว่างย่อหน้าห่าง 70 px (420–490) — จุดแบ่งเท่า ๆ กัน (~480) ตกใกล้ย่อหน้าที่สอง
    rows = [(y, 20, 50, 400) for y in range(100, 401, 30)] + [(y, 20, 50, 400) for y in range(490, 851, 30)]
    cut = pieces.cut_region(_text_page(rows), (0, 0, 600, 1000), [], max_h=500, min_line_h=12, pad=6)
    assert cut["method"] == "gaps" and len(cut["pieces"]) == 2
    assert cut["pieces"][0][1] <= 426 + 6 and cut["pieces"][1][0] >= 490 - 6  # ตัดในช่องว่างระหว่างย่อหน้า แต่ละท่อนหดตามบรรทัดของตัวเอง


def test_cut_region_uses_item_numbers_in_the_first_column():
    from fin_extractor.sections.regions import pieces

    vl = [{"x": 100, "y0": 0, "y1": 1000}]  # เส้นตั้งคั่นคอลัมน์ "ลำดับ" กับ "รายการ"
    rows = []
    for k, y in enumerate((100, 400, 700)):  # 3 รายการ แต่ละรายการมีเลขข้อ + คำอธิบาย 6 บรรทัดห่างกัน 30 px
        rows.append((y, 20, 30, 70))
        rows += [(y + 40 * i, 20, 130, 500) for i in range(6)]
    rows.append((340, 20, 130, 500))  # บรรทัดคำอธิบายที่ห่างจากบรรทัดก่อนหน้ามาก (ช่องว่างกว้างแต่อยู่กลางรายการ 1)
    cut = pieces.cut_region(_text_page(rows), (0, 0, 600, 1000), vl, max_h=500, min_line_h=12, pad=6)
    assert cut["method"] == "items"
    assert [p[0] for p in cut["pieces"][1:]] and all(p[0] in range(360, 400) or p[0] in range(660, 700) for p in cut["pieces"][1:])


def test_strips_inside_a_table_with_columns_are_merged_but_a_frame_border_alone_does_not_merge():
    from fin_extractor.sections.regions import split

    strips = [{"name": f"strip_{i:02d}", "box": (0, a, 1000, b), "mode": "generic:lines"} for i, (a, b) in enumerate([(0, 100), (100, 300), (300, 500)])]
    frame = [{"x": 0, "y0": 0, "y1": 500}, {"x": 1000, "y0": 0, "y1": 500}]  # ขอบกรอบซ้าย/ขวาพาดทั้งหน้า: ไม่ใช่ตาราง
    assert [r["name"] for r in split.merge_ruled(strips, frame, 12)] == ["strip_00", "strip_01", "strip_02"]
    columns = frame + [{"x": 400, "y0": 100, "y1": 500}]  # เส้นแบ่งคอลัมน์ของตารางที่เริ่มที่ y 100 (หัวคอลัมน์) ลงไปถึงแถวข้อมูล
    merged = split.merge_ruled(strips, columns, 12)
    assert [r["name"] for r in merged] == ["strip_00", "strip_01-02"]  # หัวคอลัมน์อยู่ภาพเดียวกับแถว · แถบบนตาราง (ปลายเส้นที่ y 100) ไม่ถูกรวม
    assert merged[1]["box"] == (0, 100, 1000, 500) and merged[1]["merged"] == 2
    assert merged[1]["header"] == (0, 100, 1000, 300)  # แถบแรกของตาราง = แถวหัวคอลัมน์ ใช้แปะซ้ำเมื่อถูกตัดตามความสูง


def test_a_tall_table_cut_into_pieces_repeats_its_column_header_on_every_piece(tmp_path):
    """ตารางที่มีคอลัมน์ (4 เส้นตั้ง) สูงเกินเพดาน: ถูกตัดเป็นท่อน แต่ทุกท่อนหลังท่อนแรกมีแถวหัวคอลัมน์ต่อไว้บนสุด."""
    img = np.full((3000, 2100, 3), 255, np.uint8)
    put = lambda t, x, y: cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)  # noqa: E731
    put("INVOICE NO 42", 150, 200)
    xs, top, bottom = [150, 750, 1350, 1950], 400, 2700
    rows = [top, 520, *range(640, bottom, 140), bottom]
    for y in rows:
        cv2.line(img, (xs[0], y), (xs[-1], y), (0, 0, 0), 5)
    for x in xs:
        cv2.line(img, (x, top), (x, bottom), (0, 0, 0), 5)
    for c, t in enumerate(("ITEM", "QTY", "AMOUNT")):
        put(t, xs[c] + 60, top + 80)
    for r, y in enumerate(rows[1:-1]):
        for c, t in enumerate((f"ROW {r}", str(r + 1), f"{(r + 1) * 100}.00")):
            put(t, xs[c] + 60, y + 90)
    m = run_page(img, "X_0001", 300.0, RegionConfig(), tmp_path / "X_0001", debug_images=False)
    regions = [f for f in m["stages"]["export"]["files"] if "region" in f]
    table = [f for f in regions if f["box_page"][1] >= top - 20]
    assert len(table) >= 2  # สูงเกินเพดาน 0.3 ของหน้า: ตัดเป็นหลายท่อน
    assert "header_box_page" not in table[0] and all("header_box_page" in f for f in table[1:])  # ท่อนแรกมีหัวอยู่แล้ว
    for f in table[1:]:
        crop = cv2.imdecode(np.fromfile(str(tmp_path / "X_0001" / f["path"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        hb, pb = f["header_box_page"], f["box_page"]
        assert crop.shape[0] > pb[3] - pb[1] and hb[3] <= pb[1]  # ภาพสูงกว่าแถวของท่อน (มีหัวต่อบน) และหัวอยู่เหนือแถวในหน้าจริง
