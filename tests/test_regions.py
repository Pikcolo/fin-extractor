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
