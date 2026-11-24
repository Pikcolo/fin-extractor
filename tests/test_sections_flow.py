"""flow หลัก (fin_extractor.sections): ผังและภาพรายส่วนถูกข้ามเมื่อผลยังใช้ได้ และทำใหม่เมื่อพารามิเตอร์/ผังเปลี่ยน."""

import os

import cv2
import numpy as np

from fin_extractor.io import save_image
from fin_extractor.paths import DataPaths
from fin_extractor.sections.build import build_images, build_layouts
from fin_extractor.sections.flow import images_are_current, layout_is_current, stems_needing


def make_paths(tmp_path) -> DataPaths:
    paths = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    page = np.full((900, 600, 3), 255, np.uint8)
    cv2.putText(page, "INVOICE HEADER", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.rectangle(page, (40, 150), (560, 450), (0, 0, 0), 2)
    cv2.line(page, (40, 300), (560, 300), (0, 0, 0), 2)
    cv2.putText(page, "TOTAL 2000.00", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(page, "footer note", (40, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    save_image(paths.preprocess_final("ds") / "doc1" / "doc1_p1.png", page)
    return paths


def test_layout_and_images_are_skipped_only_while_their_inputs_are_unchanged(tmp_path):
    paths = make_paths(tmp_path)
    assert not layout_is_current(paths, "ds", "doc1", "rows", 0.3)  # ยังไม่มีผัง
    build_layouts(paths, "ds", table_mode="rows", max_piece_frac=0.3)
    assert layout_is_current(paths, "ds", "doc1", "rows", 0.3)
    assert not layout_is_current(paths, "ds", "doc1", "whole", 0.3)  # พารามิเตอร์เปลี่ยน → ทำใหม่
    assert not layout_is_current(paths, "ds", "doc1", "rows", 0.5)

    assert not images_are_current(paths, "ds", "doc1", ("luma",))
    build_images(paths, "ds", variants=("luma",))
    assert images_are_current(paths, "ds", "doc1", ("luma",))
    assert not images_are_current(paths, "ds", "doc1", ("luma", "bin"))  # วิธีที่ยังไม่เคยสร้าง
    assert (paths.section_images("ds", "doc1", "luma") / "s00.png").exists()
    assert not paths.section_images("ds", "doc1", "bin").exists()  # สร้างเฉพาะวิธีที่สั่ง

    layout_file = paths.section_layout("ds") / "doc1.json"
    os.utime(layout_file, (layout_file.stat().st_atime, layout_file.stat().st_mtime + 10))  # ผังใหม่กว่าภาพรายส่วน
    assert not images_are_current(paths, "ds", "doc1", ("luma",))
    assert stems_needing(lambda s: False, ["a", "b"]) == ["a", "b"]


def test_final_image_newer_than_layout_invalidates_the_layout(tmp_path):
    paths = make_paths(tmp_path)
    build_layouts(paths, "ds")
    page = paths.preprocess_final("ds") / "doc1" / "doc1_p1.png"
    os.utime(page, (page.stat().st_atime, page.stat().st_mtime + 10))
    assert not layout_is_current(paths, "ds", "doc1", "rows", 0.3)


def banner_page() -> np.ndarray:
    """หน้าขาวที่มีกรอบตาราง (ขอบล่างที่ y=300) พาดผ่านกลางแถบสีเทา (y 270–330) ที่มีตัวอักษรขาว + ข้อความดำนอกแถบ."""
    page = np.full((900, 600, 3), 255, np.uint8)
    cv2.putText(page, "HEADER TEXT", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.rectangle(page, (40, 120), (560, 300), (0, 0, 0), 2)
    page[270:330, 30:570] = 170
    cv2.putText(page, "Receipt", (200, 312), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(page, "footer note", (40, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return page


def cuts_of(paths: DataPaths, fill_text: bool) -> list[int]:
    import json

    build_layouts(paths, "ds", fill_text=fill_text)
    secs = json.loads((paths.section_layout("ds") / "doc1.json").read_text(encoding="utf-8"))["layout_sections"]
    return [s["bbox"][1] for s in secs if s["bbox"][1] > 0]


def test_fill_aware_layout_does_not_cut_through_a_coloured_banner(tmp_path):
    paths = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    save_image(paths.preprocess_final("ds") / "doc1" / "doc1_p1.png", banner_page())
    plain = cuts_of(paths, False)
    aware = cuts_of(paths, True)
    assert any(270 < y < 330 for y in plain)  # แบบเดิม: ตัดผ่านกลางแถบสี (ตัวขาวบนเทาดูเป็นพื้นว่าง)
    assert not any(270 < y < 330 for y in aware)  # fill-aware: ไม่ตัดผ่าแถบ
    assert layout_is_current(paths, "ds", "doc1", "rows", 0.3, fill_text=True) and not layout_is_current(paths, "ds", "doc1", "rows", 0.3)


def qr_page() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """หน้าขาว (ขนาดเท่า A4 ที่ 150 dpi) มีข้อความด้านบน และ QR อยู่แถบของตัวเองด้านล่าง คืน (ภาพ, กรอบ QR x0,y0,x1,y1)."""
    page = np.full((1754, 1240, 3), 255, np.uint8)
    cv2.putText(page, "INVOICE HEADER", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    qr = cv2.QRCodeEncoder.create().encode("INV-001")
    qr = cv2.resize(qr, (300, 300), interpolation=cv2.INTER_NEAREST)
    x0, y0 = 100, 1200
    page[y0:y0 + 300, x0:x0 + 300] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    return page, (x0, y0, x0 + 300, y0 + 300)


def test_qr_codes_are_blanked_before_the_layout_and_never_reach_ocr(tmp_path):
    import json

    from fin_extractor.sections.crops import masked_final
    from fin_extractor.sections.read import fingerprint

    paths = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    page, (x0, y0, x1, y1) = qr_page()
    save_image(paths.preprocess_final("ds") / "doc1" / "doc1_p1.png", page)
    layout_file = paths.section_layout("ds") / "doc1.json"

    build_layouts(paths, "ds", mask_codes=False)
    plain = json.loads(layout_file.read_text(encoding="utf-8"))
    work_y = 1600 / 1754  # สำเนาทำงานด้านยาว 1600
    over_qr = lambda doc: [s for s in doc["layout_sections"] if s["keep"] and s["bbox"][1] < y1 * work_y and s["bbox"][1] + s["bbox"][3] > y0 * work_y]  # noqa: E731
    assert over_qr(plain) and plain["masked_codes"] == []  # ไม่ระบาย: แถบ QR ถูกส่ง OCR

    build_layouts(paths, "ds")  # ค่าตั้งต้น: ระบาย
    doc = json.loads(layout_file.read_text(encoding="utf-8"))
    assert [(c["kind"], c["data"]) for c in doc["masked_codes"]] == [("qr", "INV-001")]  # ข้อความในรหัสเก็บไว้ในผัง
    assert over_qr(doc) == []  # แถบที่เหลือแต่ QR ว่าง → ไม่ส่ง OCR
    assert masked_final(paths, "ds", "doc1", doc)[y0:y1, x0:x1].min() == 255  # ภาพที่ตัดส่วนไปอ่านไม่มี QR เหลือ
    assert layout_is_current(paths, "ds", "doc1", "rows", 0.3) and not layout_is_current(paths, "ds", "doc1", "rows", 0.3, mask_codes=False)
    assert fingerprint([{"id": "s01", "bbox": [0, 0, 5, 5], "keep": True, "masked": ["qr"]}]) == [["s01", [0, 0, 5, 5], ["qr"]]]
    assert fingerprint([{"id": "s01", "bbox": [0, 0, 5, 5], "keep": True}]) == [["s01", [0, 0, 5, 5]]]  # ส่วนที่ไม่ถูกระบาย: รูปแบบเดิม ผล OCR เก่ายังใช้ได้


def test_trim_sections_shrinks_bands_to_their_text_and_keeps_the_band(tmp_path):
    import json

    paths = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    page = np.full((1754, 1240, 3), 255, np.uint8)
    cv2.putText(page, "LEFT ONLY TEXT", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)  # แถบเต็มหน้า แต่ข้อความอยู่ซ้ายสุด
    cv2.putText(page, "SECOND BLOCK", (700, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    save_image(paths.preprocess_final("ds") / "doc1" / "doc1_p1.png", page)
    layout_file = paths.section_layout("ds") / "doc1.json"

    build_layouts(paths, "ds")
    plain = [s for s in json.loads(layout_file.read_text(encoding="utf-8"))["layout_sections"] if s["keep"]]
    build_layouts(paths, "ds", trim="all")
    trimmed = [s for s in json.loads(layout_file.read_text(encoding="utf-8"))["layout_sections"] if s["keep"]]

    assert [s["id"] for s in plain] == [s["id"] for s in trimmed]  # ส่วนเดิม ลำดับเดิม
    for p, t in zip(plain, trimmed):
        assert t["band"] == p["bbox"]  # กรอบเดิมเก็บไว้
        x, y, w, h = t["bbox"]
        bx, by, bw, bh = t["band"]
        assert bx <= x and by <= y and x + w <= bx + bw and y + h <= by + bh  # อยู่ในแถบเดิม
        assert w * h < 0.5 * bw * bh  # ตัดพื้นว่างทิ้งไปเกินครึ่ง
    assert layout_is_current(paths, "ds", "doc1", "rows", 0.3, trim="all") and not layout_is_current(paths, "ds", "doc1", "rows", 0.3)
    assert not layout_is_current(paths, "ds", "doc1", "rows", 0.3, trim="text-lr")

    build_layouts(paths, "ds", trim="text-lr")  # หดแค่ซ้าย/ขวา: บน/ล่างเท่าแถบเดิม
    lr = [s for s in json.loads(layout_file.read_text(encoding="utf-8"))["layout_sections"] if s["keep"]]
    for p, t in zip(plain, lr):
        assert t["bbox"][1] == p["bbox"][1] and t["bbox"][3] == p["bbox"][3] and t["bbox"][2] < p["bbox"][2]


def test_preprocess_meta_survives_cached_reads_and_orientation_flags_list_the_bad_pages(tmp_path):
    import json

    from fin_extractor.config import load_config
    from fin_extractor.sections.flow import orientation_flags
    from fin_extractor.stages import meta_path_of, prepared_pages

    paths = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    src = tmp_path / "01_input" / "ds" / "doc1.png"
    save_image(src, np.full((300, 200, 3), 255, np.uint8))
    cfg = load_config()
    profile = paths.preprocess("ds")
    first = list(prepared_pages(src, cfg, profile))
    image = profile / "final" / "doc1_p1.png"
    assert image.exists() and meta_path_of(image).name == "doc1_p1.meta.json"
    saved = json.loads(meta_path_of(image).read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(first[0][2], default=str))  # เก็บ meta ของรอบที่ทำจริง
    again = list(prepared_pages(src, cfg, profile))  # รอบถัดไปอ่านจากที่เก็บไว้ แต่ meta ไม่หาย
    assert again[0][2]["cached"] is True and {k: v for k, v in again[0][2].items() if k != "cached"} == saved
    meta_path_of(image).write_text(json.dumps({"orientation": 0, "orientation_error": "OSError: cudnn"}), encoding="utf-8")
    assert orientation_flags(paths, "ds", ["doc1"]) == [("doc1", "OSError: cudnn")]
    assert orientation_flags(paths, "ds", ["other"]) == []


def test_compare_flags_rows_that_are_not_comparable_and_totals_skip_them(tmp_path):
    import json
    import os

    from fin_extractor.sections.compare import compare_variants, to_markdown

    paths = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    layout = paths.section_layout("ds")
    layout.mkdir(parents=True)
    (layout / "a.json").write_text("{}", encoding="utf-8")
    (layout / "b.json").write_text("{}", encoding="utf-8")
    fp = {"a": [["s00", [0, 0, 1, 1]]], "b": [["s00", [0, 0, 1, 1]]]}
    for v, a_missing, b_layout in (("luma", ["x"], fp["b"]), ("bin", [], [["s00", [0, 9, 1, 1]]])):  # bin ใช้ผังของ b ต่างจาก luma
        out = paths.final("openthai", "ds", "rules", v)
        out.mkdir(parents=True)
        files = [{"file": f"{s}.pdf", "status": "needs_review", "missing_fields": a_missing if s == "a" else [], "checks_failed": 0} for s in ("a", "b")]
        (out / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        raw = paths.raw_data("openthai", "ds", v)
        for s in ("a", "b"):
            d = raw / s
            d.mkdir(parents=True)
            (d / f"{s}_p1.sections.json").write_text(json.dumps({"layout": fp["a"] if s == "a" else b_layout}), encoding="utf-8")
            os.utime(d / f"{s}_p1.sections.json", (9e9, 9e9))  # ใหม่กว่าผังเสมอ
    result = compare_variants(paths, "ds", "openthai", ("luma", "bin"))
    by = {r["file"]: r["flags"] for r in result["rows"]}
    assert by["a"] == [] and by["b"] == ["ผังต่างกัน"]
    assert result["comparable"] == 1 and result["totals"]["luma"]["missing"] == 1 and result["totals"]["bin"]["missing"] == 0
    assert "ผังต่างกัน" in to_markdown(result)
