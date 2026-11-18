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
