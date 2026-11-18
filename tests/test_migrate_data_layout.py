"""scripts/migrate_data_layout.py: ย้ายโครงเก่า → โครงเรียงตามขั้น โดยไม่ลบ/ทับ และรันซ้ำได้."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import migrate_data_layout as mig  # noqa: E402


def touch(p: Path, text: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def old_tree(root: Path) -> None:
    touch(root / "raw" / "ds" / "a.pdf")
    touch(root / "preprocess" / "ds" / "default" / "final" / "a_p1.png")
    touch(root / "preprocess" / "ds" / "default" / "settings.json")
    touch(root / "preprocess" / "ds" / "default" / "sections" / "features" / "a.json")
    touch(root / "preprocess" / "ds" / "default" / "sections" / "clean" / "luma" / "a" / "s00.png")
    touch(root / "result_ocr" / "openthai" / "ds" / "raw_data_luma" / "a" / "a_p1.md")
    touch(root / "result_ocr" / "openthai" / "ds" / "final_luma" / "rules" / "a" / "a_p1.json")
    touch(root / "result_ocr" / "paddle" / "ds" / "raw_data" / "a_p1.txt")
    touch(root / "result_ocr" / "paddle" / "ds" / "final" / "rules" / "a_p1.json")
    touch(root / "result_ocr" / "paddle" / "ds" / "visualize" / "a.png")
    touch(root / "result_ocr" / "openthai" / "ds" / "raw_data" / "a" / "a_p1.sections.json")  # อ่านทีละ section จากภาพสี


def test_plan_maps_every_old_folder_to_its_numbered_stage(tmp_path):
    old_tree(tmp_path)
    got = {s.relative_to(tmp_path).as_posix(): d.relative_to(tmp_path).as_posix() for s, d in mig.plan(tmp_path)}
    assert got["raw/ds"] == "01_input/ds"
    assert got["preprocess/ds/default/final"] == "02_preprocess/ds/default/final"
    assert got["preprocess/ds/default/settings.json"] == "02_preprocess/ds/default/settings.json"
    assert got["preprocess/ds/default/sections/features"] == "03_sections/ds/default/layout"
    assert got["preprocess/ds/default/sections/clean/luma"] == "03_sections/ds/default/images/luma"
    assert got["result_ocr/openthai/ds/raw_data_luma"] == "04_ocr/ds/openthai/luma"
    assert got["result_ocr/openthai/ds/final_luma/rules"] == "05_extract/ds/openthai/luma/rules"
    assert got["result_ocr/paddle/ds/raw_data"] == "04_ocr/ds/paddle/page"  # อ่านทั้งหน้า
    assert got["result_ocr/openthai/ds/raw_data"] == "04_ocr/ds/openthai/color"  # มี .sections.json
    assert got["result_ocr/paddle/ds/final/rules"] == "05_extract/ds/paddle/page/rules"
    assert got["result_ocr/paddle/ds/visualize"] == "04_ocr/ds/paddle/visualize"


def test_apply_moves_files_never_overwrites_and_is_idempotent(tmp_path):
    old_tree(tmp_path)
    touch(tmp_path / "01_input" / "ds" / "a.pdf", "ของใหม่")  # ปลายทางมีอยู่แล้ว
    done, conflicts = mig.apply(mig.plan(tmp_path))
    assert len(conflicts) == 1 and conflicts[0][1] == tmp_path / "01_input" / "ds"
    assert (tmp_path / "01_input" / "ds" / "a.pdf").read_text(encoding="utf-8") == "ของใหม่"  # ไม่ทับ
    assert (tmp_path / "raw" / "ds" / "a.pdf").exists()  # ต้นทางที่ conflict ยังอยู่
    assert (tmp_path / "04_ocr" / "ds" / "openthai" / "luma" / "a" / "a_p1.md").exists()
    assert (tmp_path / "05_extract" / "ds" / "paddle" / "page" / "rules" / "a_p1.json").exists()
    mig.prune_empty(tmp_path)
    assert not (tmp_path / "result_ocr").exists() and not (tmp_path / "preprocess").exists()  # ว่างแล้วถูก rmdir
    assert (tmp_path / "raw").exists()  # ยังมีไฟล์ ไม่ถูกลบ
    assert mig.plan(tmp_path) == [(tmp_path / "raw" / "ds", tmp_path / "01_input" / "ds")]  # เหลือแค่ conflict เดิม
