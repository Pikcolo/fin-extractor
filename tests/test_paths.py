"""โครงเก็บข้อมูล (paths.py) และ .gitignore ที่กันข้อมูลจริงหลุดเข้า git."""

from pathlib import Path

from fin_extractor.paths import DataPaths, dataset_of, slug

ROOT = Path(__file__).resolve().parents[1]


def paths(tmp_path, **folders):
    return DataPaths({"paths": {"data_dir": str(tmp_path), **({"folders": folders} if folders else {})}})


def test_layout_follows_stage_then_engine(tmp_path):
    p = paths(tmp_path)
    assert p.raw("finance_doc_scan") == tmp_path / "raw" / "finance_doc_scan"
    assert p.preprocess("scans") == tmp_path / "preprocess" / "scans" / "default"
    assert p.preprocess_final("scans", "crop") == tmp_path / "preprocess" / "scans" / "crop" / "final"
    assert p.preprocess_steps("scans") == tmp_path / "preprocess" / "scans" / "default" / "steps"
    assert p.raw_data("paddle", "scans") == tmp_path / "result_ocr" / "paddle" / "scans" / "raw_data"
    assert p.final("typhoon", "scans", "llm") == tmp_path / "result_ocr" / "typhoon" / "scans" / "final" / "llm"
    assert p.visualize("oxr", "scans") == tmp_path / "result_ocr" / "oxr" / "scans" / "visualize"
    assert p.benchmarks() == tmp_path / "benchmarks" and p.compare("1312") == tmp_path / "compare" / "1312"
    assert p.reports("oxr_why") == tmp_path / "reports" / "oxr_why" and p.reports() == tmp_path / "reports"


def test_one_engine_or_extractor_is_one_folder(tmp_path):
    p = paths(tmp_path)
    final = p.final("typhoon", "scans", "rules")
    assert final.parent.parent == p.engine("typhoon", "scans")  # ลบ engine = ลบโฟลเดอร์ engine/dataset เดียว
    assert p.final("typhoon", "scans", "llm").parent == final.parent  # rules กับ llm อยู่ใต้ final/ ด้วยกัน แยกกันลบได้
    assert p.raw_data("typhoon", "scans") != final  # ข้อความดิบแยกจากผลสุดท้าย (OCR ครั้งเดียว สกัดได้หลายแบบ)


def test_names_cannot_escape_their_folder(tmp_path):
    p = paths(tmp_path)
    for bad in ("../x", "a/b", "a\\b", "..", "  "):
        got = p.final(bad, bad, bad)
        assert tmp_path in got.parents and ".." not in got.relative_to(tmp_path).parts
        assert slug(bad) and "/" not in slug(bad) and "\\" not in slug(bad)


def test_dataset_is_the_input_folder_name(tmp_path):
    folder = tmp_path / "raw" / "finance doc scan"
    folder.mkdir(parents=True)
    (folder / "a.pdf").touch()
    assert dataset_of(folder) == "finance_doc_scan"
    assert dataset_of(folder / "a.pdf") == "finance_doc_scan"  # ไฟล์เดี่ยวใช้ชื่อโฟลเดอร์ที่ไฟล์อยู่


def test_top_level_folder_names_can_be_overridden(tmp_path):
    p = paths(tmp_path, result_ocr="ocr_results")
    assert p.engine("paddle", "s") == tmp_path / "ocr_results" / "paddle" / "s"


def test_config_defaults_to_data_dir_from_default_yaml():
    from fin_extractor.config import load_config

    p = DataPaths(load_config())
    assert p.root == ROOT / "data" and p.final("paddle", "s", "rules") == ROOT / "data" / "result_ocr" / "paddle" / "s" / "final" / "rules"


def test_gitignore_blocks_everything_under_data_except_keepfiles():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/**" in rules and "!data/**/.gitkeep" in rules  # ข้อมูลจริง (ภาษี/ส่วนบุคคล) ห้ามเข้า git
    for folder in ("raw", "preprocess", "result_ocr", "benchmarks", "compare", "reports"):
        assert (ROOT / "data" / folder / ".gitkeep").exists(), folder
