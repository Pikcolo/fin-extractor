"""โครงเก็บข้อมูล (paths.py) และ .gitignore ที่กันข้อมูลจริงหลุดเข้า git."""

from pathlib import Path

from fin_extractor.paths import DataPaths, dataset_of, slug

ROOT = Path(__file__).resolve().parents[1]


def paths(tmp_path, **folders):
    return DataPaths({"paths": {"data_dir": str(tmp_path), **({"folders": folders} if folders else {})}})


def test_layout_follows_stage_then_engine(tmp_path):
    p = paths(tmp_path)
    assert p.raw("finance_doc_scan") == tmp_path / "01_input" / "finance_doc_scan"
    assert p.preprocess("scans") == tmp_path / "02_preprocess" / "scans" / "default"
    assert p.preprocess_final("scans", "crop") == tmp_path / "02_preprocess" / "scans" / "crop" / "final"
    assert p.raw_data("paddle", "scans") == tmp_path / "04_ocr" / "scans" / "paddle" / "page"
    assert p.final("typhoon", "scans", "llm") == tmp_path / "05_extract" / "scans" / "typhoon" / "page" / "llm"
    assert p.visualize("oxr", "scans") == tmp_path / "04_ocr" / "scans" / "oxr" / "visualize"
    assert p.benchmarks() == tmp_path / "etc" / "benchmarks" and p.compare("1312") == tmp_path / "etc" / "compare" / "1312"
    assert p.reports("oxr_why") == tmp_path / "reports" / "oxr_why" and p.reports() == tmp_path / "reports"


def test_one_engine_or_extractor_is_one_folder(tmp_path):
    p = paths(tmp_path)
    final = p.final("typhoon", "scans", "rules")
    assert final.parent.parent == tmp_path / "05_extract" / "scans" / "typhoon"  # ลบ engine = ลบโฟลเดอร์ engine เดียวต่อขั้น
    assert p.raw_data("typhoon", "scans").parent == p.ocr_engine("typhoon", "scans")
    assert p.final("typhoon", "scans", "llm").parent == final.parent  # rules กับ llm อยู่ใต้วิธีเดียวกัน แยกกันลบได้
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
    p = paths(tmp_path, ocr="ocr_results")
    assert p.ocr_engine("paddle", "s") == tmp_path / "ocr_results" / "s" / "paddle"


def test_config_defaults_to_data_dir_from_default_yaml():
    from fin_extractor.config import load_config

    p = DataPaths(load_config())
    assert p.root == ROOT / "data" and p.final("paddle", "s", "rules") == ROOT / "data" / "05_extract" / "s" / "paddle" / "page" / "rules"


def test_gitignore_blocks_everything_under_data_except_keepfiles():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/**" in rules and "!data/**/.gitkeep" in rules  # ข้อมูลจริง (ภาษี/ส่วนบุคคล) ห้ามเข้า git
    for folder in ("01_input", "02_preprocess", "03_sections", "04_ocr", "05_extract", "etc/benchmarks", "etc/compare", "etc/experiments", "reports"):
        assert (ROOT / "data" / folder / ".gitkeep").exists(), folder


def test_nesting_rule_and_helpers(tmp_path):
    from fin_extractor.paths import doc_dir, find_dir, iter_results, nest_per_file

    assert [nest_per_file(n) for n in (1, 2, 3, 18)] == [False, False, True, True]  # เกิน 2 ไฟล์ → โฟลเดอร์ย่อยต่อไฟล์
    base = tmp_path / "raw_data"
    assert doc_dir(base, "a", nest=False) == base and doc_dir(base, "a", nest=True) == base / "a"
    assert find_dir(base, "a") == base                      # ยังไม่มีโฟลเดอร์ย่อย → ตรง ๆ
    (base / "a").mkdir(parents=True)
    assert find_dir(base, "a") == base / "a" and doc_dir(base, "a", nest=False) == base / "a"  # มีแล้วใช้ต่อ ไม่กระจายสองที่
    (base / "a" / "a_p1.txt").write_text("x")
    (base / "b_p1.txt").write_text("y")
    (base / "manifest.json").write_text("{}")
    assert [p.name for p in iter_results(base, "*_p*.txt")] == ["a_p1.txt", "b_p1.txt"]
    assert doc_dir(base, "../x", nest=True) == base / "x"   # ชื่อไฟล์แปลก ๆ ไม่ไต่ขึ้นโฟลเดอร์แม่


def test_bundled_fonts_exist_and_render_thai():
    from PIL import ImageFont

    from fin_extractor.paths import font_path

    for name in ("Sarabun-Regular.ttf", "Sarabun-Bold.ttf"):
        assert font_path(name).is_relative_to(ROOT), name          # ฟอนต์อยู่ในโปรเจกต์ ไม่ใช่ C:/Windows/Fonts
        assert ImageFont.truetype(str(font_path(name)), 30).getmask("ใบกำกับภาษี TAX").getbbox() is not None
    assert (ROOT / "assets" / "fonts" / "OFL.txt").exists()          # ใบอนุญาตของฟอนต์มาด้วย


def test_paddle_models_cache_goes_into_the_project_unless_user_overrides(monkeypatch):
    from fin_extractor.ocr.paddle_engine import use_project_model_cache
    from fin_extractor.paths import model_dir

    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    use_project_model_cache()
    import os

    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(model_dir(name="paddlex")) and model_dir().is_relative_to(ROOT)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", "X:/mine")
    use_project_model_cache()
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == "X:/mine"           # ผู้ใช้ตั้งเองได้ ไม่ทับ


def test_no_machine_specific_absolute_paths_in_code():
    import re

    bad = re.compile(r"[\"'][A-Za-z]:[/\\](?:Windows|Users|Program)", re.I)
    for folder in ("src", "scripts"):
        for p in (ROOT / folder).rglob("*.py"):
            assert not bad.search(p.read_text(encoding="utf-8")), p


def test_section_and_experiment_paths_follow_the_data_layout(tmp_path):
    p = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    base = tmp_path / "03_sections" / "s" / "default"
    assert p.sections("finance_doc_scan") == tmp_path / "03_sections" / "finance_doc_scan" / "default"
    assert p.sections("s", "alt") == tmp_path / "03_sections" / "s" / "alt"
    assert p.section_layout("s") == base / "layout" and p.section_features("s") == base / "layout"
    assert p.section_images("s", "001309") == base / "images" / "001309"
    assert p.section_images("s") == base / "images"
    assert p.section_images("s", "001309", "bin") == base / "images" / "bin" / "001309" and p.section_clean("s", "001309", "bin") == base / "images" / "bin" / "001309"
    assert p.experiments() == tmp_path / "etc" / "experiments" and p.experiments("openthai_sections") == tmp_path / "etc" / "experiments" / "openthai_sections"
    assert p.sections("../x").parent.parent == tmp_path / "03_sections"  # ชื่อชุดข้อมูลไต่ขึ้นโฟลเดอร์แม่ไม่ได้


def test_variant_is_a_folder_level_under_engine_in_ocr_and_extract(tmp_path):
    p = DataPaths({"paths": {"data_dir": str(tmp_path)}})
    assert p.raw_data("openthai", "s") == tmp_path / "04_ocr" / "s" / "openthai" / "page"  # อ่านทั้งหน้า
    assert p.raw_data("openthai", "s", "luma") == tmp_path / "04_ocr" / "s" / "openthai" / "luma"
    assert p.final("openthai", "s", "rules", "bin") == tmp_path / "05_extract" / "s" / "openthai" / "bin" / "rules"
    assert p.final("openthai", "s", "rules") == tmp_path / "05_extract" / "s" / "openthai" / "page" / "rules"
