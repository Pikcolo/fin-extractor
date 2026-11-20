"""คำสั่ง preprocess / extract เขียนผลตามโครงใน paths.py และการอ่านทั้งหน้าถูกห้าม (pipeline เป็นของปลอม ไม่เรียก OCR/GPU)."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fin_extractor import cli
from fin_extractor.config import load_config
from fin_extractor.io import save_image
from fin_extractor.schema import ExtractionResult, InvoiceData


class FakePipe:
    """แทน FinanceOCRPipeline: ไม่โหลดโมเดลใด ๆ นับว่าอ่านสดหรือสกัดจากข้อความดิบ."""

    log: list = []

    def __init__(self, cfg, mode="ocr", engine="paddle", use_llm=None, with_reader=True, strategy=None):
        self.mode, self.with_reader, self.strategy = mode, with_reader, strategy or "winner"
        self.reader = SimpleNamespace(name=engine)
        self.llm = object() if (use_llm if use_llm is not None else mode == "vlm") else None
        self.engine_name = f"{engine if mode == 'ocr' else 'vlm:fake'}+{'llm' if self.llm else 'rules'}"
        FakePipe.log.append(("init", with_reader))

    def _result(self, source, page, raw):
        return ExtractionResult(source_file=source, page=page, engine=self.engine_name, data=InvoiceData(),
                                status="needs_review", raw_text=raw, ocr_confidence=0.5)

    def process_raw(self, source, page, raw, meta=None):
        FakePipe.log.append(("raw", page, raw, dict(meta or {})))
        return self._result(source, page, raw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["preprocessing"] = {"keep_color": True, "min_long_side": 0}
    src = tmp_path / "data" / "01_input" / "scans"
    save_image(src / "a.png", np.full((30, 50, 3), 180, dtype=np.uint8))
    FakePipe.log = []
    monkeypatch.setattr("fin_extractor.pipeline.FinanceOCRPipeline", FakePipe)
    return SimpleNamespace(cfg=cfg, src=src, data=tmp_path / "data")


def run(env, *argv):
    args = cli.build_parser().parse_args(list(argv))
    args.func(args, env.cfg)


def put_raw(env, engine="openthai", variant="luma", stems=("a",), text="ข้อความจาก section"):
    """ข้อความดิบแบบที่ขั้น ocr ของ sections เขียน: 04_ocr/<ชุด>/<engine>/<วิธี>/<ไฟล์>/<ไฟล์>_p1.md"""
    for stem in stems:
        d = env.data / "04_ocr" / "scans" / engine / variant / stem
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}_p1.md").write_text(f"<!-- s00 text -->\n{text} {stem}", encoding="utf-8")


def test_extract_reads_section_text_without_loading_a_reader_and_writes_its_own_folder(env, capsys):
    put_raw(env)
    run(env, "extract", str(env.src))
    assert FakePipe.log[0] == ("init", False) and [e[0] for e in FakePipe.log[1:]] == ["raw"]   # ไม่โหลดตัวอ่าน ไม่อ่านภาพ
    assert FakePipe.log[1][2] == "<!-- s00 text -->\nข้อความจาก section a"
    final = env.data / "05_extract" / "scans" / "openthai" / "luma" / "rules"
    result = json.loads((final / "a_p1.json").read_text(encoding="utf-8"))
    assert result["raw_text"] is None                                   # ผลสุดท้ายไม่แบกข้อความดิบ (อยู่ที่ 04_ocr)
    m = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert m["files"][0]["output"] == "a_p1.json" and m["settings"]["raw_data"] == "../../../../../04_ocr/scans/openthai/luma"
    assert not (env.data / "02_preprocess").exists()                   # extract ไม่แตะภาพเลย
    capsys.readouterr()


def test_every_engine_reads_from_its_own_section_folder(env, capsys):
    for engine in ("typhoon", "paddle", "oxr"):
        put_raw(env, engine=engine, variant="flat")
        run(env, "extract", str(env.src), "--engine", engine, "--variant", "flat")
        assert (env.data / "05_extract" / "scans" / engine / "flat" / "rules" / "a_p1.json").exists()
    assert not list(env.data.rglob("page"))                             # ไม่มีโฟลเดอร์ผลอ่านทั้งหน้า
    capsys.readouterr()


def test_llm_on_writes_the_llm_folder_next_to_rules(env, capsys):
    put_raw(env)
    run(env, "extract", str(env.src), "--llm", "on")
    assert (env.data / "05_extract" / "scans" / "openthai" / "luma" / "llm" / "a_p1.json").exists()
    capsys.readouterr()


def test_extract_without_section_text_skips_the_file_and_creates_nothing_else(env, capsys):
    run(env, "extract", str(env.src))
    out = capsys.readouterr().out
    assert "ไม่มีข้อความดิบ" in out and "sections --engine openthai --to ocr" in out
    assert not list((env.data / "05_extract").rglob("a_p1.json"))


def test_whole_page_reading_is_refused_for_every_engine(env):
    for engine in ("openthai", "typhoon", "paddle", "oxr"):
        with pytest.raises(SystemExit, match="ห้ามอ่านทั้งหน้า"):
            run(env, "ocr", str(env.src), "--engine", engine)
    assert not (env.data / "04_ocr").exists()


def test_old_whole_page_options_are_gone(env):
    for argv in (["--mode", "vlm"], ["--reuse-raw"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["extract", str(env.src), *argv])


def test_keep_raw_keeps_text_in_the_final_json_too(env, capsys):
    put_raw(env)
    run(env, "extract", str(env.src), "--keep-raw")
    r = json.loads((env.data / "05_extract" / "scans" / "openthai" / "luma" / "rules" / "a_p1.json").read_text(encoding="utf-8"))
    assert r["raw_text"] == "<!-- s00 text -->\nข้อความจาก section a"
    capsys.readouterr()


def test_out_reads_and_writes_under_the_given_folder(env, capsys, tmp_path):
    out = tmp_path / "exp"
    d = out / "04_ocr" / "luma" / "a"
    d.mkdir(parents=True)
    (d / "a_p1.md").write_text("ข้อความ", encoding="utf-8")
    run(env, "extract", str(env.src), "--out", str(out))
    assert (out / "05_extract" / "luma" / "rules" / "a_p1.json").exists()
    assert not (env.data / "05_extract").exists()
    capsys.readouterr()


def test_preprocess_command_manifest_points_at_final_images(env, capsys):
    run(env, "preprocess", str(env.src), "--profile", "crop")
    pdir = env.data / "02_preprocess" / "scans" / "crop"
    m = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    assert m["files"][0]["output"] == "final/a_p1.png" and (pdir / "final" / "a_p1.png").exists()
    capsys.readouterr()


def test_preprocess_steps_are_nested_per_file_only_when_many(env, capsys):
    for name in ("b", "c"):
        save_image(env.src / f"{name}.png", np.full((30, 50, 3), 180, dtype=np.uint8))
    run(env, "preprocess", str(env.src), "--steps", "--profile", "s")
    steps = env.data / "02_preprocess" / "scans" / "s" / "steps"
    assert any((steps / "a").iterdir()) and any((steps / "c").iterdir())
    capsys.readouterr()


def test_extract_only_selects_files_by_name_suffix_but_keeps_the_per_file_layout(env, capsys):
    for name in ("b", "c"):
        save_image(env.src / f"{name}.png", np.full((30, 50, 3), 180, dtype=np.uint8))
    put_raw(env, stems=("a", "b", "c"))
    run(env, "extract", str(env.src), "--only", "b")
    ex = env.data / "05_extract" / "scans" / "openthai" / "luma" / "rules"
    m = json.loads((ex / "manifest.json").read_text(encoding="utf-8"))
    assert [f["output"] for f in m["files"]] == ["b/b_p1.json"]
    assert not (ex / "a").exists() and not (ex / "c").exists()
    capsys.readouterr()
