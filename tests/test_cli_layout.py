"""คำสั่ง preprocess / ocr / extract เขียนผลตามโครงใน paths.py (ตัวอ่านและ pipeline เป็นของปลอม ไม่เรียก OCR/GPU)."""

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

    def __init__(self, cfg, mode="ocr", engine="paddle", use_llm=None, with_reader=True):
        self.mode, self.with_reader = mode, with_reader
        self.llm = object() if (use_llm if use_llm is not None else mode == "vlm") else None
        self.engine_name = f"{engine if mode == 'ocr' else 'vlm:fake'}+{'llm' if self.llm else 'rules'}"
        FakePipe.log.append(("init", with_reader))

    def _result(self, source, page, raw):
        return ExtractionResult(source_file=source, page=page, engine=self.engine_name, data=InvoiceData(),
                                status="needs_review", raw_text=raw, ocr_confidence=0.5)

    def process_page(self, source, page, img):
        FakePipe.log.append(("page", page))
        return self._result(source, page, f"อ่านสด หน้า {page}")

    def process_raw(self, source, page, raw, meta=None):
        FakePipe.log.append(("raw", page, raw, dict(meta or {})))
        return self._result(source, page, raw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["preprocessing"] = {"keep_color": True, "min_long_side": 0}
    src = tmp_path / "data" / "raw" / "scans"
    save_image(src / "a.png", np.full((30, 50, 3), 180, dtype=np.uint8))
    FakePipe.log = []
    monkeypatch.setattr("fin_extractor.pipeline.FinanceOCRPipeline", FakePipe)
    return SimpleNamespace(cfg=cfg, src=src, data=tmp_path / "data")


def run(env, *argv):
    args = cli.build_parser().parse_args(list(argv))
    args.func(args, env.cfg)


def test_extract_writes_preprocess_raw_data_and_final_in_their_own_folders(env, capsys):
    run(env, "extract", str(env.src), "--llm", "off")
    d = env.data
    assert (d / "preprocess" / "scans" / "default" / "final" / "a_p1.png").exists()
    raw = d / "result_ocr" / "paddle" / "scans" / "raw_data"
    final = d / "result_ocr" / "paddle" / "scans" / "final" / "rules"
    assert (raw / "a_p1.txt").read_text(encoding="utf-8") == "อ่านสด หน้า 1"
    meta = json.loads((raw / "a_p1.meta.json").read_text(encoding="utf-8"))
    assert meta["profile"] == "default" and meta["ocr_confidence"] == 0.5 and meta["truncated"] is False
    result = json.loads((final / "a_p1.json").read_text(encoding="utf-8"))
    assert result["raw_text"] is None                                   # ผลสุดท้ายไม่แบกข้อความดิบ (อยู่ที่ raw_data)
    m = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert m["files"][0]["output"] == "a_p1.json" and m["settings"]["raw_data"] == "../../raw_data"
    capsys.readouterr()


def test_reuse_raw_extracts_again_without_loading_or_calling_the_reader(env, capsys):
    run(env, "extract", str(env.src), "--llm", "off")
    FakePipe.log.clear()
    run(env, "extract", str(env.src), "--llm", "on", "--reuse-raw")
    assert FakePipe.log[0] == ("init", False)                          # ไม่โหลดตัวอ่าน
    assert [e[0] for e in FakePipe.log[1:]] == ["raw"] and FakePipe.log[1][2] == "อ่านสด หน้า 1"
    base = env.data / "result_ocr" / "paddle" / "scans"
    assert (base / "final" / "llm" / "a_p1.json").exists() and (base / "final" / "rules" / "a_p1.json").exists()
    assert (base / "raw_data" / "a_p1.txt").read_text(encoding="utf-8") == "อ่านสด หน้า 1"   # ข้อความดิบไม่ถูกแตะ
    capsys.readouterr()


def test_reuse_raw_without_stored_text_skips_the_file_and_creates_nothing_else(env, capsys):
    run(env, "extract", str(env.src), "--llm", "off", "--reuse-raw")
    out = capsys.readouterr().out
    assert "ไม่มีข้อความดิบ" in out and not (env.data / "result_ocr" / "paddle" / "scans" / "final" / "rules" / "a_p1.json").exists()


def test_vlm_mode_goes_to_typhoon_folder_with_markdown_raw_data(env, capsys):
    run(env, "extract", str(env.src), "--mode", "vlm", "--llm", "off")
    base = env.data / "result_ocr" / "typhoon" / "scans"
    assert (base / "raw_data" / "a_p1.md").exists() and (base / "final" / "rules" / "a_p1.json").exists()
    capsys.readouterr()


def test_keep_raw_keeps_text_in_the_final_json_too(env, capsys):
    run(env, "extract", str(env.src), "--llm", "off", "--keep-raw")
    r = json.loads((env.data / "result_ocr" / "paddle" / "scans" / "final" / "rules" / "a_p1.json").read_text(encoding="utf-8"))
    assert r["raw_text"] == "อ่านสด หน้า 1"
    capsys.readouterr()


def test_out_moves_raw_data_and_final_together_but_not_the_shared_preprocess(env, capsys, tmp_path):
    out = tmp_path / "exp"
    run(env, "extract", str(env.src), "--llm", "off", "--out", str(out))
    assert (out / "raw_data" / "a_p1.txt").exists() and (out / "final" / "rules" / "a_p1.json").exists()
    assert (env.data / "preprocess" / "scans" / "default" / "final" / "a_p1.png").exists()
    assert not (env.data / "result_ocr").exists()
    capsys.readouterr()


def test_second_extract_with_another_engine_reuses_the_same_preprocessed_image(env, capsys, monkeypatch):
    run(env, "extract", str(env.src), "--llm", "off")
    img = env.data / "preprocess" / "scans" / "default" / "final" / "a_p1.png"
    stamp = img.stat().st_mtime_ns
    run(env, "extract", str(env.src), "--mode", "vlm", "--llm", "off")
    assert img.stat().st_mtime_ns == stamp                              # ภาพสุดท้ายชุดเดียวใช้ร่วมกันทุก engine
    capsys.readouterr()


def test_ocr_command_writes_only_raw_data(env, capsys, monkeypatch):
    class Reader:
        def recognize(self, img):
            return SimpleNamespace(text="ข้อความดิบ", mean_confidence=0.8)

    monkeypatch.setattr("fin_extractor.ocr.create_engine", lambda name, cfg: Reader())
    run(env, "ocr", str(env.src))
    base = env.data / "result_ocr" / "paddle" / "scans"
    assert (base / "raw_data" / "a_p1.txt").read_text(encoding="utf-8") == "ข้อความดิบ" and not (base / "final").exists()
    m = json.loads((base / "raw_data" / "manifest.json").read_text(encoding="utf-8"))
    assert m["command"] == "ocr" and m["files"][0]["output"] == "a_p1.txt"
    capsys.readouterr()


def test_preprocess_command_manifest_points_at_final_images(env, capsys):
    run(env, "preprocess", str(env.src), "--profile", "crop")
    pdir = env.data / "preprocess" / "scans" / "crop"
    m = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    assert m["files"][0]["output"] == "final/a_p1.png" and (pdir / "final" / "a_p1.png").exists()
    capsys.readouterr()
