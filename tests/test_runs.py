"""โครงเก็บผลของทุกคำสั่ง (runs.py)."""

import json

from fin_extractor.runs import Run

CFG = {"paths": {"output_dir": "OUT"}, "preprocessing": {"deskew": True}}


def test_default_dir_is_command_input_tag(tmp_path):
    cfg = {**CFG, "paths": {"output_dir": str(tmp_path)}}
    folder = tmp_path / "scans" / "finance doc"
    folder.mkdir(parents=True)
    with Run("extract", folder, "ocr-rules", cfg) as run:
        assert run.dir == tmp_path / "extract" / "finance_doc__ocr-rules"
    (folder / "a.pdf").touch()
    file_run = Run("ocr", folder / "a.pdf", "paddle", cfg)
    assert file_run.dir == tmp_path / "ocr" / "a__paddle"
    assert Run("preprocess", folder, None, cfg).dir == tmp_path / "preprocess" / "finance_doc"


def test_out_overrides_whole_dir_and_manifest_records_run(tmp_path):
    out = tmp_path / "mine"
    with Run("extract", tmp_path, "vlm-llm", CFG, out=out, settings={"mode": "vlm"}) as run:
        run.sub("results")
        run.add(file="a.pdf", page=1, status="needs_review")
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (out / "results").is_dir() and m["status"] == "completed" and m["command"] == "extract"
    assert m["settings"] == {"mode": "vlm"} and m["files"][0]["status"] == "needs_review"
    assert m["schema_version"] and m["preprocessing"] == {"deskew": True}


def test_manifest_is_written_as_failed_when_the_run_crashes(tmp_path):
    try:
        with Run("ocr", tmp_path, "paddle", CFG, out=tmp_path / "r") as run:
            run.add(file="a.pdf", page=1)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    m = json.loads((tmp_path / "r" / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed" and len(m["files"]) == 1
