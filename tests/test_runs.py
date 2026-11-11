"""manifest ของการรัน (runs.py) — โฟลเดอร์ผลมาจาก paths.py ผู้เรียกส่งเข้ามา."""

import json

from fin_extractor.runs import Run

CFG = {"preprocessing": {"deskew": True}}


def test_manifest_records_run_into_the_given_folder(tmp_path):
    out = tmp_path / "result_ocr" / "typhoon" / "scans" / "final" / "llm"
    with Run("extract", tmp_path, CFG, out, settings={"mode": "vlm"}) as run:
        run.sub("extra")
        run.add(file="a.pdf", page=1, status="needs_review", output="a_p1.json")
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (out / "extra").is_dir() and m["status"] == "completed" and m["command"] == "extract"
    assert m["settings"] == {"mode": "vlm"} and m["files"][0]["status"] == "needs_review"
    assert m["schema_version"] and m["preprocessing"] == {"deskew": True}


def test_manifest_is_written_as_failed_when_the_run_crashes(tmp_path):
    try:
        with Run("ocr", tmp_path, CFG, tmp_path / "r") as run:
            run.add(file="a.pdf", page=1)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    m = json.loads((tmp_path / "r" / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed" and m["files"][0]["file"] == "a.pdf"


def test_rerun_into_the_same_folder_overwrites_the_manifest(tmp_path):
    out = tmp_path / "r"
    for status in ("first", "second"):
        with Run("ocr", tmp_path, CFG, out) as run:
            run.add(file=status)
    assert [f["file"] for f in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["files"]] == ["second"]
