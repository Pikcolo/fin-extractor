"""ground truth ที่คนกรอก: เติมฟอร์มจากผลสกัด บันทึก ให้คะแนน และหน้าเว็บในเครื่อง (ข้อมูลจำลองทั้งหมด)."""

import copy
import importlib.util
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from fin_extractor import groundtruth as G
from fin_extractor.config import load_config
from fin_extractor.io import save_image
from fin_extractor.paths import DataPaths

ROOT = Path(__file__).resolve().parents[1]


def _result(**data):
    return {"source_file": "x.pdf", "page": 1, "engine": "openthai:fake+rules", "status": "needs_review", "data": data}


def test_normalize_compares_values_not_formatting():
    assert G.outcome("1,940.00", "1940", "total_amount") == "correct"
    assert G.outcome("0-9055-64002-79-4", "0905564002794", "buyer.tax_id") == "correct"
    assert G.outcome("บริษัท อิมพอร์ต สตาร์ จํากัด", "บริษัทอิมพอร์ตสตาร์จำกัด", "buyer.name") == "correct"  # ช่องว่าง + สระอำสองตัว
    assert G.outcome("2026-10-05", "2026-10-05", "date") == "correct"
    assert G.outcome("788.90", "78.90", "total_amount") == "wrong"
    assert G.outcome("INV-1", None, "invoice_no") == "missed"
    assert G.outcome(None, "INV-1", "invoice_no") == "spurious"
    assert G.outcome(None, "", "invoice_no") == "both_empty"


def test_score_counts_only_reviewed_fields_and_items_when_checked():
    gt = {"fields": {"invoice_no": "A1", "total_amount": "100.00", "discount": None},
          "document_fields": {"total_paid": "50"}, "items": [{"description": "x", "amount": "100"}], "items_checked": True}
    res = _result(invoice_no="A1", total_amount="10.00", discount="5.00", document_fields={"total_paid": "50.00"},
                  items=[{"description": "x", "amount": "100.00"}, {"description": "y", "amount": "1"}])
    s = G.score(gt, res, {"total_paid": "money"})
    assert s["invoice_no"] == "correct" and s["total_amount"] == "wrong" and s["discount"] == "spurious"
    assert s["document_fields.total_paid"] == "correct" and s["items.count"] == "wrong" and s["items[0].amount"] == "correct"
    assert "vendor.name" not in s  # ไม่ได้ตรวจ = ไม่นับ
    summ = G.summarize([s])
    assert summ["present"] == summ["correct"] + summ["wrong"] + summ["missed"] and summ["spurious"] == 1
    assert G.score({"fields": {"invoice_no": "A1"}}, None) == {"invoice_no": "missed"}


@pytest.fixture
def env(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    paths = DataPaths(cfg)
    (tmp_path / "data" / "01_input" / "scans").mkdir(parents=True)
    save_image(paths.preprocess_final("scans") / "a_p1.png", np.full((40, 30, 3), 200, np.uint8))
    out = paths.final("openthai", "scans", "rules", "luma") / "a_p1.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(_result(invoice_no="A1", total_amount="100.00", vendor={"name": "ร้าน ก"})), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("groundtruth_app", ROOT / "scripts" / "eval" / "groundtruth_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.load_config = lambda: cfg  # ชี้ไปที่ข้อมูลจำลอง
    return mod, paths, tmp_path


def test_app_prefills_from_the_extraction_and_saves_reviewed_values(env):
    mod, paths, tmp = env
    app = mod.App(str(tmp / "data" / "01_input" / "scans"), "openthai/luma/rules")
    assert [d["stem"] for d in app.docs()] == ["a"] and app.docs()[0]["status"] == "new"
    doc = app.doc("a")
    assert doc["gt"]["fields"]["invoice_no"] == "A1" and doc["gt"]["fields"]["vendor.name"] == "ร้าน ก" and doc["gt"]["status"] == "new"
    fields = dict(doc["gt"]["fields"], total_amount="107.00", junk="x")
    app.save("a", {"status": "done", "fields": fields, "items": [{"description": "", "amount": ""}], "items_checked": True})
    gt = G.load_gt(paths, "scans", "a")
    assert gt["status"] == "done" and gt["fields"]["total_amount"] == "107.00" and "junk" not in gt["fields"]
    assert gt["items"] == [] and app.docs()[0]["status"] == "done"
    assert G.score(gt, G.load_result(paths, "scans", "openthai/luma/rules", "a"))["total_amount"] == "wrong"


def test_server_listens_on_localhost_only_and_rejects_unknown_files(env):
    mod, _paths, tmp = env
    app = mod.App(str(tmp / "data" / "01_input" / "scans"), "openthai/luma/rules")
    server = mod.ThreadingHTTPServer(("127.0.0.1", 0), mod.handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        base = f"http://127.0.0.1:{port}"
        assert "Ground truth" in urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert json.loads(urllib.request.urlopen(base + "/api/doc/a").read())["stem"] == "a"
        assert urllib.request.urlopen(base + "/img/a").read()[:4] == b"\x89PNG"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/img/..%2F..%2Fsecret")
    finally:
        server.shutdown()
