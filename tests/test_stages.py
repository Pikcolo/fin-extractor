"""stages.py: ภาพที่ผ่าน preprocess เก็บแล้วใช้ซ้ำ + ข้อความดิบ (raw_data)."""

import copy
import json

import numpy as np
import pytest
from PIL import Image

from fin_extractor import stages
from fin_extractor.config import load_config
from fin_extractor.io import save_image
from fin_extractor.stages import load_raw, prepared_pages, raw_pages, save_raw

SIMPLE = {"keep_color": True, "min_long_side": 0}


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["preprocessing"] = dict(SIMPLE)  # ไม่เรียกตัวจำแนกทิศของ Paddle (ไม่ใช้ GPU ในเทสต์)
    return c


@pytest.fixture
def counted(monkeypatch):
    """นับจำนวนครั้งที่ preprocess ทำงานจริง."""
    calls = []
    real = stages.preprocess

    def spy(img, cfg, keep_steps=False):
        calls.append(1)
        return real(img, cfg, keep_steps=keep_steps)

    monkeypatch.setattr(stages, "preprocess", spy)
    return calls


def make_image(path, shade=200):
    save_image(path, np.full((40, 60, 3), shade, dtype=np.uint8))
    return path


def make_pdf(path, pages=2):
    imgs = [Image.new("RGB", (80, 100), (30 * i % 255, 120, 200)) for i in range(1, pages + 1)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:])
    return path


def test_first_call_writes_final_image_and_settings_second_call_reuses_them(tmp_path, cfg, counted):
    src = make_image(tmp_path / "raw" / "scans" / "a.png")
    pdir = tmp_path / "pre" / "scans" / "default"
    first = list(prepared_pages(src, cfg, pdir))
    assert [p for p, _, _ in first] == [1] and (pdir / "final" / "a_p1.png").exists() and (pdir / "settings.json").exists()
    assert len(counted) == 1
    again = list(prepared_pages(src, cfg, pdir))
    assert len(counted) == 1 and again[0][2] == {"cached": True}      # ไม่ preprocess ซ้ำ
    assert np.array_equal(first[0][1], again[0][1])                   # ได้ภาพเดียวกันเป๊ะ (PNG ไม่สูญเสีย)


def test_steps_flag_recomputes_and_saves_every_step(tmp_path, cfg, counted):
    src = make_image(tmp_path / "a.png")
    pdir = tmp_path / "p"
    list(prepared_pages(src, cfg, pdir))
    list(prepared_pages(src, cfg, pdir, steps=True))
    assert len(counted) == 2 and any((pdir / "steps").glob("a_p1_*original*.png"))


def test_changed_settings_need_a_new_profile_or_force(tmp_path, cfg):
    src = make_image(tmp_path / "a.png")
    other = make_image(tmp_path / "b.png")
    pdir = tmp_path / "p"
    list(prepared_pages(src, cfg, pdir))
    list(prepared_pages(other, cfg, pdir))
    cfg2 = copy.deepcopy(cfg)
    cfg2["preprocessing"]["clahe"] = True
    with pytest.raises(ValueError, match="--profile"):
        list(prepared_pages(src, cfg2, pdir))                          # ไม่ปนภาพคนละตั้งค่าใน profile เดียว
    assert (pdir / "final" / "b_p1.png").exists()                       # ล้มโดยยังไม่ลบอะไร
    list(prepared_pages(src, cfg2, pdir, force=True))                   # --force = ล้างแล้วทำใหม่
    assert not (pdir / "final" / "b_p1.png").exists() and (pdir / "final" / "a_p1.png").exists()
    assert json.loads((pdir / "settings.json").read_text(encoding="utf-8"))["preprocessing"]["clahe"] is True


def test_multi_page_pdf_is_cached_only_when_every_page_exists(tmp_path, cfg, counted):
    src = make_pdf(tmp_path / "doc.pdf", pages=2)
    pdir = tmp_path / "p"
    assert [p for p, _, _ in prepared_pages(src, cfg, pdir)] == [1, 2] and len(counted) == 2
    assert [p for p, _, m in prepared_pages(src, cfg, pdir)] == [1, 2] and len(counted) == 2   # ครบทุกหน้า ใช้แคช
    (pdir / "final" / "doc_p2.png").unlink()
    assert [p for p, _, _ in prepared_pages(src, cfg, pdir)] == [1, 2] and len(counted) == 4   # หน้าหาย = ทำใหม่ทั้งไฟล์


def test_save_and_load_raw_round_trip_with_meta(tmp_path):
    out = save_raw(tmp_path / "raw_data", "a_p1", "สวัสดี\nบรรทัดสอง", "txt", {"ocr_confidence": 0.9, "truncated": False})
    assert out.name == "a_p1.txt"
    text, meta = load_raw(out)
    assert text == "สวัสดี\nบรรทัดสอง" and meta == {"ocr_confidence": 0.9, "truncated": False}
    (tmp_path / "raw_data" / "b_p1.txt").write_text("x", encoding="utf-8")
    assert load_raw(tmp_path / "raw_data" / "b_p1.txt") == ("x", {})   # ไม่มี .meta.json = meta ว่าง


def test_raw_pages_are_sorted_by_page_number_and_ignore_other_files(tmp_path):
    d = tmp_path / "raw_data"
    for stem in ("scan_p1", "scan_p2", "scan_p10", "scan_p1x", "other_p1", "scan_p2.extra"):
        save_raw(d, stem, "t", "txt", {})
    assert [n for n, _ in raw_pages(d, "scan", "txt")] == [1, 2, 10]
    assert raw_pages(d, "scan", "md") == []                             # ต่างชนิด (txt/md) ไม่ปนกัน
    save_raw(d, "a_p1_p2", "t", "txt", {})                              # ชื่อไฟล์ที่มี _p อยู่ในตัวเอง
    assert [n for n, _ in raw_pages(d, "a_p1", "txt")] == [2] and raw_pages(d, "a", "txt") == []


# ---------- pipeline: สกัดซ้ำจากข้อความดิบ (ไม่โหลดตัวอ่าน) ----------

def _pipe(mode):
    from fin_extractor.pipeline import FinanceOCRPipeline

    return FinanceOCRPipeline(load_config(), mode=mode, engine="paddle", use_llm=False, with_reader=False)


def test_pipeline_without_reader_keeps_the_engine_name_of_a_live_run():
    assert _pipe("ocr").engine_name == "paddle+rules"
    assert _pipe("vlm").engine_name == f"vlm:{load_config()['vlm']['model']}+rules"


def test_lines_from_raw_matches_how_each_engine_produces_lines():
    assert _pipe("ocr").lines_from_raw("บรรทัดหนึ่ง\n\n  \nบรรทัดสอง\n") == ["บรรทัดหนึ่ง", "บรรทัดสอง"]
    md = "<table><tr><td>รวม</td><td>100.00</td></tr></table>\n\nข้อความ"
    assert _pipe("vlm").lines_from_raw(md)[-1] == "ข้อความ" and "100.00" in _pipe("vlm").lines_from_raw(md)[0]


def test_process_raw_gives_the_same_result_as_extracting_those_lines_and_adds_meta_checks():
    p = _pipe("ocr")
    text = "ใบกำกับภาษี\nบริษัท ผู้ขาย จำกัด\nรวมทั้งสิ้น 107.00"
    r = p.process_raw("a.pdf", 2, text, {"ocr_confidence": 0.9123456})
    direct = p.build_result("a.pdf", 2, p.lines_from_raw(text), text, 0.9123456, [])
    assert r.page == 2 and r.ocr_confidence == 0.9123 and r.raw_text == text
    assert r.data == direct.data and r.status == direct.status and r.checks == direct.checks
    truncated = p.process_raw("a.pdf", 1, text, {"truncated": True})
    assert any(c.name == "vlm_complete" and c.passed is False for c in truncated.checks)   # ตอบไม่จบจากตอนอ่านสด ติดมากับ meta
    assert not any(c.name == "vlm_complete" for c in p.process_raw("a.pdf", 1, text).checks)  # ไม่มี meta = ไม่มีธง
