"""ตัวอ่านรอบสอง (OpenThai Text): อ่านซ้ำที่ขนาดตัวตรวจจับใหญ่ขึ้นเมื่อไม่เจอบรรทัด · เทียบยอดเงินเป็นตัวเลข · ผลที่เก็บไว้ผูกกับค่าตั้งของตัวอ่าน (ใช้ตัวอ่านปลอม ไม่โหลดโมเดล)."""

import json

from fin_extractor.readers.openthai_text import OpenThaiTextOCR
from fin_extractor.sections.coverage import _load, missing_values
from fin_extractor.sections.read import fingerprint


class FakeLineOCR:
    """คืนบรรทัดเฉพาะเมื่อ ``det.limit`` ถึงเกณฑ์ที่กำหนด (จำลองตัวตรวจจับที่ต้องขยาย) และจำ limit ของทุกครั้งที่ถูกเรียก."""

    def __init__(self, need_limit: int | None):
        self.cfg = {"det": {"limit": 1280}}
        self.need = need_limit
        self.limits: list[int] = []

    def read(self, _img):
        self.limits.append(self.cfg["det"]["limit"])
        if self.need is None or self.cfg["det"]["limit"] < self.need:
            return {"text": "", "lines": []}
        return {"text": "ก 1.00", "lines": [{"text": "ก 1.00", "confidence": 0.9}]}


def reader(cfg: dict, fake: FakeLineOCR) -> OpenThaiTextOCR:
    r = OpenThaiTextOCR({"model_dir": "models/none", **cfg})
    r._ocr = fake
    return r


def test_empty_result_is_read_again_at_the_larger_detector_size_and_the_setting_is_restored():
    fake = FakeLineOCR(need_limit=2000)
    out = reader({"retry_empty_det_limit": 2560}, fake).read(object())
    assert out["lines"] and fake.limits == [1280, 2560]
    assert fake.cfg["det"]["limit"] == 1280  # ไม่ค้างค่าที่ขยาย (ภาพถัดไปไม่ถูกอ่านที่ขนาดใหญ่โดยไม่จำเป็น)


def test_a_result_with_lines_is_not_read_twice():
    fake = FakeLineOCR(need_limit=1000)
    reader({"retry_empty_det_limit": 2560}, fake).read(object())
    assert fake.limits == [1280]


def test_no_retry_when_the_setting_is_absent_and_a_second_empty_result_is_kept():
    fake = FakeLineOCR(need_limit=None)
    assert reader({}, fake).read(object())["lines"] == [] and fake.limits == [1280]
    fake = FakeLineOCR(need_limit=None)
    assert reader({"retry_empty_det_limit": 2560}, fake).read(object())["lines"] == [] and fake.limits == [1280, 2560]
    assert fake.cfg["det"]["limit"] == 1280


def test_recognize_uses_the_retry():
    fake = FakeLineOCR(need_limit=2000)
    assert reader({"retry_empty_det_limit": 2560}, fake).recognize(object()).text == "ก 1.00"


def test_an_amount_without_decimals_is_the_same_value_not_a_missing_one():
    assert missing_values("รวมเงิน 106,250.00", "รวมเงิน 106,250")["money"] == []
    assert missing_values("รวมเงิน 106,250.00", "รวมเงิน 106,250.50")["money"] == ["106250.50"]  # ค่าต่างจริงยังถูกชี้
    assert missing_values("ยอด 7,500.00", "ยอด 7,500:00")["money"] == []


def test_saved_second_reader_result_is_reused_only_for_the_same_reader_settings(tmp_path):
    layout = [{"id": "s00", "kind": "strip_00", "bbox": [0, 0, 10, 10], "table": None, "keep": True}]
    path = tmp_path / "x.coverage.json"
    path.write_text(json.dumps({"layout": fingerprint(layout), "reader": "A", "sections": []}), encoding="utf-8")
    assert _load(path, layout, False, "A") is not None
    assert _load(path, layout, False, "B") is None  # ค่าตั้งของตัวอ่านเปลี่ยน → อ่านใหม่
    assert _load(path, layout, False) is not None  # ไม่ระบุค่าตั้ง = เทียบแค่ผัง (ไฟล์ .split.json)
    assert _load(path, layout, True, "A") is None  # redo
