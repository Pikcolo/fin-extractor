"""ทุก engine มีตัวอ่านรายส่วน (sections/read.py) และสร้างได้โดยไม่โหลดโมเดล/torch."""

import sys

import numpy as np

from fin_extractor.config import load_config
from fin_extractor.ocr.base import OCRLine, OCRResult, OCRWord
from fin_extractor.sections import read


def test_every_engine_has_a_section_reader_without_loading_a_model():
    cfg = load_config()
    names = {e: type(read.make_reader(e, cfg)).__name__ for e in read.ENGINES}
    assert names == {"openthai": "OpenThaiOCR", "typhoon": "TyphoonOCR", "paddle": "PaddleSectionReader", "oxr": "OXRLocal"}
    assert set(read.TORCH_ENGINES) == {"openthai", "oxr"}  # typhoon/paddle ต้องไม่ import torch (paddle ชน cuDNN ของ torch)


def test_paddle_section_reader_returns_lines_as_text_in_the_vlm_result_shape(monkeypatch):
    class Engine:
        def recognize(self, img):
            w = lambda t, x: OCRWord(t, 0.9, (x, 0, x + 10, 10))  # noqa: E731
            return OCRResult("paddle", [OCRLine([w("รวม", 0), w("100.00", 40)]), OCRLine([w("VAT", 0)])])

    monkeypatch.setattr("fin_extractor.ocr.create_engine", lambda name, cfg: Engine())
    res = read.PaddleSectionReader({}).recognize(np.zeros((20, 60, 3), np.uint8))
    assert res.engine == "paddle" and res.finish_reason is None and not res.truncated
    assert res.markdown.splitlines()[-1] == "VAT" and "100.00" in res.markdown.splitlines()[0]


def test_without_modelscope_keeps_torch_out_and_cleans_up():
    from fin_extractor.ocr.paddle_engine import without_modelscope

    had = "modelscope" in sys.modules
    with without_modelscope():
        assert "modelscope" in sys.modules
    assert ("modelscope" in sys.modules) == had
