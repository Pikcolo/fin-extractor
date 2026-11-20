"""ทุก engine มีตัวอ่าน (readers/) ที่สร้างได้โดยไม่โหลดโมเดล/torch และคืนผลรูปแบบเดียวกัน."""

import sys

import numpy as np

from fin_extractor.config import load_config
from fin_extractor.readers import ENGINES, TORCH_ENGINES, make_reader, reader_label
from fin_extractor.readers.lines import OCRLine, OCRResult, OCRWord


def test_every_engine_has_a_reader_without_loading_a_model():
    cfg = load_config()
    names = {e: type(make_reader(e, cfg)).__name__ for e in ENGINES}
    assert names == {"openthai": "OpenThaiOCR", "typhoon": "TyphoonOCR", "paddle": "PaddleReader", "oxr": "OXRLocal"}
    assert set(TORCH_ENGINES) == {"openthai", "oxr"}  # typhoon/paddle ต้องไม่ import torch (paddle ชน cuDNN ของ torch)
    assert {e: make_reader(e, cfg).name for e in ENGINES} == {e: reader_label(e, cfg) for e in ENGINES}


def test_paddle_reader_returns_lines_and_confidence_in_the_common_result(monkeypatch):
    class Engine:
        def __init__(self, **kw):
            pass

        def recognize(self, img):
            w = lambda t, x, c: OCRWord(t, c, (x, 0, x + 10, 10))  # noqa: E731
            return OCRResult("paddle", [OCRLine([w("รวม", 0, 0.8), w("100.00", 40, 1.0)]), OCRLine([w("VAT", 0, 0.9)])])

    monkeypatch.setattr("fin_extractor.readers.paddle.PaddleEngine", Engine)
    res = make_reader("paddle", {}).recognize(np.zeros((20, 60, 3), np.uint8))
    assert res.engine == "paddle" and res.finish_reason is None and not res.truncated
    assert res.text.splitlines()[-1] == "VAT" and "100.00" in res.text.splitlines()[0]
    assert abs(res.confidence - 0.9) < 1e-9


def test_without_modelscope_keeps_torch_out_and_cleans_up():
    from fin_extractor.readers.paddle import without_modelscope

    had = "modelscope" in sys.modules
    with without_modelscope():
        assert "modelscope" in sys.modules
    assert ("modelscope" in sys.modules) == had
