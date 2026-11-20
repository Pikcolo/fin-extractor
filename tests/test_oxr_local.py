"""readers/oxr.py: ไม่โหลดโมเดลจริง (3GB/GPU) ทดสอบเฉพาะตรรกะรอบ ๆ ด้วยโมเดลปลอม."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fin_extractor.readers.oxr import OXRLocal, STOP_TOKEN  # noqa: E402


class _Tok:
    def convert_tokens_to_ids(self, t):
        assert t == STOP_TOKEN
        return 99

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if i != 99) + " "


class _Model:
    text_tokenizer = _Tok()

    def __init__(self, out):
        self.out, self.kwargs = out, None

    def preprocess_inputs(self, messages):
        assert messages[0]["content"][1]["text"] == "Recognize the content of the image."
        return torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 4), torch.ones(1, 3, dtype=torch.long)

    def generate(self, ids, **kw):
        self.kwargs = kw
        return torch.tensor([self.out])


def _reader(out, max_new=8):
    r = OXRLocal({"model_dir": "models/none", "max_new_tokens": max_new})
    r._model, r._device, r._dtype = _Model(out), "cpu", torch.float32
    return r


IMG = np.zeros((10, 10, 3), np.uint8)


def test_missing_weights_raise_a_clear_error_without_importing_the_model():
    with pytest.raises(FileNotFoundError, match="hf download SoMarkAI/OXR-1.0"):
        OXRLocal({"model_dir": "models/does-not-exist"}).recognize(IMG)


def test_stop_token_is_passed_as_eos_so_generation_ends_early():
    r = _reader([5, 6, 99])
    res = r.recognize(IMG)
    assert r._model.kwargs["eos_token_id"] == 99 and r._model.kwargs["do_sample"] is False
    assert res.text == "5 6" and res.finish_reason == "stop" and not res.truncated


def test_hitting_the_token_cap_without_a_stop_token_is_flagged_truncated():
    res = _reader([1, 2, 3, 4], max_new=4).recognize(IMG)
    assert res.finish_reason == "length" and res.truncated


def test_indicator_buffer_is_restored_after_loading():
    from types import SimpleNamespace

    from fin_extractor.readers.oxr import restore_indicator_buffer

    # Transformers 5.x ไม่เติม non-persistent buffer: ได้ [0,0,0,0] หรือค่าขยะเกินขอบเขต
    for broken in ([0, 0, 0, 0], [3444631011552, 0, 0, 0]):
        model = torch.nn.Module()
        model.config = SimpleNamespace(visual_vocab_size=65536)
        model.register_buffer("indicator_token_indices", torch.tensor(broken), persistent=False)
        restore_indicator_buffer(model)
        assert model.indicator_token_indices.tolist() == [65532, 65533, 65534, 65535]
        assert "indicator_token_indices" in dict(model.named_buffers())  # ยังเป็น buffer (ย้ายตาม .to(device))
