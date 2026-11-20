"""OXR-1.0 (SoMarkAI) โหลดตรงด้วย Transformers — ไม่ต้องใช้ Docker/vLLM (image 19.3GB) มีแค่ weights ~3GB

ใช้อ่าน **ภาพย่อย (section)** ทีละชิ้น ไม่ใช่ทั้งหน้า: โมเดลนี้ถูกออกแบบให้อ่านทีละบล็อกที่ตัวตรวจ layout ตัดมาให้ ซึ่งเรามีตัวแบ่ง
section ของเราเองอยู่แล้ว (notebooks/preprocess_lab) จึงไม่ใช้ตัวตรวจ layout ของ OXR

**OXR ไม่รู้ภาษาไทย** (ผู้พัฒนาระบุเอง และ docs/archive/oxr.md วัดแล้ว: ช่องที่มีไทยได้ข้อความมั่ว + อักษรจีน) ใช้เป็นตัวอ่านสำรองเฉพาะส่วนที่เป็น
ตัวเลข/อังกฤษ ส่วนที่มีไทยให้ตัวอ่านที่รู้ภาษาไทย (Paddle / Typhoon / OpenThai)

ข้อควรระวังที่เจอจริงตอนทำ
- ``generation_config`` ของโมเดลตั้ง eos เป็น ``<|endoftext|>`` (151643) แต่โมเดลจบคำตอบด้วย ``<|im_end|>`` (151645) ถ้าไม่ระบุ eos เอง
  จะสร้าง token จนครบ ``max_new_tokens`` ทุกครั้ง (วัดแล้ว ~18 วินาทีต่อภาพย่อย เทียบ 0.35 วินาทีเมื่อระบุ)
- โหลดผ่าน ``AutoModel`` ตรง ๆ ล้ม (Transformers คัดลอกโค้ดของโมเดลสองชุดคนละ hash แล้วคลาส config ไม่ตรงกัน) จึงโหลดโค้ดจากโฟลเดอร์เป็นแพ็กเกจเดียว
"""

from __future__ import annotations

import sys
import types
from typing import Any

import cv2
import numpy as np

from ..config import resolve_path
from .base import ReadResult

PROMPT = "Recognize the content of the image."  # prompt ตายตัวของ OXR (oxr/model/prompts.py ในรีโปของเขา)
STOP_TOKEN = "<|im_end|>"
_PACKAGE = "oxr_model_code"


def restore_indicator_buffer(model) -> None:
    """ตั้ง ``indicator_token_indices`` ให้ถูกหลังโหลด

    buffer นี้สร้างด้วย ``torch.arange`` ใน ``__init__`` และไม่อยู่ในไฟล์ weights (non-persistent) เมื่อโหลดด้วย Transformers 5.x
    ที่สร้างโมเดลแบบว่างก่อนแล้วค่อยเติม ค่าของมันไม่ถูกเติม: ได้ ``[0,0,0,0]`` หรือค่าขยะในหน่วยความจำ (วัดแล้วทั้งสองแบบ)
    ค่าขยะเกินขอบเขตตาราง embedding → ``device-side assert`` ที่ภาพแรก (cuDNN ``DEVICE_ALLOCATION_FAILED`` เป็นแค่อาการตามหลัง)
    ส่วนค่า ``[0,0,0,0]`` ไม่พัง แต่ token บอกขอบภาพเริ่ม/จบจะใช้ embedding ผิดตัวเงียบ ๆ
    (``inv_freq`` ของ rotary ในตัวเข้ารหัสภาพก็ผิดเหมือนกัน แต่ ``forward`` คำนวณใหม่เองทุกครั้ง จึงไม่มีผล)
    """
    import torch

    size = model.indicator_token_indices.numel()
    vocab = model.config.visual_vocab_size
    model.indicator_token_indices = torch.arange(vocab - size, vocab, dtype=torch.long)
    assert model.indicator_token_indices.tolist() == list(range(vocab - size, vocab))


class OXRLocal:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model_dir = resolve_path(cfg["model_dir"])
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return f"oxr:{self.model_dir.name}"

    def _load(self):
        if self._model is not None:
            return self._model
        if not (self.model_dir / "modeling_oxr.py").is_file():
            raise FileNotFoundError(
                f"ไม่พบโมเดล OXR ที่ {self.model_dir} — hf download SoMarkAI/OXR-1.0 --local-dir <model_dir> (~3GB เก็บนอก repo ได้)")
        import torch

        self._device = self.cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        if _PACKAGE not in sys.modules:  # โหลดโค้ดของโมเดลเป็นแพ็กเกจเดียว (ดูหัวไฟล์)
            pkg = types.ModuleType(_PACKAGE)
            pkg.__path__ = [str(self.model_dir)]
            sys.modules[_PACKAGE] = pkg
        import importlib

        config_mod = importlib.import_module(f"{_PACKAGE}.configuration_oxr")
        model_mod = importlib.import_module(f"{_PACKAGE}.modeling_oxr")
        config = config_mod.OXRConfig.from_pretrained(str(self.model_dir))
        model = model_mod.OXR.from_pretrained(str(self.model_dir), config=config, dtype=dtype)
        restore_indicator_buffer(model)
        self._model = model.to(self._device).eval()
        self._dtype = dtype
        return self._model

    def recognize(self, img: np.ndarray) -> ReadResult:
        """ภาพ BGR (ภาพย่อยหนึ่งส่วน) → ข้อความ ``finish_reason='length'`` เมื่อชน max_new_tokens (ข้อความอาจไม่ครบหรือวนซ้ำ)."""
        import torch
        from PIL import Image

        model = self._load()
        page = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [{"type": "image", "image": page}, {"type": "text", "text": PROMPT}]}]
        input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages)
        stop_id = model.text_tokenizer.convert_tokens_to_ids(STOP_TOKEN)
        with torch.inference_mode():
            out = model.generate(
                input_ids.to(self._device), pixel_values=pixel_values.to(self._device, self._dtype),
                grid_thws=grid_thws.to(self._device), do_sample=False,
                max_new_tokens=self.max_new_tokens, eos_token_id=stop_id,
            )
        tokens = out[0].tolist()
        truncated = len(tokens) >= self.max_new_tokens and stop_id not in tokens
        text = model.text_tokenizer.decode(tokens, skip_special_tokens=True)
        return ReadResult(self.name, text.strip(), "length" if truncated else "stop")
