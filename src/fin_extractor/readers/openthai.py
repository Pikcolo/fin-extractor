"""OpenThai 2.0 OCR Markdown (iApp) — Thai fine-tune ของ Nemotron 3.5 Parse (0.63B), อ่านภาพส่วนเป็น Markdown (ตารางเป็น HTML).

ต่างจาก Typhoon ตรงที่ output มีกล่อง+คลาสของทุกบล็อกด้วย เก็บเป็น ``ReadResult.blocks`` (ขั้นสกัดยังใช้แค่ Markdown)

ใช้ได้ 2 แบบ (ตั้งใน ``openthai:`` ของ config)
- ``backend: transformers`` : โหลดโมเดลในโปรเซสนี้จาก ``model_dir`` (ไม่ต้องใช้ vLLM ซึ่งรันบน Windows ตรง ๆ ไม่ได้)
- ``backend: vllm``         : เรียก ``vllm serve <model_dir> --served-model-name nemotron-parse --trust-remote-code`` ผ่าน ``base_url``

โค้ดอ่านผล (``nemotron_parse``) อยู่ในโฟลเดอร์โมเดลเอง: ``hf download iapp/OpenThai2.0-OCR-Markdown --local-dir <model_dir>``
ไม่ต้อง pip install แพ็กเกจนั้น แค่มี torch/torchvision/transformers/timm/einops/accelerate
"""

from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..config import resolve_path
from .base import ReadResult


class OpenThaiOCR:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model_dir = resolve_path(cfg["model_dir"])
        self.model = self.model_dir.name
        self._parser = None

    @property
    def name(self) -> str:
        return f"openthai:{self.model}"

    def _load(self):
        if self._parser is not None:
            return self._parser
        if not (self.model_dir / "nemotron_parse").is_dir():
            raise FileNotFoundError(f"ไม่พบโมเดลที่ {self.model_dir} — hf download iapp/OpenThai2.0-OCR-Markdown --local-dir <model_dir>")
        sys.path.insert(0, str(self.model_dir))  # ใช้ nemotron_parse ที่มากับโมเดลโดยไม่ต้อง pip install
        from nemotron_parse.inference import DocumentParser, GenerationSettings

        # greedy: การ sampling ทำให้หน้าเพี้ยนในการทดสอบของผู้ทำโมเดล (บน Mac) — ใช้ค่าเดียวกันทุก backend
        gen = GenerationSettings(
            temperature=0, repetition_penalty=self.cfg.get("repetition_penalty", 1.1),
            max_tokens=self.cfg.get("max_tokens", 8744), top_k=1)
        if self.cfg.get("backend", "transformers") == "vllm":
            self._parser = DocumentParser(base_url=self.cfg["base_url"], model="nemotron-parse", fullpage=gen)
        else:
            self._parser = DocumentParser(backend=self._transformers_backend(), fullpage=gen)
        return self._parser

    def _transformers_backend(self):
        import torch
        from nemotron_parse.inference import TransformersBackend
        from transformers import AutoModel, AutoProcessor

        device = self.cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(self.model_dir, trust_remote_code=True, dtype=dtype).to(device).eval()
        processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)

        class Backend(TransformersBackend):
            def __call__(self, image, *, prompt, stage, settings):
                # vLLM อ่าน prompt ต่อจากขึ้นบรรทัดใหม่ — ส่ง token ชุดเดียวกันตอนรันในเครื่อง (ตามตัวอย่างของ iApp)
                return super().__call__(image, prompt="\n" + prompt, stage=stage, settings=settings)

        return Backend(model, processor)

    def recognize(self, img: np.ndarray) -> ReadResult:
        parser = self._load()  # ต้องมาก่อน import nemotron_parse (เพิ่ม path ของโมเดลที่นี่)
        from nemotron_parse.inference import GenerationError

        page = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        try:
            doc = parser.parse(page, mode=self.cfg.get("parse_mode", "single"))
            return ReadResult(self.name, doc["markdown"], "stop", blocks=_blocks(doc))
        except GenerationError as e:  # ตอบไม่จบ (ชน max_tokens) — เก็บเท่าที่อ่านได้ แล้วติดธง truncated
            from nemotron_parse.postprocessing import clean_raw_output, postprocess_document

            raw = clean_raw_output(e.response.get("raw_out") or "")
            if not raw.strip():
                return ReadResult(self.name, "", "length", blocks=[])
            doc = postprocess_document(raw, image_size=page.size, options=parser.options)
            return ReadResult(self.name, doc["markdown"], "length", blocks=_blocks(doc))


def _blocks(doc: dict) -> list[dict]:
    """กรอบ+ชนิดของทุกบล็อกที่โมเดลทาย (prompt ขอ ``<predict_bbox><predict_classes>`` อยู่แล้ว) ``bbox_pixels`` = (x0, y0, x1, y1) บนภาพที่ส่งเข้า."""
    out = []
    for b in doc.get("blocks") or []:
        if not b.get("bbox_pixels"):
            continue
        x0, y0, x1, y1 = (round(v) for v in b["bbox_pixels"])
        out.append({"class": b.get("class"), "box": [x0, y0, x1 - x0, y1 - y0], "text": b.get("content") or b.get("text") or ""})
    return out
