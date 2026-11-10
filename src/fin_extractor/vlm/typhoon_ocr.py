"""Typhoon OCR (VLM) — อ่านภาพเอกสารทั้งหน้าเป็น Markdown (ตารางเป็น HTML).

ต่างจาก OCR แบบดั้งเดิมตรงที่เข้าใจ layout/ตาราง และภาษาไทยได้ดีมาก
แต่ไม่มี bounding box และไม่มี confidence ต่อคำ

ใช้ได้ทั้ง
- Ollama ในเครื่อง : ``ollama pull scb10x/typhoon-ocr1.5-3b``
- vLLM            : ``vllm serve <hf-repo>``                       (provider: openai)
- Typhoon API     : https://api.opentyphoon.ai/v1 + TYPHOON_API_KEY  (provider: openai)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .client import ChatClient
from .prompts import TYPHOON_OCR_PROMPT


@dataclass
class VLMResult:
    engine: str
    markdown: str
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _strip_wrappers(text: str) -> str:
    # บางรุ่นห่อผลด้วย ```markdown ... ``` หรือ JSON {"natural_text": ...}
    text = text.strip()
    m = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    if text.startswith("{") and '"natural_text"' in text:
        try:
            text = json.loads(text)["natural_text"]
        except (ValueError, KeyError):
            pass
    return text


class TyphoonOCR:
    def __init__(self, cfg: dict[str, Any], prompt: str = TYPHOON_OCR_PROMPT):
        self.cfg = cfg
        self.prompt = prompt
        self.client = ChatClient(cfg)

    @property
    def name(self) -> str:
        return f"vlm:{self.cfg['model']}"

    def recognize(self, img: np.ndarray) -> VLMResult:
        resp = self.client.chat([{"role": "user", "content": self.prompt}], images=[img])
        return VLMResult(self.name, _strip_wrappers(resp.content), resp.finish_reason)
