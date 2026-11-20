"""Typhoon OCR — อ่านภาพ (section) เป็น Markdown (ตารางเป็น HTML) ผ่าน Ollama / vLLM / Typhoon API.

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
from typing import Any

import numpy as np

from ..llm_client import ChatClient
from .base import ReadResult

# อิงรูปแบบ prompt ของ Typhoon OCR 1.5 (ดู model card: huggingface.co/scb10x)
# ถ้าเปลี่ยนรุ่นโมเดล ให้เทียบ prompt กับ model card ของรุ่นนั้นอีกครั้ง
TYPHOON_OCR_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


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
        return f"typhoon:{self.cfg['model']}"

    def recognize(self, img: np.ndarray) -> ReadResult:
        resp = self.client.chat([{"role": "user", "content": self.prompt}], images=[img])
        return ReadResult(self.name, _strip_wrappers(resp.content), resp.finish_reason)
