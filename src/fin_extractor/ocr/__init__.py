"""OCR layer — เลือก engine ผ่าน ``create_engine(name, cfg)``.

import engine แบบ lazy เพื่อไม่ต้องติดตั้งทุก library (paddle หนัก)
"""

from __future__ import annotations

from typing import Any

from .base import OCREngine, OCRLine, OCRResult, OCRWord

ENGINES = ("paddle",)


def create_engine(name: str, cfg: dict[str, Any]) -> OCREngine:
    opts = cfg.get("ocr", {}).get(name, {})
    if name == "paddle":
        from .paddle_engine import PaddleEngine

        return PaddleEngine(**opts)
    raise ValueError(f"unknown OCR engine '{name}', choose from {ENGINES}")


__all__ = ["ENGINES", "OCREngine", "OCRLine", "OCRResult", "OCRWord", "create_engine"]
