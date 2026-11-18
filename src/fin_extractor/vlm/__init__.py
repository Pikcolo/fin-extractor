from __future__ import annotations

from typing import Any

from .typhoon_ocr import TyphoonOCR, VLMResult

# ชื่อ engine ที่เป็น VLM (อ่านทั้งหน้าเป็น Markdown) → ชื่อโฟลเดอร์ใน 04_ocr/ และ 05_extract/
VLM_ENGINES = {"vlm": "typhoon", "openthai": "openthai"}


def create_reader(engine: str, cfg: dict[str, Any]):
    """import แบบ lazy: openthai ดึง torch/transformers ซึ่งหนัก."""
    if engine == "openthai":
        from .openthai_ocr import OpenThaiOCR

        return OpenThaiOCR(cfg["openthai"])
    return TyphoonOCR(cfg["vlm"])


def vlm_model(engine: str, cfg: dict[str, Any]) -> str:
    return cfg["openthai"]["model_dir"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] if engine == "openthai" else cfg["vlm"]["model"]


__all__ = ["TyphoonOCR", "VLMResult", "VLM_ENGINES", "create_reader", "vlm_model"]
