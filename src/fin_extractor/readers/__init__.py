"""ตัวอ่านภาพ (ขั้น 4) ทุก engine อยู่ที่นี่และใช้แบบเดียวกัน: ``make_reader(engine, cfg).recognize(ภาพ BGR) -> ReadResult``

    openthai  OpenThai 2.0 OCR (torch ในเครื่อง)   ค่าตั้งต้นของ flow หลัก
    typhoon   Typhoon OCR ผ่าน Ollama / vLLM / API
    paddle    PaddleOCR (มีความมั่นใจรายคำ ห้ามอยู่ร่วม process กับ torch: cuDNN คนละรุ่น ดู readers/paddle.py)
    oxr       OXR-1.0 (torch ในเครื่อง อ่านไทยไม่ได้ ดู docs/archive/oxr.md)

import ตอนสร้างตัวอ่าน จึงไม่โหลด torch/paddle ของ engine ที่ไม่ได้เลือก ค่าตั้งของแต่ละตัวอยู่ใน config คีย์ชื่อเดียวกับ engine
"""

from __future__ import annotations

from typing import Any

from .base import ReadResult

ENGINES = ("openthai", "typhoon", "paddle", "oxr")
TORCH_ENGINES = ("openthai", "oxr")  # ตัวที่ใช้ torch ในเครื่อง (typhoon เรียก Ollama, paddle ห้ามอยู่ร่วม process กับ torch)


def make_reader(engine: str, cfg: dict[str, Any]):
    """ตัวอ่านของ engine นั้น (ยังไม่โหลดโมเดล)."""
    if engine == "openthai":
        from .openthai import OpenThaiOCR
        return OpenThaiOCR(cfg["openthai"])
    if engine == "typhoon":
        from .typhoon import TyphoonOCR
        return TyphoonOCR(cfg["typhoon"])
    if engine == "paddle":
        from .paddle import PaddleReader
        return PaddleReader(cfg.get("paddle", {}))
    if engine == "oxr":
        from .oxr import OXRLocal
        return OXRLocal(cfg["oxr"])
    raise ValueError(f"ไม่รู้จัก engine '{engine}' เลือกจาก {ENGINES}")


def reader_label(engine: str, cfg: dict[str, Any]) -> str:
    """ชื่อตัวอ่านที่บันทึกในผล (``<engine>:<โมเดล>``) เท่ากับ ``make_reader(...).name`` โดยไม่ต้องสร้างตัวอ่าน
    openthai/oxr = ชื่อโฟลเดอร์ weights · typhoon = ชื่อโมเดลใน Ollama/API · paddle = ``paddle``."""
    if engine in ("openthai", "oxr"):
        return f"{engine}:" + cfg[engine]["model_dir"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if engine == "typhoon":
        return f"typhoon:{cfg['typhoon']['model']}"
    if engine == "paddle":
        return "paddle"
    raise ValueError(f"ไม่รู้จัก engine '{engine}' เลือกจาก {ENGINES}")


__all__ = ["ENGINES", "TORCH_ENGINES", "ReadResult", "make_reader", "reader_label"]
