"""ตัวอ่านภาพ (ขั้น 4) ทุก engine อยู่ที่นี่และใช้แบบเดียวกัน: ``make_reader(engine, cfg).recognize(ภาพ BGR) -> ReadResult``

    openthai       OpenThai 2.0 OCR Markdown (torch ในเครื่อง)   ค่าตั้งต้นของ flow หลัก
    openthai21     OpenThai 2.1 OCR Markdown — ตัวอ่านเดียวกับ openthai แค่ weights ใหม่ (คีย์ config ``openthai21``) ผลแยกโฟลเดอร์ ไม่ทับของ 2.0
    openthai-text  OpenThai 2.1 OCR Text (ONNX บน CPU คืนบรรทัด + ความมั่นใจ ไม่มีตาราง)
    typhoon        Typhoon OCR ผ่าน Ollama / vLLM / API
    paddle         PaddleOCR (มีความมั่นใจรายคำ ห้ามอยู่ร่วม process กับ torch: cuDNN คนละรุ่น ดู readers/paddle.py)

import ตอนสร้างตัวอ่าน จึงไม่โหลด torch/paddle ของ engine ที่ไม่ได้เลือก ค่าตั้งของแต่ละตัวอยู่ใน config คีย์ชื่อเดียวกับ engine
"""

from __future__ import annotations

from typing import Any

from .base import ReadResult

ENGINES = ("openthai", "openthai21", "openthai-text", "typhoon", "paddle")
TORCH_ENGINES = ("openthai", "openthai21")  # ตัวที่ใช้ torch ในเครื่อง (typhoon เรียก Ollama, paddle ห้ามอยู่ร่วม process กับ torch, openthai-text ใช้ onnxruntime)


def make_reader(engine: str, cfg: dict[str, Any]):
    """ตัวอ่านของ engine นั้น (ยังไม่โหลดโมเดล)."""
    if engine in ("openthai", "openthai21"):
        from .openthai import OpenThaiOCR
        return OpenThaiOCR(cfg[engine])
    if engine == "openthai-text":
        from .openthai_text import OpenThaiTextOCR
        return OpenThaiTextOCR(cfg["openthai-text"])
    if engine == "typhoon":
        from .typhoon import TyphoonOCR
        return TyphoonOCR(cfg["typhoon"])
    if engine == "paddle":
        from .paddle import PaddleReader
        return PaddleReader(cfg.get("paddle", {}))
    raise ValueError(f"ไม่รู้จัก engine '{engine}' เลือกจาก {ENGINES}")


def reader_label(engine: str, cfg: dict[str, Any]) -> str:
    """ชื่อตัวอ่านที่บันทึกในผล (``<ชนิด>:<โมเดล>``) เท่ากับ ``make_reader(...).name`` โดยไม่ต้องสร้างตัวอ่าน
    openthai/openthai21/openthai-text = ชื่อโฟลเดอร์ weights (openthai21 ใช้ตัวอ่าน openthai จึงขึ้นต้น ``openthai:``)
    typhoon = ชื่อโมเดลใน Ollama/API · paddle = ``paddle``."""
    if engine in ("openthai", "openthai21", "openthai-text"):
        kind = "openthai" if engine == "openthai21" else engine
        return f"{kind}:" + cfg[engine]["model_dir"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if engine == "typhoon":
        return f"typhoon:{cfg['typhoon']['model']}"
    if engine == "paddle":
        return "paddle"
    raise ValueError(f"ไม่รู้จัก engine '{engine}' เลือกจาก {ENGINES}")


__all__ = ["ENGINES", "TORCH_ENGINES", "ReadResult", "make_reader", "reader_label"]
