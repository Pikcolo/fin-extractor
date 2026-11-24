"""OpenThai2.1 OCR Text (iApp) — ตัวอ่านแบบบรรทัด (PP-OCRv6-tiny: ตัวตรวจบรรทัด + ตัวอ่าน CTC ที่ฝึกไทย) ONNX ขนาด ~9 MB รันบน **CPU** เท่านั้น

ต่างจาก OpenThai Markdown (VLM บน GPU): คืนข้อความทีละบรรทัดตามลำดับอ่าน พร้อม **ความมั่นใจต่อบรรทัด** ไม่มีตาราง Markdown/HTML และไม่มีเพดาน token
จึงไม่หลอนข้อความเพิ่มแบบ VLM แต่ model card ระบุว่าอ่านลายมือได้แย่ (handwriting 10.3) และตรา/ตรายางออกมาเป็นข้อความ

โค้ดอ่านผล (``runtime/line_engine.py``) อยู่ในโฟลเดอร์โมเดลเอง: ``hf download iapp/OpenThai2.1-OCR-Text-20261118 --local-dir <model_dir>``
ต้องมี ``onnxruntime`` (หรือ openvino) + ``pyclipper`` (ดู requirements.txt ในโฟลเดอร์โมเดล)
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ..config import resolve_path
from .base import ReadResult


class OpenThaiTextOCR:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model_dir = resolve_path(cfg["model_dir"])
        self._ocr = None

    @property
    def name(self) -> str:
        return f"openthai-text:{self.model_dir.name}"

    def _load(self):
        if self._ocr is not None:
            return self._ocr
        if not (self.model_dir / "runtime" / "line_engine.py").is_file():
            raise FileNotFoundError(f"ไม่พบโมเดลที่ {self.model_dir} — hf download iapp/OpenThai2.1-OCR-Text-20261118 --local-dir <model_dir>")
        sys.path.insert(0, str(self.model_dir))  # ใช้ runtime/ ที่มากับโมเดลโดยไม่ต้อง pip install
        from runtime.line_engine import LineOCR

        self._ocr = LineOCR(str(self.model_dir), threads=int(self.cfg.get("threads", 1)))
        return self._ocr

    def recognize(self, img: np.ndarray) -> ReadResult:
        """ภาพ BGR → บรรทัดข้อความ (คั่นด้วยขึ้นบรรทัดใหม่) + ค่าเฉลี่ยความมั่นใจของบรรทัด (None ถ้าไม่เจอบรรทัด)."""
        r = self._load().read(img)
        confs = [float(ln["confidence"]) for ln in r.get("lines", []) if ln.get("confidence") is not None]
        return ReadResult(self.name, r.get("text", ""), None, sum(confs) / len(confs) if confs else None)
