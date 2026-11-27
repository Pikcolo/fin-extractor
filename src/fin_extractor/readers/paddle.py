"""PaddleOCR — detection + recognition แม่นยำสูง, มีโมเดลภาษาไทย (lang='th').

รองรับทั้ง API v3 (``predict``) และ v2 (``ocr``) เพราะสองเวอร์ชันคืนค่าต่างกัน
"""

from __future__ import annotations

import os
import sys
import sysconfig
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..paths import model_dir
from .base import ReadResult
from .lines import OCRResult, OCRWord, group_into_lines, polygon_to_bbox


def use_project_model_cache() -> None:
    """ให้ PaddleX เก็บ/หาโมเดลที่ ``models/paddlex/`` ในโปรเจกต์ แทน ``~/.paddlex``

    ต้องเรียกก่อน import paddleocr/paddlex (อ่านค่านี้ตอน import) ถ้าผู้ใช้ตั้ง ``PADDLE_PDX_CACHE_HOME`` เองไว้แล้วจะไม่ทับ
    """
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(model_dir(name="paddlex")))


def prepare_paddle_env() -> None:
    """เรียกก่อน import paddle/paddlex ทุกครั้ง: ที่เก็บโมเดลในโปรเจกต์ + DLL ของ CUDA บน Windows."""
    use_project_model_cache()
    _expose_nvidia_dlls()


@contextmanager
def without_modelscope():
    """ระหว่าง import paddlex/paddleocr: ใส่ ``modelscope`` ตัวหลอก ไม่ให้ torch ถูกโหลดพ่วง

    paddlex import ``modelscope`` (ใช้แค่ดาวน์โหลดโมเดล ซึ่งเราเก็บไว้ในโปรเจกต์แล้ว) และ modelscope import torch ถ้าติดตั้งอยู่
    torch (cu128) มากับ cuDNN 9.19 ส่วน paddlepaddle-gpu ล็อก cuDNN 9.15 ชื่อ DLL เดียวกัน อยู่ใน process เดียวกันไม่ได้
    (ตัวไหนโหลดก่อน อีกตัวล้มด้วย WinError 127 ตัวตรวจทิศเคยล้มทุกใบ) process ที่ใช้ Paddle จึงโหลด torch ไม่ได้อีก
    (คำสั่ง sections รัน preprocess แยก process และไม่ import torch เมื่ออ่านด้วย paddle/typhoon)
    """
    stub = None
    if "modelscope" not in sys.modules:
        stub = sys.modules["modelscope"] = types.ModuleType("modelscope")
    try:
        yield
    finally:
        if stub is not None and sys.modules.get("modelscope") is stub:
            del sys.modules["modelscope"]


def _expose_nvidia_dlls() -> None:
    """Windows: ให้ Paddle หา DLL ของ CUDA ที่มากับแพ็กเกจ ``nvidia-*`` เจอ

    ``paddlepaddle-gpu`` ดึง cuBLAS / cuDNN มาเป็นแพ็กเกจ pip ซึ่งวาง DLL ไว้ใต้
    ``site-packages/nvidia/.../bin`` แต่ Paddle ไม่ได้มองหาในโฟลเดอร์นั้น ถ้าไม่เพิ่มให้
    จะล้มตอนรันด้วย ``Could not locate cublasLt64_13.dll`` (เจอจริงกับ paddlepaddle-gpu
    3.4.0 + CUDA 13.0) ต้องเรียกก่อน import paddle
    """
    if os.name != "nt":
        return
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not nvidia.is_dir():
        return
    dirs = sorted({str(dll.parent) for dll in nvidia.rglob("*.dll")})
    for d in dirs:
        os.add_dll_directory(d)
    # Paddle โหลด DLL ด้วยชื่อไฟล์ผ่าน PATH ด้วย การเพิ่ม add_dll_directory อย่างเดียวไม่พอ
    os.environ["PATH"] = os.pathsep.join([*dirs, os.environ.get("PATH", "")])


class PaddleEngine:
    name = "paddle"

    def __init__(self, lang: str = "th", use_angle_cls: bool = True):
        prepare_paddle_env()
        with without_modelscope():
            import paddleocr
            from paddleocr import PaddleOCR

        self._v3 = int(paddleocr.__version__.split(".")[0]) >= 3
        if self._v3:
            # orientation/unwarp ทำใน preprocessing แล้ว จึงปิดของ Paddle ไว้
            self.ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_angle_cls,
            )
        else:
            self.ocr = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, show_log=False)

    def recognize(self, img: np.ndarray) -> OCRResult:
        if img.ndim == 2:  # Paddle ต้องการภาพ 3 channel
            img = np.stack([img] * 3, axis=-1)
        words: list[OCRWord] = []
        if self._v3:
            for res in self.ocr.predict(img):
                for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                    if text.strip():
                        words.append(OCRWord(text, float(score), polygon_to_bbox(poly)))
        else:
            for page in self.ocr.ocr(img, cls=True) or []:
                for poly, (text, score) in page or []:
                    if text.strip():
                        words.append(OCRWord(text, float(score), polygon_to_bbox(poly)))
        return OCRResult(self.name, group_into_lines(words))


class PaddleReader:
    """PaddleOCR ในรูปเดียวกับตัวอ่านอื่น: ข้อความของภาพเป็นบรรทัดล้วน (ไม่มีตาราง Markdown) + ความมั่นใจเฉลี่ย
    โหลดโมเดลตอนอ่านภาพแรก (``_load``) ไม่มีเพดาน token จึง ``finish_reason`` เป็น None เสมอ."""

    name = "paddle"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.engine: PaddleEngine | None = None

    def _load(self) -> PaddleEngine:
        if self.engine is None:
            self.engine = PaddleEngine(**self.cfg)
        return self.engine

    def recognize(self, img: np.ndarray) -> ReadResult:
        res = self._load().recognize(img)
        return ReadResult(self.name, res.text, confidence=res.mean_confidence)
