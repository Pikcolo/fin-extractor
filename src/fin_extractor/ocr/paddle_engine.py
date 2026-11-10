"""PaddleOCR — detection + recognition แม่นยำสูง, มีโมเดลภาษาไทย (lang='th').

รองรับทั้ง API v3 (``predict``) และ v2 (``ocr``) เพราะสองเวอร์ชันคืนค่าต่างกัน
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path

import numpy as np

from .base import OCREngine, OCRResult, OCRWord, group_into_lines, polygon_to_bbox


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


class PaddleEngine(OCREngine):
    name = "paddle"

    def __init__(self, lang: str = "th", use_angle_cls: bool = True):
        _expose_nvidia_dlls()
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
