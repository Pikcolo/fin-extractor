"""Interface กลางของ OCR engine — ทุก engine คืน ``OCRResult`` รูปแบบเดียวกัน."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

BBox = tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class OCRWord:
    text: str
    confidence: float  # 0..1
    bbox: BBox


def _is_thai(ch: str) -> bool:
    return "฀" <= ch <= "๿"


def normalize_thai(text: str) -> str:
    # OCR มักอ่าน "ำ" (U+0E33) เป็น นิคหิต + า (U+0E4D U+0E32)
    return text.replace("ํา", "ำ")


@dataclass
class OCRLine:
    words: list[OCRWord]
    thai_space_ratio: float = 0.5   # ไทย-ไทย: เว้นวรรคเฉพาะช่องว่างกว้าง
    latin_space_ratio: float = 0.1  # ภาษาอังกฤษ/ตัวเลข: ช่องว่างแคบก็นับเป็นวรรค

    @property
    def text(self) -> str:
        """ต่อคำโดยเว้นวรรคเฉพาะเมื่อมีช่องว่างจริงในภาพ.

        ภาษาไทยไม่เว้นวรรคระหว่างคำ แต่ engine (เช่น Tesseract) มักตัดเป็นคำ/ตัวอักษรย่อย ๆ
        ถ้าต่อด้วย " " ตรง ๆ จะได้ "บ ร ิ ษั ท" แทน "บริษัท"
        """
        if not self.words:
            return ""
        height = max(w.bbox[3] - w.bbox[1] for w in self.words) or 1
        parts = [self.words[0].text]
        for prev, cur in zip(self.words, self.words[1:]):
            both_thai = _is_thai(prev.text[-1]) and _is_thai(cur.text[0])
            ratio = self.thai_space_ratio if both_thai else self.latin_space_ratio
            gap = cur.bbox[0] - prev.bbox[2]
            parts.append((" " if gap > height * ratio else "") + cur.text)
        return normalize_thai("".join(parts))

    @property
    def bbox(self) -> BBox:
        return (
            min(w.bbox[0] for w in self.words), min(w.bbox[1] for w in self.words),
            max(w.bbox[2] for w in self.words), max(w.bbox[3] for w in self.words),
        )


@dataclass
class OCRResult:
    engine: str
    lines: list[OCRLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float | None:
        confs = [w.confidence for line in self.lines for w in line.words]
        return float(np.mean(confs)) if confs else None


class OCREngine(ABC):
    name: str = "base"

    @abstractmethod
    def recognize(self, img: np.ndarray) -> OCRResult: ...


def group_into_lines(words: list[OCRWord], y_overlap: float = 0.5) -> list[OCRLine]:
    """จัดกลุ่มกล่องคำเป็นบรรทัด (สำหรับ engine ที่คืนเป็นกล่องลอย ๆ เช่น EasyOCR/Paddle).

    คำสองคำอยู่บรรทัดเดียวกันถ้าความสูงซ้อนทับกัน >= ``y_overlap`` ของกล่องที่เตี้ยกว่า
    """
    lines: list[list[OCRWord]] = []
    for w in sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2):
        for line in lines:
            ref = line[-1].bbox
            overlap = min(ref[3], w.bbox[3]) - max(ref[1], w.bbox[1])
            min_h = min(ref[3] - ref[1], w.bbox[3] - w.bbox[1]) or 1
            if overlap / min_h >= y_overlap:
                line.append(w)
                break
        else:
            lines.append([w])
    return [OCRLine(sorted(line, key=lambda w: w.bbox[0])) for line in lines]


def polygon_to_bbox(points) -> BBox:
    pts = np.asarray(points).reshape(-1, 2)
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())
