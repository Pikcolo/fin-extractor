"""ผลลัพธ์ร่วมของตัวอ่านทุกตัว (``readers/``): ข้อความของภาพหนึ่งชิ้น + เหตุที่หยุด + ความมั่นใจ (ถ้า engine รายงาน)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReadResult:
    engine: str
    text: str  # Markdown (ตารางเป็น HTML) หรือบรรทัดข้อความล้วน แล้วแต่ engine — ขั้นสกัดแปลงเป็นบรรทัดด้วยวิธีเดียวกัน
    finish_reason: str | None = None  # "length" = ชนเพดาน token (ข้อความอาจขาด) · None = engine ไม่มีเพดาน (Paddle)
    confidence: float | None = None  # ค่าเฉลี่ยที่ engine รายงานเอง (Paddle) ตัวอื่นไม่มี
    # กรอบบล็อกที่ engine บอกว่าอ่านจากตรงไหน [{"class", "box": [x, y, w, h] px บนภาพที่ส่งเข้า, "text"}] (OpenThai) ตัวอื่น None
    blocks: list[dict] | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"
