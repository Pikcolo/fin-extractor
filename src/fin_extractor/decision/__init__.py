"""decision model (shadow mode): ถามคำถาม choice / yes-no กับผลสกัดที่ได้แล้ว บันทึกคำตอบไว้ใน ``ExtractionResult.decisions``

แผนอยู่ที่ docs/draft-11-openthai-systemone-plan.md ตอนนี้เป็นขั้น P2: **บันทึกอย่างเดียว ไม่แตะ ``data`` / ``review`` / ``status``**
จนกว่าจะวัดเทียบกฎเดิมแล้ว (P3) ปิดเป็นค่าตั้งต้น (``decision.enabled: false``) โหลดโมเดลไม่ได้ = ไม่มี ``decisions`` และ pipeline ไม่ล้ม

    client.py     Decider: ส่งคำถามให้ backend (systemone = แพ็กเกจ openthai-systemone บน CPU) สลับลำดับตัวเลือกแล้วเฉลี่ย
    questions.py  สร้าง state + คำถามของแต่ละงานจากผลสกัด (ที่เดียวที่กำหนดว่าป้อนอะไรเข้าโมเดล ไม่ป้อนข้อความทั้งหน้า)
"""

from __future__ import annotations

import logging
from typing import Any

from .client import Decider

log = logging.getLogger(__name__)


def make_decider(cfg: dict[str, Any], force: bool | None = None) -> Decider | None:
    """Decider ตาม ``cfg["decision"]`` · ``force`` = ค่าจาก CLI (``--decision on|off``) ทับ ``enabled`` · ปิด = None (ระบบทำงานเหมือนเดิม)."""
    dc = cfg.get("decision") or {}
    enabled = dc.get("enabled", False) if force is None else force
    return Decider(dc) if enabled else None


__all__ = ["Decider", "make_decider"]
