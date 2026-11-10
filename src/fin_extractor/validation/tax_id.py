"""ตรวจ checksum เลขประจำตัวผู้เสียภาษี/เลขบัตรประชาชนไทย 13 หลัก.

หลักที่ 13 = (11 - (Σ d[i] * (13 - i), i=0..11) mod 11) mod 10
ช่วยจับกรณี OCR อ่านเลขผิดหลักเดียว (เช่น 8 ↔ 3, 1 ↔ 7)
"""

from __future__ import annotations


def is_valid_thai_tax_id(tax_id: str | None) -> bool:
    if not tax_id or len(tax_id) != 13 or not tax_id.isdigit():
        return False
    total = sum(int(tax_id[i]) * (13 - i) for i in range(12))
    return (11 - total % 11) % 10 == int(tax_id[12])
