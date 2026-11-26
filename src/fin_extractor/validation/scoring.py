"""ตัดสินสถานะเอกสารจากผลที่วัดได้จริงของการ process.

ไม่รวมเป็นคะแนนถ่วงน้ำหนัก เพราะยังไม่มีข้อมูลเฉลยมากพอจะหาน้ำหนักหรือเกณฑ์ตัดได้
ค่าที่รายงานจึงเป็นค่าดิบทั้งหมด: field ที่หาไม่เจอ, ผล check แต่ละข้อ, ocr_confidence จาก engine
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from ..doctypes import DOC_PREFIX, KEY_FIELDS
from ..schema import InvoiceData, ValidationCheck


# check ที่ยืนยันค่าของ field ได้จริง — field ที่ไม่อยู่ในนี้ (ชื่อ, เลขที่เอกสาร, วันที่) ไม่มีอะไรยืนยันนอกจากตัวอ่านเอง
# date_range ไม่นับ เพราะตรวจแค่ว่าอยู่ในช่วงที่สมเหตุสมผล ไม่ได้ยืนยันว่าอ่านถูกวัน
CORROBORATING_CHECKS = {
    "vendor.tax_id": ("vendor_tax_id_checksum",),
    "total_amount": ("items_sum", "before_vat_plus_vat", "vat_rate", "discount"),
}

Status = Literal["ready", "check", "incomplete"]


def _get(d: InvoiceData, dotted: str):
    """ค่าของ field แบบ ``vendor.name`` หรือ ``document_fields.total_paid`` (ฟิลด์ประจำชนิดอยู่ใน dict)."""
    if dotted.startswith(DOC_PREFIX):
        return d.document_fields.get(dotted[len(DOC_PREFIX):])
    obj = d
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def missing_fields(
    d: InvoiceData, not_applicable: Collection[str] = (), required: Collection[str] = KEY_FIELDS
) -> list[str]:
    return [f for f in required if f not in not_applicable and _get(d, f) in (None, "")]


def uncorroborated(handwritten: list[str], checks: list[ValidationCheck]) -> list[str]:
    """field ลายมือที่ไม่มี check ไหนยืนยัน (ไม่มี check ที่ผ่าน หรือมี check ที่เกี่ยวข้องไม่ผ่าน)."""
    by_name = {c.name: c.passed for c in checks}
    out = []
    for f in handwritten:
        results = [by_name[n] for n in CORROBORATING_CHECKS.get(f, ()) if n in by_name]
        if True not in results or False in results:
            out.append(f)
    return out


def decide_status(
    d: InvoiceData, checks: list[ValidationCheck], handwritten: list[str] | None = None,
    not_applicable: Collection[str] = (), required: Collection[str] = KEY_FIELDS,
    primary_amount: str | None = "total_amount",
) -> tuple[list[str], Status]:
    """คืน (field ที่ชนิดเอกสารนี้ต้องมีแต่หาไม่เจอ, สถานะ)

    ``required`` และ ``primary_amount`` มาจากโปรไฟล์ชนิดเอกสาร (ค่าตั้งต้น = กฎเดิมของใบกำกับภาษี)
    สถานะ = ลำดับความเร่งในการตรวจ (เจ้าหน้าที่ตรวจทุกใบอยู่แล้ว ไม่มีสถานะไหนข้ามการตรวจ)
    - incomplete: ไม่มียอดหลัก ``primary_amount`` ต้องกรอกเอง (ยกเว้นชนิดที่ไม่มีฟิลด์นี้ หรือโปรไฟล์ตั้งเป็น null)
    - ready: field ที่ต้องมีครบ, ไม่มี check ที่ไม่ผ่าน และมี check ที่ยืนยันตัวเลขได้อย่างน้อยหนึ่งข้อ (ไม่นับ date_range / checksum เลขภาษี) → ดูเทียบภาพผ่าน ๆ
    - check: กรณีอื่น (มีช่องขัดแย้ง/หาไม่เจอ หรือ field ที่อ่านจากลายมือไม่มี check ยืนยัน) → ดูช่องที่ไฮไลต์
    """
    missing = missing_fields(d, not_applicable, required)
    if primary_amount and primary_amount not in not_applicable and _get(d, primary_amount) in (None, ""):
        return missing, "incomplete"
    from .evidence import NOT_CORROBORATING  # กฎเดียวกับระดับรายฟิลด์: date_range / checksum เลขภาษีผ่านไม่นับเป็นตัวยืนยัน

    verified = any(c.passed is True and c.name not in NOT_CORROBORATING for c in checks)
    if handwritten and uncorroborated(handwritten, checks):
        return missing, "check"
    if not missing and verified and all(c.passed is not False for c in checks):
        return missing, "ready"
    return missing, "check"
