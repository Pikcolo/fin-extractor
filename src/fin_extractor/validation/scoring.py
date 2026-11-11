"""ตัดสินสถานะเอกสารจากผลที่วัดได้จริงของการ process.

ไม่รวมเป็นคะแนนถ่วงน้ำหนัก เพราะยังไม่มีข้อมูลเฉลยมากพอจะหาน้ำหนักหรือเกณฑ์ตัดได้
ค่าที่รายงานจึงเป็นค่าดิบทั้งหมด: field ที่หาไม่เจอ, ผล check แต่ละข้อ, ocr_confidence จาก engine
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from ..doctypes import DOC_PREFIX, KEY_FIELDS, legacy_not_applicable
from ..schema import InvoiceData, ValidationCheck


# check ที่ยืนยันค่าของ field ได้จริง — field ที่ไม่อยู่ในนี้ (ชื่อ, เลขที่เอกสาร, วันที่) ไม่มีอะไรยืนยันนอกจากตัวอ่านเอง
# date_range ไม่นับ เพราะตรวจแค่ว่าอยู่ในช่วงที่สมเหตุสมผล ไม่ได้ยืนยันว่าอ่านถูกวัน
CORROBORATING_CHECKS = {
    "vendor.tax_id": ("vendor_tax_id_checksum",),
    "total_amount": ("items_sum", "before_vat_plus_vat", "vat_rate", "discount"),
}

Status = Literal["auto_approved", "needs_review", "failed"]


def _get(d: InvoiceData, dotted: str):
    """ค่าของ field แบบ ``vendor.name`` หรือ ``document_fields.total_paid`` (ฟิลด์ประจำชนิดอยู่ใน dict)."""
    if dotted.startswith(DOC_PREFIX):
        return d.document_fields.get(dotted[len(DOC_PREFIX):])
    obj = d
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def not_applicable_for(document_type: str | None, table: dict | None) -> set[str]:
    """field ที่เอกสารชนิดนี้ไม่มี ตามตาราง ``document_types`` ใน config (เทียบชื่อชนิดแบบตัดช่องว่าง/ตัวพิมพ์)

    รูปแบบเก่า (เทียบกับ ``data.document_type`` ตรง ๆ) โปรไฟล์ใหม่ใช้ ``Profile.not_applicable`` ผ่าน ``doctypes.classify``
    ``document_type`` ที่ LLM ตอบเป็นข้อความอิสระ จึงต้องใส่ชื่อที่ตรงกับที่ระบบสกัดได้จริง
    """
    return legacy_not_applicable(table, document_type)


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
    primary_amount: str | None = "total_amount", can_auto_approve: bool = True,
) -> tuple[list[str], Status]:
    """คืน (field ที่ชนิดเอกสารนี้ต้องมีแต่หาไม่เจอ, สถานะ)

    ``required`` และ ``primary_amount`` มาจากโปรไฟล์ชนิดเอกสาร (ค่าตั้งต้น = กฎเดิมของใบกำกับภาษี)
    - failed: ไม่มียอดหลัก ``primary_amount`` (ยกเว้นชนิดที่ไม่มีฟิลด์นี้ หรือโปรไฟล์ตั้งเป็น null)
    - auto_approved: field ที่ต้องมีครบ, ไม่มี check ที่ไม่ผ่าน และมี check ที่ยืนยันตัวเลขได้อย่างน้อยหนึ่งข้อ
    - needs_review: กรณีอื่น รวมถึงเมื่อ field ที่อ่านจากลายมือไม่มี check ยืนยัน และเมื่อโปรไฟล์ยังเป็นฉบับร่าง
      (``can_auto_approve=False``: รายการฟิลด์ที่ต้องมียังไม่ผ่านการยืนยัน จึงไม่ปล่อยผ่านโดยไม่มีคนตรวจ)
    """
    missing = missing_fields(d, not_applicable, required)
    if primary_amount and primary_amount not in not_applicable and _get(d, primary_amount) in (None, ""):
        return missing, "failed"
    verified = any(c.passed is True for c in checks)
    if handwritten and uncorroborated(handwritten, checks):
        return missing, "needs_review"
    if can_auto_approve and not missing and verified and all(c.passed is not False for c in checks):
        return missing, "auto_approved"
    return missing, "needs_review"
