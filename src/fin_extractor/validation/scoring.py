"""ตัดสินสถานะเอกสารจากผลที่วัดได้จริงของการ process.

ไม่รวมเป็นคะแนนถ่วงน้ำหนัก เพราะยังไม่มีข้อมูลเฉลยมากพอจะหาน้ำหนักหรือเกณฑ์ตัดได้
ค่าที่รายงานจึงเป็นค่าดิบทั้งหมด: field ที่หาไม่เจอ, ผล check แต่ละข้อ, ocr_confidence จาก engine
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from ..schema import InvoiceData, ValidationCheck

KEY_FIELDS = ("vendor.name", "vendor.tax_id", "invoice_no", "date", "total_amount")

# check ที่ยืนยันค่าของ field ได้จริง — field ที่ไม่อยู่ในนี้ (ชื่อ, เลขที่เอกสาร, วันที่) ไม่มีอะไรยืนยันนอกจากตัวอ่านเอง
# date_range ไม่นับ เพราะตรวจแค่ว่าอยู่ในช่วงที่สมเหตุสมผล ไม่ได้ยืนยันว่าอ่านถูกวัน
CORROBORATING_CHECKS = {
    "vendor.tax_id": ("vendor_tax_id_checksum",),
    "total_amount": ("items_sum", "before_vat_plus_vat", "vat_rate", "discount"),
}

Status = Literal["auto_approved", "needs_review", "failed"]


def _get(d: InvoiceData, dotted: str):
    obj = d
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def not_applicable_for(document_type: str | None, table: dict | None) -> set[str]:
    """field ที่เอกสารชนิดนี้ไม่มี ตามตาราง ``document_types`` ใน config (เทียบชื่อชนิดแบบตัดช่องว่าง/ตัวพิมพ์)

    ตารางตั้งต้นว่าง = ทุกชนิดต้องมีทุก field สำคัญ ต้องกรอกจากรายการที่ฝ่ายบัญชียืนยัน ไม่เดาเอง
    ``document_type`` ที่ LLM ตอบเป็นข้อความอิสระ จึงต้องใส่ชื่อที่ตรงกับที่ระบบสกัดได้จริง
    """
    if not document_type or not table:
        return set()

    def norm(s: str) -> str:
        return "".join(str(s).split()).casefold()

    for name, spec in table.items():
        if norm(name) == norm(document_type):
            return set((spec or {}).get("not_applicable", []))
    return set()


def missing_fields(d: InvoiceData, not_applicable: Collection[str] = ()) -> list[str]:
    return [f for f in KEY_FIELDS if f not in not_applicable and _get(d, f) in (None, "")]


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
    not_applicable: Collection[str] = (),
) -> tuple[list[str], Status]:
    """คืน (field สำคัญที่หาไม่เจอ, สถานะ)

    - failed: ไม่มียอดรวม (ยกเว้นชนิดเอกสารที่ไม่มีฟิลด์นี้)
    - auto_approved: field สำคัญครบ, ไม่มี check ที่ไม่ผ่าน และมี check ที่ยืนยันตัวเลขได้อย่างน้อยหนึ่งข้อ
    - needs_review: กรณีอื่น รวมถึงเมื่อ field ที่อ่านจากลายมือไม่มี check ยืนยัน
    """
    missing = missing_fields(d, not_applicable)
    if d.total_amount is None and "total_amount" not in not_applicable:
        return missing, "failed"
    verified = any(c.passed is True for c in checks)
    if handwritten and uncorroborated(handwritten, checks):
        return missing, "needs_review"
    if not missing and verified and all(c.passed is not False for c in checks):
        return missing, "auto_approved"
    return missing, "needs_review"
