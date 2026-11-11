"""ระดับหลักฐานรายฟิลด์ (confidence แบบไม่ใช้เฉลย ไม่ใช้น้ำหนักหรือ threshold).

ทุกสัญญาณเป็นจริง/เท็จ/ไม่ทราบ ตัดสินระดับตามกฎตามลำดับ ข้อแรกที่ตรงคือคำตอบ (docs/08 §4.3)

1. ไม่มีค่า                                                          → missing
   (หรือ not_applicable ถ้าชนิดเอกสารนี้ไม่มีฟิลด์นั้น ตามตาราง ``document_types`` ใน config)
2. format ไม่ผ่าน / อยู่ใน check ที่ไม่ผ่าน / ไม่พบในข้อความ / สองแหล่งได้ค่าต่างกัน → conflict
3. มี check ที่ผ่านและยืนยันค่านี้ได้อย่างน้อย 1 ข้อ                       → verified
4. อื่น ๆ                                                            → unverified

ข้อจำกัดที่ตั้งใจไว้:
- ``grounded`` และ ``sources_agree`` ตรวจกับ *ข้อความที่ผู้สกัดเห็น* ในโหมด vlm ทั้ง rules และ LLM อ่านจาก Markdown
  ของ Typhoon ชุดเดียวกัน จึงยืนยันการสกัดได้ ไม่ได้ยืนยันว่า VLM อ่านภาพถูก
- ``handwritten`` เป็น None เว้นแต่ field อยู่ใน ``handwritten_fields`` (ยังไม่มีตัวตรวจลายมือใน pipeline)
- ``model_conf`` เป็น None เสมอ (confidence ระดับคำของ OCR ยังไม่ถูกส่งต่อมาถึงขั้นสกัด)
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date
from decimal import Decimal
from typing import Any

from ..doctypes import DOC_PREFIX, KEY_FIELDS, Profile
from ..extraction.patterns import find_dates, find_numbers, find_percents, find_tax_ids
from ..schema import FieldEvidence, InvoiceData, ValidationCheck
from .tax_id import is_valid_thai_tax_id

# check ที่ผ่านแล้วไม่ได้ยืนยันว่าอ่านค่าถูก (ตรวจแค่ความสมเหตุสมผล) แต่ถ้าไม่ผ่านยังนับว่าขัดแย้ง
NOT_CORROBORATING = {"date_range"}

_SCALARS = ("document_type", "book_no", "invoice_no", "date",
            "subtotal", "discount", "amount_before_vat", "vat_rate", "vat_amount", "total_amount")
_PARTY = ("name", "tax_id", "branch", "address")
_ITEM = ("description", "quantity", "unit_price", "amount")
_AMOUNT_KEYS = {"subtotal", "discount", "amount_before_vat", "vat_amount", "total_amount"}
_TEXT_KEYS = {"vendor.name", "vendor.address", "buyer.name", "buyer.address", "invoice_no", "book_no"}


def flatten(d: InvoiceData) -> dict[str, Any]:
    """ค่าที่ไม่ว่างของทุก field เป็นคีย์แบบ ``vendor.tax_id`` / ``items[0].amount`` (ไม่รวม currency ที่เป็นค่าตั้งต้น)."""
    out: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        if value not in (None, ""):
            out[key] = value

    for k in _SCALARS:
        put(k, getattr(d, k))
    for party in ("vendor", "buyer"):
        for k in _PARTY:
            put(f"{party}.{k}", getattr(getattr(d, party), k))
    for i, it in enumerate(d.items):
        for k in _ITEM:
            put(f"items[{i}].{k}", getattr(it, k))
    for key, value in d.document_fields.items():
        put(f"{DOC_PREFIX}{key}", value)  # ชนิดของค่าอยู่ที่โปรไฟล์ (FieldSpec.type)
    for i, extra in enumerate(d.remark):
        put(f"remark[{i}]", extra.value)  # ป้ายอยู่ที่ data.remark[i].label
    return out


def comparable(v: Any) -> Any:
    """รูปที่ใช้เทียบว่าสองแหล่งได้ค่าเดียวกัน — ข้อความตัดช่องว่าง ที่เหลือเทียบตรง ๆ (ไม่มีเกณฑ์ความคล้าย)."""
    return "".join(v.split()) if isinstance(v, str) else v


def _squash(text: str) -> str:
    return "".join(text.replace("|", " ").split())


def _doc_field_grounded(kind: str, value: str, squashed: str,
                        numbers: set[Decimal], percents: set[Decimal], dates: set[date], tax_ids: set[str]) -> bool | None:
    """ฟิลด์ประจำชนิด: ตรวจตามชนิดที่โปรไฟล์ประกาศ (ค่าเก็บเป็นสตริงมาตรฐานแล้ว)."""
    try:
        if kind == "money":
            return Decimal(value) in numbers
        if kind == "percent":
            return Decimal(value) * 100 in percents
        if kind == "date":
            return date.fromisoformat(value) in dates
    except (ValueError, ArithmeticError):
        return False
    if kind == "tax_id":
        return value in tax_ids
    return _squash(value) in squashed  # text / code


def _is_grounded(key: str, value: Any, text: str, squashed: str,
                 numbers: set[Decimal], percents: set[Decimal], dates: set[date], tax_ids: set[str],
                 kinds: dict[str, str] | None = None) -> bool | None:
    if key.startswith(DOC_PREFIX):
        return _doc_field_grounded((kinds or {}).get(key, "text"), value, squashed, numbers, percents, dates, tax_ids)
    leaf = key.rsplit(".", 1)[-1]
    if key in _AMOUNT_KEYS or key.startswith("items[") and leaf in ("amount", "unit_price", "quantity"):
        return value in numbers
    if key == "vat_rate":
        return value * 100 in percents
    if leaf == "tax_id":
        return value in tax_ids
    if key == "date":
        return value in dates
    if key in _TEXT_KEYS or key.startswith("remark["):
        return _squash(value) in squashed
    return None  # document_type, branch, items[].description: ค่าถูกแปลง/ประกอบจากหลายบรรทัด ไม่ใช่ข้อความตรง ๆ


def _show(v: Any) -> str:
    return v.isoformat() if isinstance(v, date) else str(v)


def build_field_evidence(
    data: InvoiceData,
    checks: list[ValidationCheck],
    text: str,
    sources: dict[str, str] | None = None,
    agree: dict[str, bool | None] | None = None,
    handwritten: list[str] | None = None,
    not_applicable: Collection[str] = (),
    required: Collection[str] = KEY_FIELDS,
    profile: Profile | None = None,
) -> dict[str, FieldEvidence]:
    """``required`` และ ``profile`` มาจากโปรไฟล์ชนิดเอกสาร (ไม่ส่ง = กฎเดิมของใบกำกับภาษี)."""
    sources, agree, handwritten = sources or {}, agree or {}, handwritten or []
    kinds = {DOC_PREFIX + f.key: f.type for f in profile.fields} if profile else {}
    passed: dict[str, list[str]] = {}
    failed: dict[str, list[str]] = {}
    for c in checks:
        for f in c.fields:
            if c.passed is False:
                failed.setdefault(f, []).append(c.name)
            elif c.passed is True and c.name not in NOT_CORROBORATING:
                passed.setdefault(f, []).append(c.name)

    squashed = _squash(text)
    numbers, percents = find_numbers(text), find_percents(text)
    dates, tax_ids = find_dates(text), set(find_tax_ids(text))

    out: dict[str, FieldEvidence] = {}
    flat = flatten(data)
    for key, value in flat.items():
        fmt = is_valid_thai_tax_id(value) if key.endswith("tax_id") or kinds.get(key) == "tax_id" else None
        grounded = _is_grounded(key, value, text, squashed, numbers, percents, dates, tax_ids, kinds)
        ev = FieldEvidence(
            level="unverified", value=_show(value), format_valid=fmt,
            checks_passed=passed.get(key, []), checks_failed=failed.get(key, []),
            grounded=grounded, sources_agree=agree.get(key),
            handwritten=True if key in handwritten else None, source=sources.get(key),
        )
        if fmt is False or ev.checks_failed or grounded is False or ev.sources_agree is False:
            ev.level = "conflict"
        elif ev.checks_passed:
            ev.level = "verified"
        out[key] = ev
    for key in required:
        if key not in flat:
            out[key] = FieldEvidence(level="not_applicable" if key in not_applicable else "missing")
    return out
