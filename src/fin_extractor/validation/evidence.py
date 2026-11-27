"""ระดับหลักฐานรายฟิลด์ (confidence แบบไม่ใช้เฉลย ไม่ใช้น้ำหนักหรือ threshold).

ทุกสัญญาณเป็นจริง/เท็จ/ไม่ทราบ ตัดสินระดับตามกฎตามลำดับ ข้อแรกที่ตรงคือคำตอบ (docs/08 §4.3)

1. ไม่มีค่า                                                          → missing
   (หรือ not_applicable ถ้าชนิดเอกสารนี้ไม่มีฟิลด์นั้น ตามตาราง ``document_types`` ใน config)
2. format ไม่ผ่าน / อยู่ใน check ที่ไม่ผ่าน / ไม่พบในข้อความ / สองแหล่งได้ค่าต่างกัน → conflict
3. มี check ที่ผ่านและยืนยันค่านี้ได้อย่างน้อย 1 ข้อ                       → verified
4. อื่น ๆ                                                            → unverified

ข้อจำกัดที่ตั้งใจไว้:
- ``grounded`` และ ``sources_agree`` ตรวจกับ *ข้อความที่ผู้สกัดเห็น* ทั้ง rules และ LLM อ่านจากข้อความ
  ของตัวอ่านชุดเดียวกัน จึงยืนยันการสกัดได้ ไม่ได้ยืนยันว่าตัวอ่านอ่านภาพถูก
- ``handwritten`` เป็น None เว้นแต่ field อยู่ใน ``handwritten_fields`` (ยังไม่มีตัวตรวจลายมือใน pipeline)
- ``model_conf`` เป็น None เสมอ (confidence ระดับคำของ OCR ยังไม่ถูกส่งต่อมาถึงขั้นสกัด)

ผลที่บันทึก (schema 4.0) คือ ``review_from_evidence``: ระดับ + เหตุผลเป็นคำพูด เรียงตามที่ควรตรวจก่อน หลักฐานละเอียดใช้ภายในเท่านั้น
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date
from decimal import Decimal
from typing import Any

from ..doctypes import DOC_PREFIX, KEY_FIELDS, Profile
from ..extraction.patterns import find_dates, find_numbers, find_percents, find_tax_ids
from ..schema import CORE_AMOUNT_KEYS, ITEM_KEYS, MONEY_KEYS, PARTY_KEYS, FieldEvidence, FieldReview, InvoiceData, ValidationCheck
from .tax_id import is_valid_thai_tax_id

# check ที่ผ่านแล้วไม่ได้ยืนยันว่าอ่านค่าถูก (ตรวจแค่ความสมเหตุสมผล/รูปแบบ) แต่ถ้าไม่ผ่านยังนับว่าขัดแย้ง
# checksum เลขภาษี: ผ่าน = เป็นเลขภาษีที่ใช้ได้ ไม่บอกว่าเป็นของฝ่ายไหน (เอกสารมีเลขของทั้งผู้ซื้อและผู้ขาย
# และผู้ซื้อ/ผู้ขายสลับกันเกิดจริงในชุดสแกนทดสอบ) จึงไม่ยกเป็น verified เลขภาษีจึงไม่มีวันเป็น verified
NOT_CORROBORATING = {"date_range", "vendor_tax_id_checksum", "buyer_tax_id_checksum"}

_SCALARS = ("document_type", "book_no", "invoice_no", "date", *CORE_AMOUNT_KEYS)
_PARTY = PARTY_KEYS
_ITEM = ITEM_KEYS
_AMOUNT_KEYS = set(MONEY_KEYS)
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
        # ค่าหลายบรรทัด (ที่อยู่ที่พิมพ์สองบรรทัด) ต้องพบทุกบรรทัด
        return all(_squash(part) in squashed for part in str(value).split("\n") if part.strip())
    return None  # document_type, branch, items[].description: ค่าถูกแปลง/ประกอบจากหลายบรรทัด ไม่ใช่ข้อความตรง ๆ


def _show(v: Any) -> str:
    return v.isoformat() if isinstance(v, date) else str(v)


def field_grounding(data: InvoiceData, text: str, profile: Profile | None = None) -> dict[str, bool | None]:
    """ค่าแต่ละช่องปรากฏในข้อความหรือไม่ (กฎเดียวกับ ``grounded`` ใน evidence) ไว้ใช้คัดค่าที่ LLM เสนอก่อนรับเข้า."""
    kinds = {DOC_PREFIX + f.key: f.type for f in profile.fields} if profile else {}
    squashed = _squash(text)
    numbers, percents = find_numbers(text), find_percents(text)
    dates, tax_ids = find_dates(text), set(find_tax_ids(text))
    return {key: _is_grounded(key, value, text, squashed, numbers, percents, dates, tax_ids, kinds)
            for key, value in flatten(data).items()}


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


# check → ฟิลด์ที่ check นั้นอ่าน (ตาม checks.py) ใช้บอกว่า check ไหนยังตรวจไม่ได้เพราะขาดช่องไหน
CHECK_FIELDS = {
    "vendor_tax_id_checksum": ["vendor.tax_id"], "buyer_tax_id_checksum": ["buyer.tax_id"],
    "items_sum": ["subtotal", "amount_before_vat", "total_amount"], "discount": ["subtotal", "discount", "amount_before_vat", "total_amount"],
    "before_vat_plus_vat": ["amount_before_vat", "non_taxable_amount", "vat_amount", "total_amount"],
    "vat_rate": ["vat_amount", "amount_before_vat", "total_amount", "vat_rate"], "date_range": ["date"],
}
REVIEW_ORDER = {"conflict": 0, "missing": 1, "unverified": 2, "verified": 3, "not_applicable": 4}


def _reasons(key: str, ev: FieldEvidence, checks: list[ValidationCheck], primary_amount: str | None,
             by_human: bool, source: str | None = None) -> tuple[str, list[str]]:
    """(ระดับ, เหตุผลเป็นคำพูด) ของฟิลด์หนึ่ง ระดับตามหลักฐานเดิม ยกเว้นค่าที่คนแก้: ไม่ต้องปรากฏในข้อความ OCR (คนแก้เพราะ OCR อ่านผิด)."""
    from .checks import NOT_APPLICABLE_DETAIL

    level, out = ev.level, []
    if level == "not_applicable":
        return level, ["เอกสารชนิดนี้ไม่มีฟิลด์นี้"]
    if level == "missing":
        out.append("เอกสารชนิดนี้ต้องมี แต่ระบบหาไม่เจอ ให้กรอกจากภาพ")
        if key == primary_amount:
            out.append("เป็นยอดหลักของเอกสาร: ไม่มีค่านี้สถานะจะเป็น failed")
    if ev.format_valid is False:
        out.append("เลขผู้เสียภาษีไม่ผ่าน checksum: ผิดอย่างน้อยหนึ่งหลัก")
    by_name = {c.name: c for c in checks}
    for name in ev.checks_failed:
        out.append(f"check {name} ไม่ผ่าน ({by_name[name].detail})" if name in by_name else f"check {name} ไม่ผ่าน")
    if ev.grounded is False and not by_human:
        out.append("ค่านี้ไม่ปรากฏในข้อความที่ OCR อ่านได้: ระบบอาจหยิบผิดบรรทัดหรือแปลงผิด")
    if ev.sources_agree is False:
        out.append("rules กับ LLM ได้ค่าไม่ตรงกัน")
    if source not in (None, "rules", "llm", "human") and ev.value not in (None, ""):
        out.append(f"ค่านี้มาจากตัวอ่านเสริม {source} (ตัวอ่านหลักอ่านส่วนนี้ไม่ครบ) ตัวอ่านเสริมอ่านตัวอักษรเพี้ยนได้ ให้ดูเทียบกับภาพ")
    if by_human and level == "conflict" and ev.format_valid is not False and not ev.checks_failed and ev.sources_agree is not False:
        level = "unverified"  # ขัดแย้งแค่เพราะไม่อยู่ในข้อความ OCR ซึ่งไม่ใช่เหตุสำหรับค่าที่คนแก้
    if level == "verified":
        out.append("ยืนยันแล้วด้วย " + ", ".join(ev.checks_passed))
    elif level == "unverified" and ev.value not in (None, ""):
        out.append("แก้โดยคน" if by_human else "ไม่มี check ที่ยืนยันค่านี้ได้ ให้ดูเทียบกับภาพ")
    for c in checks:  # check ที่ใช้ฟิลด์นี้แต่ตรวจไม่ได้เพราะขาดช่องอื่น: บอกว่าเติมช่องไหนแล้วจะตรวจได้
        if c.passed is None and key in CHECK_FIELDS.get(c.name, []) and c.detail != NOT_APPLICABLE_DETAIL and level != "verified":
            out.append(f"check {c.name} ยังตรวจไม่ได้ ({c.detail})")
    return level, out


def review_from_evidence(evidence: dict[str, FieldEvidence], checks: list[ValidationCheck], primary_amount: str | None = None,
                         sources: dict[str, str] | None = None) -> dict[str, FieldReview]:
    """หลักฐานละเอียดรายฟิลด์ → ``review`` ที่บันทึกในผล (ระดับ + เหตุผล + ผู้ให้ค่าถ้าไม่ใช่ rules) เรียงตามที่ควรตรวจก่อน
    ``sources`` = {คีย์: rules | llm | human} ไม่ส่ง = ใช้ ``source`` ในหลักฐาน."""
    sources = sources or {}
    out: dict[str, FieldReview] = {}
    for i, (key, ev) in enumerate(evidence.items()):
        src = sources.get(key, ev.source)
        level, reasons = _reasons(key, ev, checks, primary_amount, src == "human", src)
        out[key] = FieldReview(level=level, reasons=reasons, source=None if src in (None, "rules") else src)
    order = list(out)
    return dict(sorted(out.items(), key=lambda kv: (REVIEW_ORDER[kv[1].level], order.index(kv[0]))))
