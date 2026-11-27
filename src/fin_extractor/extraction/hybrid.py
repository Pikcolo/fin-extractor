"""สกัดแบบ rules + LLM ร่วมกัน: rules นำ, LLM ถามเฉพาะช่องที่ยังไม่เสร็จ, ค่าจาก LLM ต้องผ่านด่านก่อนรับ.

ต่างจาก ``winner`` (เลือกผลทั้งชุดที่ check ดีกว่า แล้วเติมช่องว่าง) ตรงที่
- ช่องที่ rules ได้แล้วและ check ไม่ติดใจ → ใช้ค่า rules ไม่ให้ LLM แตะ
- ช่องที่ยังไม่เสร็จ = field ที่ชนิดเอกสารต้องมีแต่หาไม่เจอ + field ที่อยู่ใน check ที่ไม่ผ่าน (ไม่มีเกณฑ์ตัวเลขที่ตั้งเอง)
- ค่าจาก LLM รับเมื่อ (1) ปรากฏในข้อความ (2) เลขภาษีผ่าน checksum (3) ยอดเงิน/รายการแล้ว check รวมไม่แย่ลงกว่าเดิม
  ไม่ผ่านด่านเท่ากับไม่ได้ค่า (ปล่อยเป็นว่าง ไม่เดา)

remark ใช้ของ rules เท่านั้น (LLM ไม่ถูกขอ remark ในโหมดนี้)
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import date

from ..doctypes import DOC_PREFIX, Profile
from ..schema import CORE_AMOUNT_KEYS, InvoiceData, ValidationCheck
from ..validation.evidence import comparable, field_grounding, flatten
from ..validation.scoring import _get, missing_fields
from ..validation.tax_id import is_valid_thai_tax_id

AMOUNT_KEYS = set(CORE_AMOUNT_KEYS)
SKIP_PREFIXES = ("remark[",)


def top_level(key: str) -> str:
    """คีย์แบบ flatten → ชื่อ property ระดับบนของ schema: vendor.name → vendor, items[0].amount → items."""
    if key.startswith(DOC_PREFIX):
        return "document_fields"
    return key.split("[", 1)[0].split(".", 1)[0]


def unresolved_keys(
    data: InvoiceData, checks: list[ValidationCheck], required: Collection[str],
    not_applicable: Collection[str], primary_amount: str | None,
) -> list[str]:
    """ช่องที่ผล rules ยังไม่เสร็จ: ที่ต้องมีแต่ว่าง + ที่อยู่ใน check ที่ไม่ผ่าน."""
    keys = missing_fields(data, not_applicable, required)
    if primary_amount and primary_amount not in not_applicable and _get(data, primary_amount) in (None, ""):
        keys.append(primary_amount)
    keys += [f for c in checks if c.passed is False for f in c.fields]
    return list(dict.fromkeys(keys))


def _show(v) -> str:
    return v.isoformat() if isinstance(v, date) else str(v)


def known_context(rules: InvoiceData, unresolved: Collection[str]) -> dict[str, str]:
    """ค่าที่ rules ได้และไม่ติดใจ — ส่งให้ LLM เป็นบริบท (ไม่รวมรายการ/หมายเหตุที่ยาว)."""
    return {k: _show(v) for k, v in flatten(rules).items()
            if k not in unresolved and not k.startswith(("items[", *SKIP_PREFIXES))}


def _set(d: InvoiceData, key: str, value) -> None:
    if key.startswith(DOC_PREFIX):
        d.document_fields[key[len(DOC_PREFIX):]] = value
    elif "." in key:
        obj, attr = key.split(".", 1)
        setattr(getattr(d, obj), attr, value)
    else:
        setattr(d, key, value)


def _other_party_has(key: str, value, rules_flat: dict, cand: dict) -> bool:
    party, attr = key.split(".", 1)
    other = ("buyer" if party == "vendor" else "vendor") + "." + attr
    theirs = rules_flat.get(other) or cand.get(other)
    return theirs not in (None, "") and comparable(theirs) == comparable(value)


def merge_llm(
    rules: InvoiceData, llm: InvoiceData, asked: Collection[str], unresolved: Collection[str], text: str,
    profile: Profile, score: Callable[[InvoiceData], float],
) -> tuple[InvoiceData, set[str], list[tuple[str, str]], dict[str, bool]]:
    """รับค่าจาก ``llm`` เข้าผล ``rules`` เฉพาะช่องที่ยังไม่เสร็จ/ว่าง และผ่านด่าน

    ``asked`` = property ระดับบนที่ถาม LLM, ``score`` = คะแนนความสอดคล้องของ check (ผ่าน − ไม่ผ่าน)
    คืน (ผลรวม, คีย์ที่รับจาก LLM, [(คีย์, เหตุผลที่ปัด)], ผลเทียบสองแหล่ง)
    """
    rules_flat, llm_flat = flatten(rules), flatten(llm)
    grounded = field_grounding(llm, text, profile)
    cand = {k: v for k, v in llm_flat.items() if top_level(k) in asked and not k.startswith(SKIP_PREFIXES)}
    agree = {k: comparable(v) == comparable(rules_flat[k]) for k, v in cand.items()
             if k in rules_flat and not k.startswith("items[")}

    rejected: list[tuple[str, str]] = []
    scalar: dict[str, object] = {}
    for k, v in cand.items():
        if k.startswith("items[") or (k in rules_flat and k not in unresolved) or comparable(v) == comparable(rules_flat.get(k)):
            continue
        if grounded.get(k) is False:
            rejected.append((k, "ไม่พบในข้อความ"))
        elif k.endswith("tax_id") and not is_valid_thai_tax_id(v):
            rejected.append((k, "เลขภาษีไม่ผ่าน checksum"))
        elif k.split(".")[-1] in ("name", "tax_id", "address") and k.startswith(("vendor.", "buyer.")) and _other_party_has(k, v, rules_flat, cand):
            rejected.append((k, "ซ้ำกับค่าของอีกฝ่าย"))  # ชื่อ/เลขภาษี/ที่อยู่เดียวกันทั้งสองฝ่าย = ใส่ผิดฝ่าย (สาขาไม่นับ: ทั้งสองฝ่ายเป็นสำนักงานใหญ่ได้) (เช่น LLM ใส่เลขภาษีผู้ขายให้ผู้ซื้อ)
        else:
            scalar[k] = v

    # รายการสินค้า: รับทั้งชุดหรือไม่รับเลย (สองแหล่งแบ่งรายการไม่ตรงกัน เทียบรายตัวไม่ได้)
    take_items = False
    if "items" in asked and llm.items and (not rules.items or any(k.startswith("items[") for k in unresolved)):
        bad = [k for k in cand if k.startswith("items[") and grounded.get(k) is False]
        if bad:
            rejected.append(("items", f"ไม่พบในข้อความ: {bad[0]}"))
        else:
            take_items = True

    def build(keys: Collection[str], items: bool) -> InvoiceData:
        merged = rules.model_copy(deep=True)
        for k in keys:
            _set(merged, k, scalar[k])
        if items:
            merged.items = [it.model_copy(deep=True) for it in llm.items]
        return merged

    # ยอดเงิน/รายการ: ต้องไม่ทำให้ check รวมแย่ลงกว่า rules ล้วน ไม่งั้นทิ้งชุดนั้น
    amountish = {k for k in scalar if k in AMOUNT_KEYS}
    merged = build(scalar, take_items)
    if (amountish or take_items) and score(merged) < score(rules):
        for k in sorted(amountish):
            rejected.append((k, "ทำให้ check รวมแย่ลง"))
            del scalar[k]
        if take_items:
            rejected.append(("items", "ทำให้ check รวมแย่ลง"))
        take_items = False
        merged = build(scalar, False)

    accepted = set(scalar)
    if take_items:
        accepted |= {k for k in flatten(merged) if k.startswith("items[")}
    return merged, accepted, rejected, agree
