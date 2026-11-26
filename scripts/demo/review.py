"""ขั้น "ตรวจ/แก้รายฟิลด์" ของหน้า demo: เรียงฟิลด์ตามที่ควรตรวจก่อน บอกเหตุผลเป็นคำพูด ให้คนแก้ค่าแล้วตรวจใหม่ทันที

ไม่ใช้ decision engine: ระดับและเหตุผลมาจาก ``review`` ของผล (schema 4.0: ``validation/evidence.review_from_evidence``) ตัวเดียวกับที่ CLI บันทึก
หน้านี้เพิ่มแค่การแสดงผล: แถวของฟิลด์ที่ไม่มีค่า (ให้กรอกได้) ช่องที่คนแก้ (``source: human``) แสดงเป็น "แก้แล้ว" และตารางรายการสินค้าที่แก้ได้
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import ValidationError

from fin_extractor.doctypes import DOC_PREFIX
from fin_extractor.extraction.patterns import parse_date
from fin_extractor.groundtruth import FIELDS, ITEM_KEYS, MONEY
from fin_extractor.pipeline import FinanceOCRPipeline
from fin_extractor.schema import ExtractionResult, InvoiceData, ValidationCheck
from fin_extractor.validation.evidence import flatten, review_from_evidence

PRIORITY = {"conflict": 0, "missing": 1, "unverified": 2, "edited": 3, "verified": 4, "empty": 5, "not_applicable": 6}  # ลำดับแถวบนหน้าเว็บ


def _set(d: dict, key: str, value: Any) -> None:
    if key.startswith(DOC_PREFIX):
        name = key[len(DOC_PREFIX):]
        if value is None:
            d["document_fields"].pop(name, None)
        else:
            d["document_fields"][name] = value
        return
    *parents, last = key.split(".")
    for p in parents:
        d = d[p]
    d[last] = value


def _clean(key: str, value: Any) -> Any:
    """ค่าที่คนพิมพ์ → ค่าที่ schema รับ: ว่าง = None · ยอดเงินตัด , และ "บาท" · วันที่รับได้ทั้ง ISO และแบบไทย (พ.ศ.) ผ่าน parse_date ตัวเดียวกับ rules."""
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip()
    name = key.split(".")[-1]
    if name in MONEY:
        return re.sub(r"[,\s]|บาท|THB", "", v)
    if name == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        d = parse_date(v)
        return d.isoformat() if d else v
    if name == "tax_id":
        return re.sub(r"\D", "", v) or v
    return v


def apply_edits(data: dict, edits: dict[str, Any]) -> tuple[InvoiceData | None, dict[str, str]]:
    """ข้อมูลเดิม (dict ของ ``InvoiceData``) + การแก้ {คีย์แบบ flatten: ค่า} → (ข้อมูลใหม่ หรือ None ถ้าค่าไม่ผ่าน schema, ข้อผิดพลาดรายคีย์)."""
    d = copy.deepcopy(data)
    for key, value in edits.items():
        if key == "items":  # ทั้งตารางรายการ: [{description, quantity, unit_price, amount}] แถวว่างทั้งแถวถูกทิ้ง
            d["items"] = [{**{k: _clean(k, it.get(k)) for k in ITEM_KEYS}, "description": str(it.get("description") or "").strip() or "-"}
                          for it in value if any(str(it.get(k) or "").strip() for k in ITEM_KEYS)]
            continue
        _set(d, key, _clean(key, value))
    try:
        return InvoiceData.model_validate(d), {}
    except ValidationError as e:
        errors = {}
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors[loc] = err["msg"]
        return None, errors


def build(pipe: FinanceOCRPipeline, data: InvoiceData, lines: list[str], extra: list[ValidationCheck],
          edited: set[str], sections: dict[str, list[str]], original: dict[str, Any]) -> dict:
    """ผลตรวจของข้อมูลชุดนี้ในรูปที่หน้าเว็บใช้: แถวรายฟิลด์ (เรียงตามที่ควรตรวจก่อน) + check + สถานะ."""
    res = corrected_result({"source_file": "", "page": 1, "engine": ""}, pipe, data, lines, extra, edited)
    checks, status, missing = res.checks, res.status, res.missing_fields
    profile = pipe.profile_for(data)
    flat = {k: v for k, v in flatten(data).items()}
    keys = [*FIELDS, *(DOC_PREFIX + f.key for f in profile.fields)]
    labels = {DOC_PREFIX + f.key: f.label for f in profile.fields}
    rows = []
    for k in keys:
        rv = res.review.get(k)
        if rv is None:  # ไม่มีค่าและไม่บังคับ: ไม่อยู่ใน review ของผล แต่หน้าเว็บให้กรอกได้
            level, reasons = "empty", ["ไม่มีค่า · ฟิลด์นี้ไม่บังคับสำหรับเอกสารชนิดนี้ ถ้าเอกสารมีให้กรอก"]
        else:
            level, reasons = ("edited" if rv.source == "human" and rv.level == "unverified" else rv.level), rv.reasons
        v = flat.get(k)
        rows.append({"key": k, "label": labels.get(k), "value": "" if v is None else str(v), "level": level, "reasons": reasons,
                     "edited": k in edited, "original": original.get(k, ""), "sections": sections.get(k, [])})
    rows.sort(key=lambda r: (PRIORITY[r["level"]], keys.index(r["key"])))
    items_check = next((c for c in checks if c.name == "items_sum"), None)
    items_reason = ("ผลรวมจำนวนเงินของรายการ = ยอดรวม ผ่าน" if items_check and items_check.passed
                    else f"ผลรวมรายการไม่ตรงกับยอดรวม ({items_check.detail}): รายการที่ระบบสกัดอาจผิด เช่น เอาแถวยอดรวมมาเป็นรายการ" if items_check and items_check.passed is False
                    else f"ยังตรวจผลรวมรายการไม่ได้ ({items_check.detail})" if items_check else "")
    return {"rows": rows, "status": status, "missing": missing, "profile": profile.name,
            "items": [{k: "" if it.get(k) is None else str(it.get(k)) for k in ITEM_KEYS} for it in data.model_dump(mode="json")["items"]],
            "items_edited": "items" in edited, "items_level": "verified" if items_check and items_check.passed else "conflict" if items_check and items_check.passed is False else "unverified",
            "items_reason": items_reason,
            "checks": [c.model_dump() for c in checks], "data": data.model_dump(mode="json"),
            "result": json.loads(res.model_dump_json())}


def corrected_result(base: dict, pipe: FinanceOCRPipeline, data: InvoiceData, lines: list[str], extra: list[ValidationCheck],
                     edited: set[str]) -> ExtractionResult:
    """ผลหลังคนแก้ (schema เดียวกับผลของระบบ) ตรวจใหม่ทั้งหมดจากค่าที่แก้แล้ว ช่องที่แก้มี ``source: human`` ใน review."""
    sources = {k: ("human" if k in edited or k.split("[")[0] in edited else "rules") for k in flatten(data)}
    checks, _missing, status, evidence = pipe.assess(data, lines, sources, None, extra)
    review = review_from_evidence(evidence, checks, pipe.profile_for(data).primary_amount, sources)
    return ExtractionResult(source_file=base["source_file"], page=base["page"], engine=(base["engine"] + "+human").lstrip("+"), data=data,
                            checks=checks, review=review, ocr_confidence=base.get("ocr_confidence"), status=status)
