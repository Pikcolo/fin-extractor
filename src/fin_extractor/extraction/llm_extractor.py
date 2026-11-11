"""ใช้ LLM แปลงข้อความ OCR/Markdown → JSON ตาม schema.

ทนต่อเอกสารที่ "ไม่มี pattern" ได้ดีกว่า rule-based มาก เพราะ LLM เข้าใจบริบท
ผลลัพธ์ยังต้องผ่าน normalize + validation เสมอ (LLM อาจ hallucinate ตัวเลข)
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from ..doctypes import GENERIC_PROFILE, Profile
from ..schema import InvoiceData
from ..vlm.client import ChatClient
from ..vlm.prompts import EXTRACTION_SYSTEM_PROMPT, extraction_user_prompt
from .patterns import normalize_tax_id, parse_amount, parse_date
from .core_labels import split_branch
from .profile_extractor import normalize_value

_AMOUNT_KEYS = ("subtotal", "discount", "amount_before_vat", "vat_rate", "vat_amount", "total_amount")


def _parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group())


def profile_schema(profile: Profile) -> dict[str, Any]:
    """JSON schema ที่ส่งให้ LLM: แกนกลางของ InvoiceData + ``document_fields`` ตามฟิลด์ของโปรไฟล์ (generic = ไม่มีช่องนี้)

    schema เล็กลงตามชนิดเอกสาร ซึ่งสำคัญกับ LLM เล็กที่ context จำกัด และ LLM เห็นเฉพาะคีย์ที่ชนิดนั้นมีจริง
    """
    schema = InvoiceData.model_json_schema()
    props = dict(schema["properties"])
    props.pop("document_profile", None)
    for name in profile.not_applicable:  # ช่องที่ชนิดนี้ไม่มี (เช่น 50 ทวิไม่มีรายการสินค้า/ยอดรวมแบบใบกำกับ) ไม่ให้ LLM เห็น
        props.pop(name, None)
    if profile.fields:
        props["document_fields"] = {
            "type": "object",
            "description": f"ฟิลด์เฉพาะ{profile.title or profile.name} ละคีย์ที่ไม่พบในเอกสาร",
            "properties": {f.key: {"type": "string", "description": f"{f.label} ({f.type})"} for f in profile.fields},
            "additionalProperties": False,
        }
    else:
        props.pop("document_fields", None)
    return {**schema, "properties": props}


def normalize_llm_output(raw: dict[str, Any], profile: Profile | None = None) -> dict[str, Any]:
    """แก้รูปแบบที่ LLM ชอบตอบผิด ก่อนส่งเข้า pydantic."""
    profile = profile or GENERIC_PROFILE
    out = dict(raw)
    for party in ("vendor", "buyer"):
        p = out.get(party)
        if isinstance(p, dict):
            p["tax_id"] = normalize_tax_id(str(p.get("tax_id") or ""))
        elif p in (None, ""):
            out.pop(party, None)
        else:
            out[party] = {"name": str(p)}
    buyer = out.get("buyer")
    if isinstance(buyer, dict) and buyer.get("name"):  # "(สำนักงานใหญ่)" ท้ายชื่อ → สาขา เหมือนฝั่ง rules ไม่งั้นสองแหล่งเทียบกันไม่ตรง
        buyer["name"], branch = split_branch(str(buyer["name"]))
        buyer["branch"] = buyer.get("branch") or branch
    if out.get("date") is not None:
        out["date"] = parse_date(str(out["date"]))
    for key in _AMOUNT_KEYS:
        if out.get(key) is not None:
            out[key] = parse_amount(out[key])
    if out.get("vat_rate") and out["vat_rate"] > 1:  # ตอบ 7 แทน 0.07
        out["vat_rate"] = out["vat_rate"] / 100
    items = []
    for it in out.get("items") or []:
        if not isinstance(it, dict):
            continue
        for k in ("quantity", "unit_price", "amount"):
            it[k] = parse_amount(it.get(k)) if it.get(k) is not None else None
        it["description"] = str(it.get("description") or "-")
        items.append(it)
    out["items"] = items
    remark = _normalize_extra(out.pop("remark", None) or out.pop("extra_fields", None))
    out.pop("extra_fields", None)
    out["document_fields"], leftover = _normalize_doc_fields(out.get("document_fields"), profile)
    out["remark"] = remark + leftover  # คีย์ที่โปรไฟล์ไม่ได้ประกาศ/อ่านค่าไม่ได้ ไม่ทิ้ง เก็บเป็นหมายเหตุ
    return out


def _normalize_doc_fields(raw: Any, profile: Profile) -> tuple[dict[str, str], list[dict[str, str]]]:
    """คืน (ฟิลด์ประจำชนิดที่ normalize แล้ว, รายการที่ไม่ผ่านเป็นหมายเหตุ).  ค่าว่าง = ไม่มีคีย์ ไม่ใช่สตริงว่าง."""
    fields: dict[str, str] = {}
    leftover: list[dict[str, str]] = []
    for key, value in (raw.items() if isinstance(raw, dict) else ()):
        if value is None or str(value).strip() == "":
            continue
        spec = profile.spec(str(key))
        norm = normalize_value(spec.type, value) if spec else None
        if norm is not None:
            fields[spec.key] = norm
        else:
            leftover.append({"label": spec.label if spec else str(key), "value": str(value).strip()})
    return fields, leftover


def _normalize_extra(raw: Any) -> list[dict[str, str]]:
    """รับทั้ง [{"label","value"}] และ {"ป้าย": "ค่า"} ตัดรายการที่ค่าว่างทิ้ง."""
    pairs = raw.items() if isinstance(raw, dict) else (
        (e.get("label"), e.get("value")) for e in raw or [] if isinstance(e, dict))
    out = [{"label": str(k).strip(), "value": str(v).strip()}
           for k, v in pairs if k and v is not None and str(v).strip()]
    # LLM เล็กมักตอบป้ายซ้ำเป็นค่า (ช่องว่างในแบบฟอร์ม) ซึ่งไม่ใช่ข้อมูล และ grounded ตรวจไม่เจอเพราะค่ามีอยู่ในข้อความจริง
    return [e for e in out if "".join(e["label"].split()).casefold() != "".join(e["value"].split()).casefold()]


class LLMExtractor:
    def __init__(self, cfg: dict[str, Any]):
        if not cfg.get("model"):
            raise ValueError("ต้องตั้ง llm_extractor.model ใน config")
        self.cfg = cfg
        self.client = ChatClient(cfg)

    @property
    def name(self) -> str:
        return f"llm:{self.cfg['model']}"

    def extract(self, document_text: str, profile: Profile | None = None) -> InvoiceData:
        profile = profile or GENERIC_PROFILE
        schema_json = json.dumps(profile_schema(profile), ensure_ascii=False)
        resp = self.client.chat(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": extraction_user_prompt(document_text, schema_json, profile.title)},
            ],
            json_mode=True,
        )
        raw = _parse_json(resp.content or "{}")
        try:
            data = InvoiceData.model_validate(normalize_llm_output(raw, profile))
            data.document_profile = profile.name
            return data
        except ValidationError as e:
            raise ValueError(f"LLM ตอบไม่ตรง schema: {e}") from e
