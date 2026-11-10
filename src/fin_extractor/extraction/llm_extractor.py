"""ใช้ LLM แปลงข้อความ OCR/Markdown → JSON ตาม schema.

ทนต่อเอกสารที่ "ไม่มี pattern" ได้ดีกว่า rule-based มาก เพราะ LLM เข้าใจบริบท
ผลลัพธ์ยังต้องผ่าน normalize + validation เสมอ (LLM อาจ hallucinate ตัวเลข)
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from ..schema import InvoiceData
from ..vlm.client import ChatClient
from ..vlm.prompts import EXTRACTION_SYSTEM_PROMPT, extraction_user_prompt
from .patterns import normalize_tax_id, parse_amount, parse_date

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


def normalize_llm_output(raw: dict[str, Any]) -> dict[str, Any]:
    """แก้รูปแบบที่ LLM ชอบตอบผิด ก่อนส่งเข้า pydantic."""
    out = dict(raw)
    for party in ("vendor", "buyer"):
        p = out.get(party)
        if isinstance(p, dict):
            p["tax_id"] = normalize_tax_id(str(p.get("tax_id") or ""))
        elif p in (None, ""):
            out.pop(party, None)
        else:
            out[party] = {"name": str(p)}
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
    out["extra_fields"] = _normalize_extra(out.get("extra_fields"))
    return out


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
        self.schema_json = json.dumps(InvoiceData.model_json_schema(), ensure_ascii=False)

    @property
    def name(self) -> str:
        return f"llm:{self.cfg['model']}"

    def extract(self, document_text: str) -> InvoiceData:
        resp = self.client.chat(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": extraction_user_prompt(document_text, self.schema_json)},
            ],
            json_mode=True,
        )
        raw = _parse_json(resp.content or "{}")
        try:
            return InvoiceData.model_validate(normalize_llm_output(raw))
        except ValidationError as e:
            raise ValueError(f"LLM ตอบไม่ตรง schema: {e}") from e
