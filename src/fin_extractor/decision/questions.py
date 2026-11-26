"""สร้าง state + คำถามของแต่ละงาน (docs/draft-11 หัวข้อ 4) — ที่เดียวที่กำหนดว่าป้อนอะไรเข้า decision model

ข้อมูลที่ป้อน = ข้อความท่อนแรกของหน้า (``s00`` = region บนสุดของผัง) + **บรรทัด OCR ที่มีค่าที่สกัดได้** + ค่าที่สกัดได้
**ไม่ป้อนข้อความทั้งหน้า** (ข้อความในเอกสารไว้ใจไม่ได้ และลดสิ่งที่โมเดลต้องอ่าน) โมเดลจึงเห็นทั้งข้อความดิบที่เกี่ยวข้องและผลหลังสกัดพร้อมกัน

งาน (เปิดด้วย ``decision.tasks``)
    doc_type      choice  ชนิดเอกสารตามที่พิมพ์ จากรายการ ``decision.doc_types``
    profile       choice  โปรไฟล์ใน ``document_types`` + generic
    name_only     yes/no  ค่าในช่องชื่อผู้ขาย/ผู้ซื้อเป็นชื่อบริษัทหรือบุคคลล้วน ๆ ไหม (state = ค่านั้นอย่างเดียว)

ไม่มีคำถามกว้าง "ค่าของช่องนี้ถูกต้องไหม": ทดลองแล้ว (2026-11-26, 18 เคสที่รู้คำตอบ ทั้งแบบให้แค่ค่า แบบให้บรรทัด OCR ของค่า
และแบบถามรวมทุกช่องบน state เดียว) โมเดลตอบว่าถูกทั้ง 9 เคสที่ผิด และจัดอันดับค่าผิดต่ำกว่าค่าถูกได้ ~55% ของคู่ ไม่ต่างจากสุ่ม
คำถามแคบ "เป็นชื่อล้วนไหม" จับ 1321 (คำแนะนำของฟอร์มล้วน) ได้ จึงเหลือแบบนี้ (ผลอยู่ใน docs/draft-11)
    missing_hint  choice  ฟิลด์ที่ต้องมีแต่หาไม่เจอ (ชนิดเงิน/วันที่/เลขภาษี): ค่าไหนในบรรทัด OCR คือค่าของช่องนี้ + "ไม่พบในเอกสาร"
                          **เป็นข้อเสนอให้คนดูเท่านั้น ไม่เขียนลง data** ตัวเลือกยิ่งมากลำดับยิ่งมีผล จึงบันทึก agreed เสมอ
"""

from __future__ import annotations

import re

from ..doctypes import DOC_PREFIX, Profile
from ..extraction.patterns import find_amounts, find_dates, find_tax_ids
from ..extraction.rule_extractor import chunk_lines
from ..schema import InvoiceData

GENERIC_DESC = "เอกสารทั่วไปที่ไม่ตรงกลุ่มอื่น (ใบกำกับภาษี / ใบเสร็จรับเงิน / ใบแจ้งหนี้ / ใบส่งของ)"
NOT_FOUND = "ไม่พบในเอกสาร"

# ชื่อไทยของฟิลด์แกนกลาง (ใช้ในคำถาม missing_hint)
CORE_FIELDS = {
    "vendor.name": "ชื่อผู้ขาย (ผู้ออกเอกสาร / ผู้รับเงิน)", "buyer.name": "ชื่อผู้ซื้อ (ลูกค้า / ผู้จ่ายเงิน)",
    "invoice_no": "เลขที่เอกสาร", "date": "วันที่ออกเอกสาร", "amount_before_vat": "มูลค่าก่อนภาษีมูลค่าเพิ่ม",
    "vat_amount": "ภาษีมูลค่าเพิ่ม", "total_amount": "ยอดรวมสุทธิ",
}
MONEY = {"subtotal", "discount", "amount_before_vat", "vat_amount", "total_amount"}


def first_chunk(raw_text: str | None) -> str:
    """บรรทัดของท่อน s00 (ไม่รวมท่อนเสริมของชั้นตรวจความครบ) ว่างถ้าข้อความไม่มีคอมเมนต์ <!-- sNN -->."""
    return "\n".join(ln for ln, c, s in chunk_lines(raw_text or "") if c == "s00" and not s)


def _lines(raw_text: str | None) -> list[str]:
    return list(dict.fromkeys(ln for ln, _c, _s in chunk_lines(raw_text or "")))


def _squash(s: str) -> str:
    return re.sub(r"[\s,|:]", "", str(s)).lower()


def lines_with(value, kind: str, lines: list[str]) -> list[str]:
    """บรรทัดที่มีค่านี้: เงินเทียบเป็นค่า (1,234.50 = 1234.50) วันที่เทียบเป็นวันที่ ข้อความเทียบแบบตัดช่องว่าง/ตัวคั่น."""
    if value in (None, ""):
        return []
    if kind == "money":
        try:
            from decimal import Decimal

            v = Decimal(str(value))
        except Exception:  # noqa: BLE001
            return []
        return [ln for ln in lines if v in find_amounts(ln)]
    if kind == "date":
        return [ln for ln in lines if any(d.isoformat() == str(value) for d in find_dates(ln))]
    sq = _squash(value)
    return [ln for ln in lines if sq and sq in _squash(ln)]


def _kind(key: str, profile: Profile | None) -> str:
    if key.startswith(DOC_PREFIX) and profile and (f := profile.spec(key[len(DOC_PREFIX):])):
        return f.type
    return "money" if key in MONEY else "date" if key == "date" else "text"


def build(data: InvoiceData, raw_text: str | None, profiles: list[Profile], cfg: dict,
          missing: list[str] | tuple = ()) -> list[dict]:
    """คำถามของเอกสารหนึ่งใบ ``missing`` = ฟิลด์ที่ต้องมีแต่หาไม่เจอ (``review`` ระดับ missing) สำหรับ missing_hint."""
    tasks = set(cfg.get("tasks") or [])
    head = first_chunk(raw_text)
    lines = _lines(raw_text)
    profile = next((p for p in profiles if p.name == data.document_profile), None)
    out: list[dict] = []

    names = {"ข้อความหัวเอกสาร (OCR)": head, "ชื่อผู้ขายที่สกัดได้": data.vendor.name, "ชื่อผู้ซื้อที่สกัดได้": data.buyer.name}
    if "doc_type" in tasks and cfg.get("doc_types"):
        out.append({"task": "doc_type", "key": "document_type", "type": "choice", "state": names,
                    "instructions": "เอกสารนี้เป็นเอกสารชนิดใด ตามชื่อเอกสารที่พิมพ์บนหัวเอกสาร",
                    "options": {t: None for t in cfg["doc_types"]}})
    if "profile" in tasks:
        opts = {p.name: p.title or None for p in profiles if p.name != "generic"} | {"generic": GENERIC_DESC}
        out.append({"task": "profile", "key": "document_profile", "type": "choice", "state": names,
                    "instructions": "เอกสารนี้จัดอยู่ในกลุ่มใด", "options": opts})

    if "name_only" in tasks:
        for key, value in (("vendor.name", data.vendor.name), ("buyer.name", data.buyer.name)):
            if value:
                out.append({"task": "name_only", "key": key, "type": "noul", "state": {"ค่าในช่องชื่อ": value},
                            "instructions": "ข้อความนี้เป็นชื่อบริษัทหรือชื่อบุคคลเท่านั้น ไม่มีข้อความอื่นปน ใช่หรือไม่"})

    if "missing_hint" in tasks:
        for k in missing:
            kind = _kind(k, profile)
            label = CORE_FIELDS.get(k) or (profile.spec(k[len(DOC_PREFIX):]).label if profile and k.startswith(DOC_PREFIX)
                                           and profile.spec(k[len(DOC_PREFIX):]) else k)
            if kind == "money":
                cands = {str(a): [ln for ln in lines if a in find_amounts(ln)] for ln in lines for a in find_amounts(ln)}
            elif kind == "date":
                cands = {d.isoformat(): [ln for ln in lines if d in find_dates(ln)] for ln in lines for d in find_dates(ln)}
            elif kind == "tax_id" or k.endswith("tax_id"):
                cands = {t: [ln for ln in lines if t in find_tax_ids(ln)] for ln in lines for t in find_tax_ids(ln)}
            else:
                continue  # ข้อความ/รหัส: ไม่มีวิธีหาค่าที่เป็นไปได้โดยไม่ตั้งเกณฑ์เอง
            if not cands:
                continue
            ctx = list(dict.fromkeys(ln for ls in cands.values() for ln in ls))
            out.append({"task": "missing_hint", "key": k, "type": "choice",
                        "state": {"ข้อความหัวเอกสาร (OCR)": head, "บรรทัดจาก OCR ที่มีค่าที่เป็นไปได้": ctx},
                        "instructions": f"ค่าใดคือ \"{label}\" ของเอกสารนี้",
                        "options": {**{c: None for c in cands}, NOT_FOUND: None}})
    return out
