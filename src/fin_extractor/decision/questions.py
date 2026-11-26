"""สร้าง state + คำถามของแต่ละงาน (docs/draft-11 หัวข้อ 4) — ที่เดียวที่กำหนดว่าป้อนอะไรเข้า decision model

ป้อนเฉพาะฟิลด์ที่สกัดแล้ว และข้อความท่อนแรกของหน้า (ท่อน ``s00`` ของ .md = region บนสุดของผัง เป็นขอบเขตตามโครงหน้า
ไม่ใช่จำนวนบรรทัดที่ตั้งเอง) **ไม่ป้อนข้อความทั้งหน้า** (ข้อความในเอกสารไว้ใจไม่ได้ และลดสิ่งที่โมเดลต้องอ่าน)

งาน (เปิดทีละงานด้วย ``decision.tasks``)
    doc_type   choice: ชนิดเอกสารตามที่พิมพ์ จากรายการ ``decision.doc_types`` (ไม่มีรายการ = ไม่ถาม)
    profile    choice: โปรไฟล์ใน ``document_types`` + generic (เทียบกับ doctypes.classify)
    party_name yes/no: ค่าในช่องชื่อผู้ขาย/ผู้ซื้อเป็นชื่อบริษัทหรือบุคคลล้วน ๆ ไหม (ถามเฉพาะช่องที่มีค่า)
"""

from __future__ import annotations

from ..doctypes import Profile
from ..extraction.rule_extractor import chunk_lines
from ..schema import InvoiceData

GENERIC_DESC = "เอกสารทั่วไปที่ไม่ตรงกลุ่มอื่น (ใบกำกับภาษี / ใบเสร็จรับเงิน / ใบแจ้งหนี้ / ใบส่งของ)"


def first_chunk(raw_text: str | None) -> str:
    """บรรทัดของท่อน s00 (ไม่รวมท่อนเสริมของชั้นตรวจความครบ) ว่างถ้าข้อความไม่มีคอมเมนต์ <!-- sNN -->."""
    return "\n".join(ln for ln, c, s in chunk_lines(raw_text or "") if c == "s00" and not s)


def build(data: InvoiceData, raw_text: str | None, profiles: list[Profile], cfg: dict) -> list[dict]:
    tasks = set(cfg.get("tasks") or [])
    head = {"ข้อความหัวเอกสาร (OCR)": first_chunk(raw_text),
            "ชื่อผู้ขายที่สกัดได้": data.vendor.name, "ชื่อผู้ซื้อที่สกัดได้": data.buyer.name}
    out: list[dict] = []
    if "doc_type" in tasks and cfg.get("doc_types"):
        out.append({"task": "doc_type", "key": "document_type", "type": "choice", "state": head,
                    "instructions": "เอกสารนี้เป็นเอกสารชนิดใด ตามชื่อเอกสารที่พิมพ์บนหัวเอกสาร",
                    "options": {t: None for t in cfg["doc_types"]}})
    if "profile" in tasks:
        opts = {p.name: p.title or None for p in profiles if p.name != "generic"} | {"generic": GENERIC_DESC}
        out.append({"task": "profile", "key": "document_profile", "type": "choice", "state": head,
                    "instructions": "เอกสารนี้จัดอยู่ในกลุ่มใด", "options": opts})
    if "party_name" in tasks:
        for key, value in (("vendor.name", data.vendor.name), ("buyer.name", data.buyer.name)):
            if value:
                out.append({"task": "party_name", "key": key, "type": "noul", "state": {"ค่าในช่องชื่อ": value},
                            "instructions": "ข้อความนี้เป็นชื่อบริษัทหรือชื่อบุคคลเท่านั้น ไม่มีข้อความอื่นปน ใช่หรือไม่"})
    return out
