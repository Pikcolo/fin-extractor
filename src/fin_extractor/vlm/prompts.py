"""Prompt สำหรับ VLM / LLM — แยกไว้ที่เดียวเพื่อปรับจูนได้ง่าย."""

# อิงรูปแบบ prompt ของ Typhoon OCR 1.5 (ดู model card: huggingface.co/scb10x)
# ถ้าเปลี่ยนรุ่นโมเดล ให้เทียบ prompt กับ model card ของรุ่นนั้นอีกครั้ง
TYPHOON_OCR_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


EXTRACTION_SYSTEM_PROMPT = """คุณคือระบบสกัดข้อมูลจากเอกสารการเงินไทย (ใบกำกับภาษี ใบเสร็จรับเงิน invoice)
ตอบเป็น JSON object เท่านั้น ตาม schema ที่ให้ ห้ามมีข้อความอื่น

กฎ:
- ใช้เฉพาะข้อมูลที่ปรากฏในเอกสาร ถ้าไม่พบให้ใส่ null ห้ามเดา
- vendor = ผู้ขาย/ผู้ออกเอกสาร (มักอยู่หัวกระดาษ), buyer = ลูกค้า/ผู้ซื้อ
- tax_id: ตัวเลข 13 หลักติดกัน ไม่มีขีดหรือช่องว่าง
- date: รูปแบบ YYYY-MM-DD เป็นปี ค.ศ. (ถ้าเอกสารเป็น พ.ศ. ให้ลบ 543)
- จำนวนเงินเป็นตัวเลข ไม่มีเครื่องหมายจุลภาคหรือสกุลเงิน
- vat_rate เป็นทศนิยม เช่น 7% → 0.07
- subtotal = ยอดรวมก่อนหักส่วนลด, amount_before_vat = มูลค่าก่อนภาษี, total_amount = ยอดสุทธิรวมภาษี
- items: ทุกบรรทัดรายการสินค้า/บริการ พร้อม quantity และ unit_price ถ้าตารางมีคอลัมน์นั้น
- document_fields (ถ้า schema มีช่องนี้): ฟิลด์เฉพาะของเอกสารชนิดนี้ คีย์ตาม schema ค่าเป็นสตริงตามที่พิมพ์ ไม่พบให้ละคีย์นั้น
- remark: ข้อมูลที่เอกสารพิมพ์ไว้แต่ไม่มีช่องใน schema (เช่น เลขที่ใบสั่งซื้อ, เงื่อนไขชำระเงิน)
  ใส่เป็น {"label": ป้ายตามที่พิมพ์, "value": ค่าตามที่พิมพ์} ห้ามใส่ข้อมูลที่มีช่องอยู่แล้ว (รวมถึง document_fields) ถ้าไม่มีให้เป็น []"""


def extraction_user_prompt(document_text: str, json_schema: str, document_kind: str = "") -> str:
    kind = f"ชนิดเอกสาร: {document_kind}\n\n" if document_kind else ""
    return f"""{kind}JSON schema:
{json_schema}

เนื้อหาเอกสาร (จาก OCR อาจมีตัวอักษรผิดบ้าง):
<document>
{document_text}
</document>

ตอบเป็น JSON object เท่านั้น"""
