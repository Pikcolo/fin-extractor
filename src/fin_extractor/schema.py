"""Data Contract — โครงสร้าง JSON มาตรฐานของผลลัพธ์.

ทุก engine (ตัวอ่านทุกตัว + rules / LLM) ต้องคืนค่าเป็น ``ExtractionResult``
เพื่อให้ระบบปลายทางใช้ได้โดยไม่ต้องรู้ว่ามาจาก engine ไหน
รัน ``python -m fin_extractor schema`` เพื่อ export เป็น schemas/invoice.schema.json

ยอดเงินเป็น ``Decimal`` (JSON เป็นสตริง เช่น "47.56") ปัดเป็น 2 ตำแหน่งตามหน่วยสตางค์ของบาท
``vat_rate``, ``quantity`` และ ``unit_price`` เป็น Decimal เช่นกันแต่ไม่ถูกปัดเศษ
"""

from __future__ import annotations

from datetime import date as Date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

SCHEMA_VERSION = "4.0"  # 4.0: field_evidence → review (ระดับ + เหตุผล) · ตัด missing_fields/handwritten_fields ออกจาก JSON (ดู docs/05-validation-output.md)
_CENT = Decimal("0.01")


def _to_cents(v: Decimal | None) -> Decimal | None:
    return v.quantize(_CENT, rounding=ROUND_HALF_UP) if v is not None else None


class LineItem(BaseModel):
    description: str
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = Field(None, description="ยอดเงินของรายการ (ก่อนส่วนลดท้ายบิล)")

    @field_validator("amount")
    @classmethod
    def _amount_cents(cls, v):
        return _to_cents(v)


class Party(BaseModel):
    name: str | None = None
    tax_id: str | None = Field(None, pattern=r"^\d{13}$", description="เลขประจำตัวผู้เสียภาษี 13 หลัก")
    branch: str | None = Field(None, description="สำนักงานใหญ่ / เลขสาขา 5 หลัก")
    address: str | None = None


class ExtraField(BaseModel):
    """รายการหมายเหตุ (remark): ข้อมูลที่เอกสารพิมพ์ไว้แต่ไม่มีช่องใน schema ของชนิดเอกสารนั้น (เช่น เลขที่ใบสั่งซื้อ, เงื่อนไขชำระเงิน)."""

    label: str = Field(description="ป้ายตามที่เอกสารพิมพ์ ใช้ภาษาเดิมของเอกสาร")
    value: str = Field(description="ค่าตามที่เอกสารพิมพ์ เป็นสตริงตามเดิม ไม่แปลงชนิด")
    kind: Literal["pair", "unmapped"] = Field(
        "pair", description="pair = บรรทัดรูป ป้าย: ค่า ที่ไม่มีช่องรองรับ · unmapped = บรรทัดที่มียอดเงิน/วันที่/เลขภาษี แต่ไม่มีช่องรับค่านั้น (ป้าย = (ไม่มีช่องรองรับ) ค่า = ทั้งบรรทัด)")
    source_chunk: str | None = Field(None, description="รหัส section ที่บรรทัดนี้มาจาก (sNN ใน .md ของขั้น OCR รายส่วน) None = ไม่ทราบ (ข้อความไม่มีคอมเมนต์ <!-- sNN -->)")


class RejectedValue(BaseModel):
    """ค่าที่ LLM เสนอแต่ด่านของ hybrid ปัดตก เก็บไว้ให้คน debug ได้ว่าทำไมช่องนั้นว่าง."""

    key: str = Field(description="คีย์ฟิลด์ เช่น vendor.tax_id")
    reason: str


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_type: str | None = Field(None, description="ชนิดเอกสารตามที่ระบบสกัดได้ (rules ได้ชื่อชนิดจาก document_kinds ใน configs/default.yaml เช่น tax_invoice/receipt/delivery_note, LLM ตอบข้อความอิสระ)")
    document_profile: str | None = Field(
        None, description="โปรไฟล์ที่ใช้กำหนดฟิลด์ (ชื่อใน configs/default.yaml → document_types; generic = ไม่ตรงชนิดใดเลย)")
    vendor: Party = Field(default_factory=Party)
    buyer: Party = Field(default_factory=Party)
    book_no: str | None = Field(None, description="เล่มที่ (บิลเล่ม) ใช้คู่กับ invoice_no กันเอกสารซ้ำ")
    invoice_no: str | None = None
    date: Date | None = Field(None, description="ISO-8601 (ค.ศ.) — แปลงจาก พ.ศ. แล้ว")
    currency: str = "THB"
    items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = Field(None, description="รวมเงินก่อนส่วนลด")
    discount: Decimal | None = None
    amount_before_vat: Decimal | None = Field(None, description="มูลค่าก่อนภาษีมูลค่าเพิ่ม")
    vat_rate: Decimal | None = Field(None, description="เช่น 0.07")
    vat_amount: Decimal | None = None
    total_amount: Decimal | None = Field(None, description="ยอดสุทธิรวม VAT")
    document_fields: dict[str, str] = Field(
        default_factory=dict,
        description="ฟิลด์เฉพาะชนิดเอกสาร คีย์ตามโปรไฟล์ (เช่น หัก ณ ที่จ่าย: total_paid, total_tax_withheld) "
        "ค่าเป็นสตริงที่ normalize แล้วตามชนิดที่โปรไฟล์ประกาศ: money=ทศนิยมไม่มีจุลภาค, date=ISO ค.ศ., tax_id=13 หลัก, percent=ทศนิยม",
    )
    remark: list[ExtraField] = Field(
        default_factory=list,
        description="หมายเหตุ: ข้อมูลอื่นที่เอกสารพิมพ์ไว้แต่ไม่มีช่องข้างบนหรือใน document_fields (ป้าย + ค่า) ห้ามใส่ข้อมูลที่มีช่องอยู่แล้ว",
    )

    @field_validator("subtotal", "discount", "amount_before_vat", "vat_amount", "total_amount")
    @classmethod
    def _money_cents(cls, v):
        return _to_cents(v)


class ValidationCheck(BaseModel):
    name: str
    passed: bool | None = Field(None, description="None = ข้อมูลไม่พอให้ตรวจ")
    detail: str = ""
    fields: list[str] = Field(
        default_factory=list,
        description="field ที่ check นี้ใช้จริง (ชื่อแบบ vendor.tax_id, items[0].amount) — ว่างเมื่อข้อมูลไม่พอให้ตรวจ",
    )


EvidenceLevel = Literal["missing", "not_applicable", "conflict", "verified", "unverified"]


class FieldEvidence(BaseModel):
    """หลักฐานละเอียดของ field หนึ่งช่อง **ใช้ภายใน** (validation/evidence.py) แล้วสรุปเป็น ``FieldReview`` ที่บันทึกในผล
    (schema 3.0 บันทึกตัวนี้ตรง ๆ ใน ``field_evidence`` ช่องส่วนใหญ่ว่างเสมอ) ไม่ใช่ความน่าจะเป็น ไม่มีน้ำหนักหรือ threshold."""

    level: EvidenceLevel = Field(
        description="missing=เอกสารชนิดนี้ควรมีแต่หาไม่เจอ · not_applicable=เอกสารชนิดนี้ไม่มีฟิลด์นี้ · conflict=ขัดแย้ง ต้องแก้หรือยืนยัน · "
        "verified=มี check ผ่านและไม่มีอะไรขัด · unverified=ไม่มีตัวยืนยัน ให้ดูเทียบภาพ"
    )
    value: str | None = Field(None, description="ค่าที่ระบบสกัดได้ (เป็นสตริง) ไว้เทียบกับภาพ")
    format_valid: bool | None = Field(None, description="ผ่านรูปแบบ (เลขภาษี 13 หลัก + checksum); None = field นี้ไม่มีกฎรูปแบบ")
    checks_passed: list[str] = Field(default_factory=list, description="check ที่ผ่านและยืนยันค่านี้ได้")
    checks_failed: list[str] = Field(default_factory=list, description="check ที่ใช้ค่านี้และไม่ผ่าน")
    grounded: bool | None = Field(None, description="ค่าปรากฏในข้อความที่อ่านได้หรือไม่; None = ไม่ได้ตรวจ")
    sources_agree: bool | None = Field(None, description="rules กับ LLM ได้ค่าเดียวกันหรือไม่; None = มีแหล่งเดียว")
    handwritten: bool | None = Field(None, description="True = อ่านจากลายมือ; None = ยังไม่มีตัวตรวจลายมือ")
    model_conf: float | None = Field(None, description="ยังไม่ได้เก็บ (ต้นทางระดับคำของ OCR ยังไม่ถูกส่งต่อ) จึงเป็น null เสมอ")
    source: str | None = Field(None, description="ตัวสกัดที่ให้ค่านี้: rules หรือ llm")


class FieldReview(BaseModel):
    """ฟิลด์หนึ่งช่องในมุมของคนตรวจ: ต้องดูไหม และทำไม (ค่าอยู่ใน ``data`` ที่เดียว)."""

    level: EvidenceLevel = Field(
        description="conflict=ขัดแย้ง ต้องแก้หรือยืนยัน · missing=เอกสารชนิดนี้ต้องมีแต่หาไม่เจอ · unverified=ไม่มีตัวยืนยัน ให้ดูเทียบภาพ · "
        "verified=มี check ผ่านและไม่มีอะไรขัด · not_applicable=เอกสารชนิดนี้ไม่มีฟิลด์นี้"
    )
    reasons: list[str] = Field(default_factory=list, description="เหตุผลเป็นคำพูด เช่น check ที่ไม่ผ่าน, ค่าไม่ปรากฏในข้อความ, check ที่ยืนยันค่านี้")
    source: str | None = Field(None, description="ผู้ให้ค่าเมื่อไม่ใช่ rules: llm | human | ชื่อตัวอ่านเสริม เช่น openthai-text "
                               "(ค่าที่พบเฉพาะในข้อความเสริมของชั้นตรวจความครบ) (ไม่มีคีย์ = rules)")

    @model_serializer(mode="wrap")
    def _drop_empty_source(self, handler):
        d = handler(self)
        if d.get("source") is None:
            d.pop("source", None)
        return d


class ExtractionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source_file: str
    page: int = 1
    engine: str = Field(..., description="<engine>[:<โมเดล>]+<ผู้สกัด> เช่น paddle+rules, openthai:OpenThai2.0-OCR-Markdown+rules, typhoon:scb10x/typhoon-ocr1.5-3b+llm")
    data: InvoiceData
    checks: list[ValidationCheck] = Field(default_factory=list)
    review: dict[str, FieldReview] = Field(
        default_factory=dict,
        description="ระดับ + เหตุผลรายฟิลด์ เรียงตามที่ควรตรวจก่อน (conflict → missing → unverified → verified → not_applicable) "
        "คีย์แบบ vendor.tax_id, total_amount, items[0].amount, document_fields.<key> · ฟิลด์ที่ไม่มีค่าและไม่บังคับไม่อยู่ที่นี่",
    )
    status: Literal["ready", "check", "incomplete"]
    ocr_confidence: float | None = Field(None, ge=0, le=1, description="ค่าเฉลี่ยที่ OCR engine รายงานเอง (มีเฉพาะ Paddle / OpenThai Text) ไม่มี = ไม่มีคีย์")
    raw_text: str | None = Field(None, description="ข้อความดิบจากตัวอ่าน (เฉพาะเมื่อสั่ง --keep-raw) ไว้ debug")
    rejected: list[RejectedValue] = Field(default_factory=list, description="ค่าที่ LLM เสนอแต่ไม่ถูกรับ (โหมด hybrid) พร้อมเหตุผล ไม่มี = ไม่มีคีย์")

    @property
    def missing_fields(self) -> list[str]:
        """ฟิลด์ที่เอกสารชนิดนี้ต้องมีแต่หาไม่เจอ (= ``review`` ระดับ missing) ไม่บันทึกแยกใน JSON แล้ว (ซ้ำกับ review)."""
        return [k for k, r in self.review.items() if r.level == "missing"]

    @model_serializer(mode="wrap")
    def _drop_empty_extras(self, handler):
        """ช่องเสริมที่ว่างไม่ถูกเขียนลง JSON (ค่าใน ``data`` ยังเขียน null ตามเดิม: ไม่มีค่า ≠ ไม่มีช่อง)."""
        d = handler(self)
        for k in ("ocr_confidence", "raw_text"):
            if d.get(k) is None:
                d.pop(k, None)
        if not d.get("rejected"):
            d.pop("rejected", None)
        return d
