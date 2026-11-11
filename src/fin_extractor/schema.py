"""Data Contract — โครงสร้าง JSON มาตรฐานของผลลัพธ์.

ทุก engine (OCR แบบ rule-based / VLM / LLM) ต้องคืนค่าเป็น ``ExtractionResult``
เพื่อให้ระบบปลายทางใช้ได้โดยไม่ต้องรู้ว่ามาจาก engine ไหน
รัน ``python -m fin_extractor schema`` เพื่อ export เป็น schemas/invoice.schema.json

ยอดเงินเป็น ``Decimal`` (JSON เป็นสตริง เช่น "47.56") ปัดเป็น 2 ตำแหน่งตามหน่วยสตางค์ของบาท
``vat_rate``, ``quantity`` และ ``unit_price`` เป็น Decimal เช่นกันแต่ไม่ถูกปัดเศษ
"""

from __future__ import annotations

from datetime import date as Date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "3.0"
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


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_type: str | None = Field(None, description="ชนิดเอกสารตามที่ระบบสกัดได้ (rules ได้ tax_invoice/receipt/..., LLM ตอบข้อความอิสระ)")
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

    @model_validator(mode="before")
    @classmethod
    def _legacy_extra_fields(cls, data):
        """schema 2.x เรียก remark ว่า extra_fields — รับชื่อเก่าตอนอ่านผลเก่า/คำตอบ LLM."""
        if isinstance(data, dict) and "extra_fields" in data:
            data = dict(data)
            old = data.pop("extra_fields")
            if not data.get("remark"):
                data["remark"] = old
        return data

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
    """หลักฐานของ field หนึ่งช่อง ไว้ไฮไลต์ใน Review UI (ไม่ใช่ความน่าจะเป็น ไม่มีน้ำหนักหรือ threshold)."""

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


class ExtractionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source_file: str
    page: int = 1
    engine: str = Field(..., description="เช่น paddle+rules, vlm:typhoon-ocr+llm")
    data: InvoiceData
    checks: list[ValidationCheck] = Field(default_factory=list)
    field_evidence: dict[str, FieldEvidence] = Field(
        default_factory=dict,
        description="หลักฐานรายฟิลด์ คีย์แบบ vendor.tax_id, total_amount, items[0].amount",
    )
    ocr_confidence: float | None = Field(None, ge=0, le=1, description="ค่าเฉลี่ยที่ OCR engine รายงานเอง (VLM ไม่มีค่านี้)")
    missing_fields: list[str] = Field(default_factory=list, description="field สำคัญที่หาไม่เจอ")
    handwritten_fields: list[str] = Field(
        default_factory=list, description="field ที่อ่านมาจากลายมือ (เช่น vendor.name, total_amount) — ต้องมีหลักฐานยืนยันจึงอนุมัติอัตโนมัติ"
    )
    status: Literal["auto_approved", "needs_review", "failed"]
    raw_text: str | None = Field(None, description="ข้อความดิบจาก OCR/VLM ไว้ debug")
