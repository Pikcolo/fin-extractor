"""โปรไฟล์ชนิดเอกสาร — กำหนดว่าเอกสารแต่ละชนิดมีฟิลด์อะไร ต้องมีอะไร และใช้ยอดไหนตัดสินว่า "ไม่มียอด".

อ่านจากตาราง ``document_types`` ใน configs/default.yaml (ไม่มีรายชื่อชนิดหรือฟิลด์ฝังในโค้ด) ต่อโปรไฟล์

    aliases        ข้อความที่พิมพ์บนเอกสารที่ใช้จำแนกชนิด (เทียบแบบตัดช่องว่าง/ตัวพิมพ์ ลำดับในไฟล์ = ลำดับความสำคัญ)
    required       field ที่ชนิดนี้ต้องมี (ค่าตั้งต้น = ชุดเดิมของใบกำกับภาษี KEY_FIELDS)
    primary_amount field ที่ถ้าหาไม่เจอแล้วเอกสารเป็น ``failed`` (ค่าตั้งต้น total_amount; null = ไม่มีกฎนี้)
    not_applicable field ที่ชนิดนี้ไม่มี (ไม่นับเป็นหาไม่เจอ)
    fields         ฟิลด์เฉพาะชนิด: key, label, type, aliases (ป้ายที่พิมพ์), source (ที่มาของฟิลด์นี้)

โปรไฟล์ที่ไม่มี ``aliases`` เป็นรูปแบบเก่า (มีแค่ ``not_applicable`` เทียบด้วยชื่อชนิด) ยังใช้ได้เหมือนเดิม
ชนิดที่จำแนกไม่ได้ใช้โปรไฟล์ ``generic`` = พฤติกรรมเดิมของใบกำกับภาษีทุกประการ
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

GENERIC = "generic"
FIELD_TYPES = ("text", "code", "money", "date", "tax_id", "percent")
KEY_FIELDS = ("vendor.name", "vendor.tax_id", "invoice_no", "date", "total_amount")
# ชื่อ check ทั้งหมดของ validation/checks.py (โปรไฟล์ประกาศ ``checks`` จากรายการนี้เท่านั้น)
KNOWN_CHECKS = ("vendor_tax_id_checksum", "buyer_tax_id_checksum", "items_sum", "discount",
                "before_vat_plus_vat", "vat_rate", "date_range")
DOC_PREFIX = "document_fields."


def squash(text: str) -> str:
    """ตัดช่องว่างทุกชนิดและ ``|`` แล้วทำเป็นตัวพิมพ์เล็ก — OCR ไทยแทรกช่องว่างกลางคำได้."""
    return "".join(str(text).replace("|", " ").split()).casefold()


def alias_regex(alias: str) -> re.Pattern:
    """regex ของป้ายที่ยอมให้มีช่องว่างแทรกระหว่างตัวอักษร (ใช้หาตำแหน่งป้ายในบรรทัดต้นฉบับ)."""
    chars = [re.escape(c) for c in alias if not c.isspace()]
    return re.compile(r"\s*".join(chars), re.I)


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    type: str = "text"
    aliases: tuple[str, ...] = ()
    source: str = ""
    nth: int = 0  # money: ยอดลำดับที่เท่าไรหลังป้าย (0 = ตัวแรก) ใช้เมื่อเอกสารพิมพ์หลายยอดในบรรทัดเดียว
    of: int | None = None  # money: บรรทัดนั้นต้องมีกี่ยอดพอดี ไม่ครบ = ไม่ให้ค่า (กันหยิบยอดผิดตัวเมื่อ OCR ทำยอดหายไป)
    next_line: bool = False  # text: ถ้าหลังป้ายไม่มีค่าที่ใช้ได้ ลองบรรทัดถัดไป (เช่น 50 ทวิ ชื่ออยู่บรรทัดใต้ป้ายผู้ถูกหักภาษี)
    next_line_caption: bool = False  # text: เหมือน next_line แต่รับเฉพาะบรรทัดถัดไปที่ขึ้นต้นด้วยป้ายกำกับชื่อ ("Name |", "Customer Name:")

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(f"field '{self.key}': type '{self.type}' ไม่รู้จัก (ใช้ได้: {', '.join(FIELD_TYPES)})")
        if self.nth < 0 or (self.of is not None and self.nth >= self.of):
            raise ValueError(f"field '{self.key}': nth={self.nth} ต้องน้อยกว่า of={self.of}")


@dataclass(frozen=True)
class Profile:
    name: str
    title: str = ""
    aliases: tuple[str, ...] = ()
    required: tuple[str, ...] = KEY_FIELDS
    primary_amount: str | None = "total_amount"
    not_applicable: frozenset[str] = frozenset()
    fields: tuple[FieldSpec, ...] = ()
    checks: tuple[str, ...] | None = None  # check ที่ใช้กับชนิดนี้ (None = ทุกข้อ) ข้อที่ไม่ได้ระบุรายงานเป็น passed=null "ไม่ใช้กับเอกสารชนิดนี้"
    core_labels: tuple[FieldSpec, ...] = ()  # ป้ายของฟิลด์แกนกลางที่ชนิดนี้ใช้แทนค่าตั้งต้น (key เดียวกัน = ทับ)
    status: str = ""  # สถานะของรายการฟิลด์ เช่น "draft: รอฝ่ายบัญชียืนยัน" (แสดงใน docs/ผล ไม่มีผลกับการตัดสิน)
    _squashed: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_squashed", tuple(squash(a) for a in self.aliases))
        for c in self.checks or ():
            if c not in KNOWN_CHECKS:
                raise ValueError(f"profile '{self.name}': check '{c}' ไม่รู้จัก (ใช้ได้: {', '.join(KNOWN_CHECKS)})")
        known = {DOC_PREFIX + f.key for f in self.fields}
        for r in self.required:
            if r.startswith(DOC_PREFIX) and r not in known:
                raise ValueError(f"profile '{self.name}': required '{r}' ไม่มีใน fields")
        if self.primary_amount and self.primary_amount.startswith(DOC_PREFIX) and self.primary_amount not in known:
            raise ValueError(f"profile '{self.name}': primary_amount '{self.primary_amount}' ไม่มีใน fields")

    def matches(self, squashed_text: str) -> bool:
        return any(a and a in squashed_text for a in self._squashed)

    def spec(self, key: str) -> FieldSpec | None:
        return next((f for f in self.fields if f.key == key), None)

    @property
    def can_auto_approve(self) -> bool:
        """โปรไฟล์ที่สถานะขึ้นต้นด้วย "draft" (รายการฟิลด์/required ยังไม่ผ่านการยืนยัน) อนุมัติอัตโนมัติไม่ได้ ต้องให้คนตรวจ."""
        return not self.status.strip().lower().startswith("draft")


GENERIC_PROFILE = Profile(name=GENERIC, title="ทั่วไป (ใบกำกับภาษี/ใบเสร็จ/invoice)")

# ฟิลด์แกนกลางที่อ่านจากป้ายได้ตอนนี้ (ป้ายอยู่ที่ config: core_labels ค่าตั้งต้น และ document_types.<ชนิด>.core_labels)
CORE_LABEL_KEYS = ("buyer.name",)


def _specs(items, *, core: bool = False) -> tuple[FieldSpec, ...]:
    specs = tuple(
        FieldSpec(key=f["key"], label=f.get("label", f["key"]), type="text" if core else f.get("type", "text"),
                  aliases=tuple(f.get("aliases", ())), source=f.get("source", ""),
                  nth=f.get("nth", 0), of=f.get("of"), next_line=bool(f.get("next_line", False)),
                  next_line_caption=bool(f.get("next_line_caption", False)))
        for f in items or ()
    )
    if core:
        for f in specs:
            if f.key not in CORE_LABEL_KEYS:
                raise ValueError(f"core_labels: key '{f.key}' ยังไม่รองรับ (รองรับ: {', '.join(CORE_LABEL_KEYS)})")
    return specs


def load_core_labels(items) -> tuple[FieldSpec, ...]:
    """ป้ายค่าตั้งต้นของฟิลด์แกนกลาง (config ระดับบน ``core_labels``)."""
    return _specs(items, core=True)


def core_labels_for(profile: Profile, defaults: tuple[FieldSpec, ...]) -> tuple[FieldSpec, ...]:
    """ป้ายของชนิดนี้: ของโปรไฟล์ทับค่าตั้งต้นที่ key เดียวกัน ที่เหลือใช้ค่าตั้งต้น."""
    own = {f.key for f in profile.core_labels}
    return (*profile.core_labels, *(f for f in defaults if f.key not in own))


def _profile(name: str, spec: dict[str, Any] | None) -> Profile:
    spec = spec or {}
    fields = _specs(spec.get("fields"))
    return Profile(
        name=name,
        title=spec.get("title", ""),
        aliases=tuple(spec.get("aliases", ())),
        required=tuple(spec["required"]) if "required" in spec else KEY_FIELDS,
        primary_amount=spec["primary_amount"] if "primary_amount" in spec else "total_amount",
        not_applicable=frozenset(spec.get("not_applicable", ())),
        fields=fields,
        checks=tuple(spec["checks"]) if "checks" in spec else None,
        core_labels=_specs(spec.get("core_labels"), core=True),
        status=spec.get("status", ""),
    )


def load_profiles(table: dict[str, Any] | None) -> list[Profile]:
    """แปลงตาราง ``document_types`` เป็นโปรไฟล์ (คงลำดับตามไฟล์)."""
    return [_profile(name, spec) for name, spec in (table or {}).items()]


def classify(profiles: list[Profile], text: str, document_type: str | None = None) -> Profile:
    """จำแนกโปรไฟล์จากข้อความทั้งหน้า (ป้ายที่ตรงก่อนชนะ) ถ้าไม่ตรงเลย ดูชื่อ ``document_type`` ที่สกัดได้ ไม่งั้นเป็น generic."""
    sq = squash(text)
    for p in profiles:
        if p.matches(sq):
            return p
    if document_type:
        sd = squash(document_type)
        for p in profiles:
            if squash(p.name) == sd or (p.title and squash(p.title) == sd):
                return p
    return GENERIC_PROFILE


def legacy_not_applicable(table: dict[str, Any] | None, document_type: str | None) -> set[str]:
    """รูปแบบเก่า: ``not_applicable`` ต่อชื่อชนิดเอกสารที่ตรงกับ ``document_type`` ตรง ๆ (เทียบแบบตัดช่องว่าง)."""
    if not document_type or not table:
        return set()
    sd = squash(document_type)
    for name, spec in table.items():
        if squash(name) == sd:
            return set((spec or {}).get("not_applicable", []))
    return set()
