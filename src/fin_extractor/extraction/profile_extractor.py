"""สกัดฟิลด์เฉพาะชนิดเอกสาร (``document_fields``) และหมายเหตุ (``remark``) ด้วย rules.

ฟิลด์ที่ได้ถูกกำหนดโดยโปรไฟล์ (doctypes.py) ไม่ใช่โค้ดนี้: หาป้ายที่โปรไฟล์ประกาศในบรรทัด แล้วอ่านค่าที่ตามหลังป้าย
ตามชนิดของฟิลด์ (money / date / tax_id / percent / code / text) ค่าที่ได้ผ่าน normalize ชุดเดียวกับ ``patterns.py`` เสมอ
ไม่เดา: ไม่เจอป้ายหรืออ่านค่าไม่ได้ = ไม่มีคีย์ (ไม่ใส่ค่าว่าง)

``remark`` = บรรทัดรูป "ป้าย: ค่า" ที่ไม่ใช่ป้ายของฟิลด์แกนกลางหรือฟิลด์ประจำชนิดใด ๆ และค่ายังไม่ซ้ำกับฟิลด์ที่สกัดไว้แล้ว
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from ..doctypes import FieldSpec, Profile, alias_regex, squash
from ..schema import ExtraField, InvoiceData
from .patterns import AMOUNT_RE, PERCENT_RE, find_tax_ids, normalize_tax_id, parse_amount, parse_date
from .rule_extractor import AMOUNT_FIELDS, AMOUNT_ONLY_RE, BUYER_RE, COMPANY_RE, DATE_LABEL_RE, INVOICE_NO_RE

_LEAD = re.compile(r"^[\s:：.\-/|]+")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/._]*")
# ป้ายยาวไม่เกิน 40 ตัวอักษร: วัดจากสแกน 18 ใบ (66 บรรทัดรูป "ป้าย: ค่า") ป้ายจริงยาวสุด 31 ตัวอักษร ส่วนที่ยาว 44 ขึ้นไป
# เป็นบรรทัดหลายคอลัมน์ที่ OCR ต่อกัน (ที่อยู่/เบอร์โทร/ประโยคท้ายฟอร์ม) ทั้งหมด ค่า 40 อยู่ในช่องว่างระหว่างสองกลุ่ม
# ผูกกับชุดข้อมูลนี้ ถ้าเจอเอกสารที่มีป้ายยาวกว่านี้ให้วัดใหม่
_KV = re.compile(r"^\s*([^:：|]{1,40}?)\s*[:：]\s*(.+?)\s*$")
# ป้ายของฟิลด์แกนกลางที่ rules อ่านอยู่แล้ว — บรรทัดที่ขึ้นต้นด้วยป้ายเหล่านี้ไม่ใช่หมายเหตุ
_FIELD_LABELS = [DATE_LABEL_RE, INVOICE_NO_RE, *(rx for _, rx in AMOUNT_FIELDS),
                 re.compile(r"เลขประจำตัว|tax\s*id|สำนักงานใหญ่|head\s*office|สาขา|ผู้ขาย|vendor|seller", re.I)]
# ป้ายบทบาท/ชื่อ: เป็นป้ายของช่อง vendor/buyer แต่ปรากฏเป็นค่าได้ (ชื่อบริษัท) จึงใช้ตรวจเฉพาะตัวป้าย ไม่ใช้ตรวจค่า
_NAME_LABELS = [BUYER_RE, COMPANY_RE, re.compile(r"^\s*(?:ชื่อ|name)\s*$", re.I)]


def normalize_value(kind: str, raw: Any) -> str | None:
    """ค่าดิบ → สตริงมาตรฐานตามชนิดฟิลด์ (None = อ่านค่าไม่ได้)."""
    if raw is None or str(raw).strip() == "":
        return None
    if kind == "money":
        amt = parse_amount(raw)
        return f"{amt.quantize(Decimal('0.01')):f}" if amt is not None else None
    if kind == "date":
        d = parse_date(str(raw))
        return d.isoformat() if d else None
    if kind == "tax_id":
        return normalize_tax_id(str(raw))
    if kind == "percent":
        m = PERCENT_RE.search(str(raw))
        num = Decimal(m.group(1)) if m else parse_amount(raw)
        if num is None:
            return None
        frac = num / 100 if (m or num > 1) else num  # "7%" / 7 → 0.07, 0.07 คงเดิม
        return f"{frac.normalize():f}"
    if kind == "code":
        # ก้อนแรกที่คั่นด้วยช่องว่าง ตัดเครื่องหมาย/อักษรไทยที่ OCR แทรกหัวท้าย (ไม่ตัดกลางก้อน) ต้องมีตัวเลข
        chunk = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", (str(raw).split() or [""])[0])
        return chunk if re.search(r"\d", chunk) else None
    return re.sub(r"\s+", " ", str(raw)).strip()


def _amounts(rest: str) -> list[str]:
    """ยอดเงินในข้อความ เรียงตามตำแหน่ง ("7.00 %" คืออัตรา ไม่ใช่ยอด)."""
    return [m.group() for m in AMOUNT_RE.finditer(rest) if not re.match(r"\s*%", rest[m.end():])]


def _first_amount(rest: str) -> str | None:
    found = _amounts(rest)
    return found[0] if found else None


def _read_value(spec: FieldSpec, lines: list[str], i: int, end: int, cut_at: list[re.Pattern]) -> str | None:
    rest = _LEAD.sub("", lines[i][end:])
    if spec.type == "money":
        if spec.of is not None:  # หลายยอดในบรรทัดเดียว: ต้องครบตามที่ป้ายระบุ ไม่ครบ = ไม่รู้ว่ายอดไหนเป็นยอดไหน ไม่ให้ค่า
            found = _amounts(rest)
            return normalize_value("money", found[spec.nth]) if len(found) == spec.of else None
        found = _amounts(rest)
        raw = found[spec.nth] if len(found) > spec.nth else None
        if raw is None and spec.nth == 0 and i + 1 < len(lines) and AMOUNT_ONLY_RE.match(lines[i + 1]):
            raw = _first_amount(lines[i + 1])
        return normalize_value("money", raw)
    if spec.type == "tax_id":
        ids = find_tax_ids(rest)
        return ids[0] if ids else None
    if spec.type == "text":
        cuts = [m.start() for rx in cut_at if (m := rx.search(rest))]
        return normalize_value("text", rest[: min(cuts)] if cuts else rest)
    return normalize_value(spec.type, rest)


def extract_document_fields(lines: list[str], profile: Profile) -> dict[str, str]:
    """ฟิลด์ที่โปรไฟล์ประกาศ — ป้ายที่เจอก่อนและอ่านค่าได้ชนะ (ลำดับ alias ในโปรไฟล์ = ลำดับความสำคัญ)."""
    out: dict[str, str] = {}
    all_alias = [alias_regex(a) for f in profile.fields for a in f.aliases]
    for spec in profile.fields:
        found = False
        for alias in spec.aliases:
            rx = alias_regex(alias)
            for i, line in enumerate(lines):
                for m in rx.finditer(line):
                    others = [r for r in all_alias if r.pattern != rx.pattern]
                    if (val := _read_value(spec, lines, i, m.end(), others)) is not None:
                        out[spec.key] = val
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return out


def _mentions_known_label(text: str, profile: Profile) -> bool:
    """ข้อความมีป้ายของฟิลด์แกนกลาง/ประจำชนิดปนอยู่ (ไม่นับป้ายที่เป็นข้อความทั้งก้อน เช่น "ชื่อ")."""
    sq = squash(text)
    return (any(rx.search(text) for rx in _FIELD_LABELS)
            or any(squash(a) in sq for f in profile.fields for a in f.aliases if a))


def _is_known_label(label: str, profile: Profile) -> bool:
    if any(rx.search(label) for rx in (*_FIELD_LABELS, *_NAME_LABELS)):
        return True
    sq = squash(label)
    return any(squash(a) in sq or sq in squash(a) for f in profile.fields for a in (f.label, *f.aliases) if a)


def _filled_values(data: InvoiceData) -> set[str]:
    """ค่าที่สกัดไว้แล้วทุกช่อง (เทียบแบบตัดช่องว่าง) ไว้กันหมายเหตุที่ซ้ำกับช่องที่มีอยู่."""
    vals: set[str] = set()

    def add(v: Any) -> None:
        if v not in (None, "", [], {}):
            vals.add(squash(v.isoformat() if hasattr(v, "isoformat") else f"{v:f}" if isinstance(v, Decimal) else str(v)))

    for name in ("document_type", "book_no", "invoice_no", "date", "subtotal", "discount",
                 "amount_before_vat", "vat_amount", "total_amount"):
        add(getattr(data, name))
    for party in (data.vendor, data.buyer):
        for sub in type(party).model_fields:
            add(getattr(party, sub))
    for v in data.document_fields.values():
        add(v)
    for it in data.items:
        add(it.description), add(it.amount)
    return vals


def _same_value(value: str, filled: set[str]) -> bool:
    sq = squash(value)
    if sq in filled:
        return True
    amt = parse_amount(value)  # "1,940.00" กับ "1940.00" คือค่าเดียวกัน
    return amt is not None and any(parse_amount(f) == amt for f in filled if re.fullmatch(r"[\d.,-]+", f))


def dedupe_remark(remark: list[ExtraField], data: InvoiceData, profile: Profile) -> list[ExtraField]:
    """ตัดหมายเหตุที่ซ้ำกับช่องที่มีอยู่: ป้ายเป็นป้ายของฟิลด์แกนกลาง/ประจำชนิด หรือค่าตรงกับค่าที่สกัดไว้แล้ว."""
    filled, seen, out = _filled_values(data), set(), []
    for e in remark:
        key = (squash(e.label), squash(e.value))
        if key in seen or _is_known_label(e.label, profile) or _same_value(e.value, filled):
            continue
        seen.add(key)
        out.append(e)
    return out


def extract_remark(lines: list[str], profile: Profile, data: InvoiceData) -> list[ExtraField]:
    """บรรทัดรูป "ป้าย: ค่า" ที่ไม่มีช่องรองรับ (rules) — LLM เติมหมายเหตุได้ละเอียดกว่า ใช้เป็นแหล่งสำรองเมื่อไม่มี LLM."""
    pairs = []
    for line in lines:
        m = _KV.match(line)
        if not m or not re.search(r"[^\W\d_]", m.group(1)):
            continue
        label, value = m.group(1).strip(), m.group(2).strip()
        # บรรทัดที่ค่ามีโคลอนอีกตัว หรือพูดถึงป้ายของช่องอื่น คือหลายคอลัมน์ที่ OCR ต่อเป็นบรรทัดเดียว แยกคู่ไม่ได้ ไม่เดา
        if re.search(r"[:：]", value) or _mentions_known_label(value, profile):
            continue
        pairs.append(ExtraField(label=label, value=value))
    return dedupe_remark(pairs, data, profile)
