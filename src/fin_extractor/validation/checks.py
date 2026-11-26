"""Cross-check ข้อมูลที่สกัดได้ — ตัวเลขทางการเงินต้องสอดคล้องกันเอง.

ถ้า OCR อ่านเลขผิดหลักเดียว ผลรวมจะไม่ลงตัว → ใช้จับข้อผิดพลาดได้โดยไม่ต้องมี ground truth
แต่ละ check บันทึก ``fields`` = field ที่ใช้คำนวณจริง เพื่อให้ระดับหลักฐานรายฟิลด์อ้างอิงได้
(ถ้า check ไม่ผ่าน จะไม่รู้ว่าช่องไหนผิด จึงรวมทุกช่องในสูตร)
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from collections.abc import Collection

from ..schema import InvoiceData, ValidationCheck
from .tax_id import is_valid_thai_tax_id


def _close(a: Decimal, b: Decimal, tol: Decimal) -> bool:
    return abs(a - b) <= tol


def _dec(v: float | str | Decimal) -> Decimal:
    return v if isinstance(v, Decimal) else Decimal(str(v))


NOT_APPLICABLE_DETAIL = "ไม่ใช้กับเอกสารชนิดนี้"


def run_checks(
    d: InvoiceData, tol: float | str | Decimal = "0.02", default_vat_rate: float | str | Decimal = "0.07",
    applicable: Collection[str] | None = None,
) -> list[ValidationCheck]:
    """``applicable`` = ชื่อ check ที่ใช้กับชนิดเอกสารนี้ (None = ทุกข้อ) ข้อที่ไม่ใช้ยังอยู่ในผลแต่ ``passed=None``
    (ไม่นับเป็นผ่านหรือไม่ผ่าน ไม่ยืนยันฟิลด์ใด) เพื่อให้ผลของทุกชนิดมีชื่อ check ชุดเดียวกัน"""
    checks = _run_all_checks(d, tol, default_vat_rate)
    if applicable is None:
        return checks
    return [c if c.name in applicable else ValidationCheck(name=c.name, passed=None, detail=NOT_APPLICABLE_DETAIL)
            for c in checks]


def _run_all_checks(
    d: InvoiceData, tol: float | str | Decimal, default_vat_rate: float | str | Decimal
) -> list[ValidationCheck]:
    tol, default_vat_rate = _dec(tol), _dec(default_vat_rate)
    checks: list[ValidationCheck] = []

    def add(name: str, passed: bool | None, detail: str = "", fields: list[str] | None = None) -> None:
        checks.append(ValidationCheck(name=name, passed=passed, detail=detail, fields=fields or []))

    # 1) เลขผู้เสียภาษี
    add("vendor_tax_id_checksum",
        is_valid_thai_tax_id(d.vendor.tax_id) if d.vendor.tax_id else None,
        d.vendor.tax_id or "ไม่พบ",
        ["vendor.tax_id"] if d.vendor.tax_id else [])
    if d.buyer.tax_id:
        add("buyer_tax_id_checksum", is_valid_thai_tax_id(d.buyer.tax_id), d.buyer.tax_id, ["buyer.tax_id"])

    # 2) ผลรวมรายการ = ยอดใดยอดหนึ่ง: subtotal / ยอดก่อน VAT (ราคาไม่รวมภาษี) หรือยอดรวม (ร้านที่พิมพ์ราคารายการรวม VAT แล้ว เช่น 1308 1312)
    #    ไม่มียอดก่อน VAT ที่อ่านได้: ยอดก่อน VAT = ยอดรวม − VAT (สูตรเดียวกับข้อ 4 · 1318: 8,025 − 525 = 7,500 = รายการ)
    item_fields = [f"items[{i}].amount" for i, it in enumerate(d.items) if it.amount is not None]
    item_amounts = [it.amount for it in d.items if it.amount is not None]
    targets = [([n], n, v) for n, v in (("subtotal", d.subtotal), ("amount_before_vat", d.amount_before_vat),
                                        ("total_amount", d.total_amount)) if v is not None]
    if d.amount_before_vat is None and None not in (d.vat_amount, d.total_amount):
        targets.append((["total_amount", "vat_amount"], "total_amount − vat_amount", d.total_amount - d.vat_amount))
    if item_amounts and targets:
        s = sum(item_amounts)
        matched = [f for fs, _n, v in targets if _close(s, v, tol) for f in fs]
        add("items_sum", bool(matched), f"sum(items)={s} vs {' / '.join(f'{n} {v}' for _fs, n, v in targets)}",
            [*item_fields, *dict.fromkeys(matched or [f for fs, _n, _v in targets for f in fs])])
    else:
        add("items_sum", None, "ไม่มีรายการหรือยอดรวม")

    # 3) ส่วนลด: subtotal - discount = ยอดก่อน VAT (ราคาไม่รวมภาษี) หรือ = ยอดรวม (ราคารวมภาษี)
    if d.subtotal is not None and d.discount is not None:
        net = d.subtotal - d.discount
        candidates = [(n, v) for n, v in (("amount_before_vat", d.amount_before_vat),
                                          ("total_amount", d.total_amount)) if v is not None]
        matched = [n for n, v in candidates if _close(net, v, tol)]
        add("discount", bool(matched) if candidates else None,
            f"{d.subtotal} - {d.discount} = {net} vs {[str(v) for _, v in candidates]}",
            ["subtotal", "discount", *(matched or [n for n, _ in candidates])] if candidates else [])

    # 4) ยอดก่อน VAT + VAT = ยอดรวม · ไม่มียอดก่อน VAT ใช้ subtotal (− ส่วนลด) แทน (1308: รวมเงิน 23,140.19 + VAT 1,619.81 = 24,760.00)
    #    ยกเว้น subtotal (− ส่วนลด) = ยอดรวม: ราคารวม VAT แล้ว subtotal ไม่ใช่ฐานก่อน VAT ไม่ตรวจ (ไม่ให้ล้มผิด)
    base_name, base = ("amount_before_vat", d.amount_before_vat) if d.amount_before_vat is not None else (None, None)
    if base is None and d.subtotal is not None:
        net = d.subtotal - (d.discount or 0)
        if d.total_amount is None or not _close(net, d.total_amount, tol):
            base_name, base = "subtotal", net
    if None not in (base, d.vat_amount, d.total_amount):
        s = base + d.vat_amount
        used = [base_name, *(["discount"] if base_name == "subtotal" and d.discount else [])]
        add("before_vat_plus_vat", _close(s, d.total_amount, tol),
            f"{base} + {d.vat_amount} = {s} vs {d.total_amount}", [*used, "vat_amount", "total_amount"])
    else:
        add("before_vat_plus_vat", None, "ข้อมูลไม่ครบ")

    # 5) VAT = ยอดก่อน VAT x อัตรา (หรือ total x rate/(1+rate) ถ้าไม่มียอดก่อน VAT)
    if d.vat_amount is not None:
        rate = d.vat_rate or default_vat_rate
        base_name = base = None
        if d.amount_before_vat is not None:
            base_name, base, expected = "amount_before_vat", d.amount_before_vat, d.amount_before_vat * rate
        elif d.total_amount is not None:
            base_name, base, expected = "total_amount", d.total_amount, d.total_amount * rate / (1 + rate)
        else:
            expected = None
        if expected is not None:
            expected = expected.quantize(Decimal("0.01"))
        # vat_rate เป็น input ของสูตรเฉพาะเมื่อเอกสารระบุเอง ถ้าใช้ค่าตั้งต้นของระบบก็ไม่ได้ยืนยันค่าที่อ่านมา
        used = ["vat_amount", *([base_name] if base_name else []), *(["vat_rate"] if d.vat_rate else [])]
        add("vat_rate", _close(d.vat_amount, expected, tol) if expected is not None else None,
            f"expected {expected} at {rate:.0%}, got {d.vat_amount}", used if expected is not None else [])

    # 6) วันที่สมเหตุสมผล
    if d.date:
        add("date_range", date(2000, 1, 1) <= d.date <= date.today(), d.date.isoformat(), ["date"])
    else:
        add("date_range", None, "ไม่พบวันที่")

    # items_sum ไม่ผ่าน: ยอดที่ check อื่นยืนยันแล้วไม่ใช่ตัวผิด ชี้เฉพาะรายการ + ยอดที่ยังไม่มีอะไรยืนยัน
    # (1309: รวมเงิน + VAT = ยอดรวม ผ่าน แต่รายการรวม 3,860 ไม่ตรง → รายการหายหรืออ่านผิด ไม่ใช่ยอดรวม)
    items_check = next(c for c in checks if c.name == "items_sum")
    if items_check.passed is False:
        confirmed = {f for c in checks if c.passed for f in c.fields}
        items_check.fields = [f for f in items_check.fields if f.startswith("items[") or f not in confirmed]

    return checks
