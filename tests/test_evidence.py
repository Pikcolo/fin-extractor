"""ระดับหลักฐานรายฟิลด์ — ทดสอบทุกกิ่งของกฎ missing → conflict → verified → unverified."""

from datetime import date
from decimal import Decimal

from test_extraction_validation import BITKUB_LINES

from fin_extractor.extraction import extract_fields
from fin_extractor.schema import ExtraField, ExtractionResult, InvoiceData, LineItem
from fin_extractor.validation import run_checks
from fin_extractor.validation.evidence import build_field_evidence, comparable, flatten

TEXT = "\n".join(BITKUB_LINES)


def evidence(d: InvoiceData, text: str = TEXT, **kw):
    return build_field_evidence(d, run_checks(d), text, **kw)


def test_amounts_in_passing_formulas_are_verified():
    ev = evidence(extract_fields(BITKUB_LINES))
    total = ev["total_amount"]
    assert total.level == "verified"
    assert "before_vat_plus_vat" in total.checks_passed and total.grounded is True
    assert ev["vendor.tax_id"].level == "verified" and ev["vendor.tax_id"].format_valid is True


def test_fields_without_a_corroborating_check_are_never_verified():
    ev = evidence(extract_fields(BITKUB_LINES))
    for key in ("vendor.name", "date"):
        assert ev[key].level == "unverified" and not ev[key].checks_passed
    assert ev["date"].checks_passed == []  # date_range ผ่านแต่ไม่ยืนยันว่าอ่านถูกวัน


def test_missing_key_field():
    ev = evidence(extract_fields(BITKUB_LINES))  # ช่องเลขที่เอกสารว่าง
    assert ev["invoice_no"].level == "missing" and ev["invoice_no"].value is None


def test_failed_check_marks_every_field_in_the_formula_as_conflict():
    d = extract_fields(BITKUB_LINES)
    d.amount_before_vat, d.vat_amount = Decimal("44.46"), Decimal("3.17")
    ev = evidence(d)
    for key in ("amount_before_vat", "vat_amount", "total_amount"):
        assert ev[key].level == "conflict" and "before_vat_plus_vat" in ev[key].checks_failed


def test_bad_tax_id_checksum_is_conflict():
    d = InvoiceData(vendor={"tax_id": "0105559117129"})
    ev = evidence(d, text="Tax ID 0105559117129")
    assert ev["vendor.tax_id"].format_valid is False and ev["vendor.tax_id"].level == "conflict"


def test_value_not_in_text_is_conflict_even_when_formula_passes():
    d = extract_fields(BITKUB_LINES)
    text_without_total = TEXT.replace("47.56", "")
    ev = evidence(d, text=text_without_total)
    assert ev["total_amount"].grounded is False and ev["total_amount"].level == "conflict"


def test_sources_disagree_is_conflict_and_agreement_is_reported():
    d = extract_fields(BITKUB_LINES)
    ev = evidence(d, agree={"total_amount": False, "vat_amount": True}, sources={"total_amount": "llm"})
    assert ev["total_amount"].level == "conflict" and ev["total_amount"].source == "llm"
    assert ev["vat_amount"].sources_agree is True and ev["vat_amount"].level == "verified"
    assert ev["subtotal"].sources_agree is None


def test_items_sum_is_attributed_to_the_total_when_there_is_no_subtotal():
    d = InvoiceData(items=[LineItem(description="a", amount=Decimal("10.00")),
                           LineItem(description="b", amount=Decimal("5.00"))],
                    total_amount=Decimal("15.00"))
    check = next(c for c in run_checks(d) if c.name == "items_sum")
    assert check.passed is True
    assert check.fields == ["items[0].amount", "items[1].amount", "total_amount"]
    ev = evidence(d, text="a 10.00\nb 5.00\nTotal 15.00")
    assert ev["total_amount"].level == "verified" and ev["items[1].amount"].level == "verified"
    assert ev["items[0].description"].level == "unverified"


def test_vat_rate_check_only_counts_vat_rate_field_when_document_states_it():
    d = InvoiceData(amount_before_vat=Decimal("100.00"), vat_amount=Decimal("7.00"), total_amount=Decimal("107.00"))
    check = next(c for c in run_checks(d) if c.name == "vat_rate")
    assert "vat_rate" not in check.fields  # ใช้ค่าตั้งต้นของระบบ ไม่ได้ยืนยันค่าที่อ่านมา
    d.vat_rate = Decimal("0.07")
    check = next(c for c in run_checks(d) if c.name == "vat_rate")
    assert check.passed is True and "vat_rate" in check.fields


def test_date_out_of_range_is_conflict():
    d = InvoiceData(date=date(1999, 1, 1))
    ev = evidence(d, text="Date 1999-01-01")
    assert ev["date"].level == "conflict" and ev["date"].checks_failed == ["date_range"]


def test_handwritten_flag_is_null_unless_listed():
    d = extract_fields(BITKUB_LINES)
    assert evidence(d)["total_amount"].handwritten is None
    assert evidence(d, handwritten=["total_amount"])["total_amount"].handwritten is True


def test_flatten_and_comparable():
    d = InvoiceData(vendor={"name": "A B"}, items=[LineItem(description="x", amount=Decimal("1.5"))])
    assert flatten(d) == {"vendor.name": "A B", "items[0].description": "x", "items[0].amount": Decimal("1.50")}
    assert comparable("A B") == comparable(" A  B ") and comparable(Decimal("1.5")) == Decimal("1.50")


def test_money_is_decimal_quantized_and_serialized_as_string():
    d = InvoiceData(total_amount=0.1 + 0.2, vat_rate=0.07, subtotal="47.5")
    assert d.total_amount == Decimal("0.30") and d.subtotal == Decimal("47.50") and d.vat_rate == Decimal("0.07")
    r = ExtractionResult(source_file="x", engine="e", data=d, status="needs_review")
    j = r.model_dump_json()
    assert '"total_amount":"0.30"' in j and '"schema_version":"4.0"' in j
    assert all(k not in j for k in ('"raw_text"', '"ocr_confidence"', '"rejected"', '"missing_fields"', '"field_evidence"'))  # ช่องว่าง/ช่องซ้ำไม่ถูกเขียน


def test_review_orders_fields_by_what_to_check_first_and_explains_why():
    from fin_extractor.schema import ValidationCheck
    from fin_extractor.validation.evidence import review_from_evidence

    d = InvoiceData(vendor={"name": "บริษัท ก", "tax_id": "1234567890123"}, total_amount="107.00", amount_before_vat="100.00", vat_amount="7.00")
    text = "บริษัท ก 1234567890123 100.00 7.00 107.00"
    checks = run_checks(d)
    rv = review_from_evidence(build_field_evidence(d, checks, text), checks, "total_amount")
    levels = [r.level for r in rv.values()]
    assert levels == sorted(levels, key=["conflict", "missing", "unverified", "verified", "not_applicable"].index)  # สิ่งที่ต้องดูก่อนอยู่บนสุด
    assert rv["vendor.tax_id"].level == "conflict" and any("checksum" in x for x in rv["vendor.tax_id"].reasons)
    assert rv["total_amount"].level == "verified" and any("before_vat_plus_vat" in x for x in rv["total_amount"].reasons)
    assert rv["vendor.name"].level == "unverified" and rv["vendor.name"].source is None  # rules = ไม่มีคีย์ source
    # ค่าที่คนแก้ ไม่อยู่ในข้อความ OCR ไม่นับเป็นขัดแย้ง
    d2 = d.model_copy(update={"invoice_no": "INV-9"})
    ev2 = build_field_evidence(d2, run_checks(d2), text, sources={"invoice_no": "human"})
    r2 = review_from_evidence(ev2, run_checks(d2), sources={"invoice_no": "human"})["invoice_no"]
    assert r2.level == "unverified" and r2.source == "human" and r2.reasons == ["แก้โดยคน"]
    assert ValidationCheck  # import ใช้ได้


class _StubLLM:
    def __init__(self, data=None, error=None):
        self.data, self.error = data, error

    def extract(self, text, profile=None):
        if self.error:
            raise self.error
        return self.data


def _pipe(llm):
    from fin_extractor.pipeline import FinanceOCRPipeline

    p = object.__new__(FinanceOCRPipeline)  # ไม่โหลดตัวอ่าน — ทดสอบเฉพาะตรรกะรวมผล
    p.cfg = {"validation": {"amount_tolerance": 0.02, "default_vat_rate": 0.07}}
    p.llm = llm
    p.profiles, p.core_labels = [], ()
    return p


def test_extract_traced_records_source_and_agreement():
    rules = extract_fields(BITKUB_LINES)
    llm = rules.model_copy(deep=True)
    llm.total_amount = Decimal("47.65")          # LLM อ่านต่างจาก rules
    llm.vendor.address = "99 ถนนสุขุมวิท"          # มีเฉพาะฝั่ง LLM
    llm.items = []
    data, sources, agree = _pipe(_StubLLM(llm)).extract_traced(BITKUB_LINES, TEXT)
    assert data.total_amount == Decimal("47.56") and sources["total_amount"] == "rules"  # rules ผ่าน check มากกว่า
    assert agree["total_amount"] is False and agree["vat_amount"] is True
    assert sources["vendor.address"] == "llm" and "vendor.address" not in agree
    assert sources["items[0].amount"] == "rules"


def test_extract_traced_without_llm_or_when_llm_fails():
    for llm in (None, _StubLLM(error=ValueError("bad json"))):
        data, sources, agree = _pipe(llm).extract_traced(BITKUB_LINES, TEXT)
        assert set(sources.values()) == {"rules"} and agree == {}


# ---------- remark และ not_applicable ----------

def test_llm_normalize_accepts_list_or_dict_and_drops_empty_values():
    from fin_extractor.extraction.llm_extractor import normalize_llm_output

    as_list = normalize_llm_output({"remark": [{"label": "หัก ณ ที่จ่าย 3%", "value": 60.0},
                                                {"label": "PO", "value": ""}, "ขยะ"]})
    as_dict = normalize_llm_output({"remark": {"หัก ณ ที่จ่าย 3%": "60.0", "PO": None}})
    assert as_list["remark"] == as_dict["remark"] == [{"label": "หัก ณ ที่จ่าย 3%", "value": "60.0"}]
    assert normalize_llm_output({})["remark"] == []
    echoed = normalize_llm_output({"remark": [{"label": "ผู้จัดการ", "value": " ผู้จัดการ "}]})
    assert echoed["remark"] == []


def test_remark_gets_evidence_without_check_and_is_never_verified():
    d = InvoiceData(remark=[{"label": "เลขที่ใบสั่งซื้อ", "value": "PO-123"},
                                  {"label": "เงื่อนไข", "value": "โอนเงิน"}])
    ev = evidence(d, text="เลขที่ใบสั่งซื้อ PO-123")
    assert ev["remark[0]"].level == "unverified" and ev["remark[0]"].grounded is True
    assert ev["remark[1]"].level == "conflict" and ev["remark[1]"].grounded is False  # ไม่ปรากฏในข้อความ


def test_not_applicable_is_distinct_from_missing():
    from fin_extractor.validation.scoring import decide_status

    na = {"invoice_no", "total_amount"}  # โปรไฟล์ชนิดเอกสารที่ไม่มีสองช่องนี้ (Profile.not_applicable)

    d = InvoiceData(vendor={"name": "A", "tax_id": "0105559117128"}, date=date(2023, 5, 18))
    ev = evidence(d, not_applicable=na)
    assert ev["invoice_no"].level == "not_applicable" and ev["total_amount"].level == "not_applicable"
    assert evidence(d)["invoice_no"].level == "missing"

    missing, status = decide_status(d, run_checks(d), not_applicable=na)
    assert missing == [] and status != "failed"
    assert decide_status(d, run_checks(d))[1] == "failed"  # ไม่ประกาศว่าไม่มี → ยังเป็น failed เหมือนเดิม


def test_extract_traced_treats_remark_like_items():
    rules = extract_fields(BITKUB_LINES)
    llm = rules.model_copy(deep=True)
    llm.remark = [ExtraField(label="เงื่อนไข", value="โอนเงิน")]
    data, sources, agree = _pipe(_StubLLM(llm)).extract_traced(BITKUB_LINES, TEXT)
    assert sources["remark[0]"] == "llm" and "remark[0]" not in agree


def test_vat_inclusive_items_match_the_total_and_subtotal_stands_in_for_before_vat():
    """1308: รายการพิมพ์ราคารวม VAT (ผลรวม = ยอดรวม ไม่ใช่ subtotal) และไม่มียอดก่อน VAT แยก: subtotal + VAT = ยอดรวม."""
    d = InvoiceData(items=[LineItem(description="a", amount=Decimal("24760.00"))], subtotal=Decimal("23140.19"),
                    vat_amount=Decimal("1619.81"), total_amount=Decimal("24760.00"))
    by = {c.name: c for c in run_checks(d)}
    assert by["items_sum"].passed is True and by["items_sum"].fields == ["items[0].amount", "total_amount"]
    assert by["before_vat_plus_vat"].passed is True
    assert by["before_vat_plus_vat"].fields == ["subtotal", "vat_amount", "total_amount"]


def test_subtotal_equal_to_the_total_is_not_used_as_the_base_before_vat():
    """ราคารวม VAT แล้ว: subtotal = ยอดรวม ไม่ใช่ฐานก่อน VAT ไม่ตรวจ (ไม่ใช่ล้ม)."""
    d = InvoiceData(subtotal=Decimal("107.00"), vat_amount=Decimal("7.00"), total_amount=Decimal("107.00"))
    assert next(c for c in run_checks(d) if c.name == "before_vat_plus_vat").passed is None


def test_items_matching_no_total_fail_and_name_every_total():
    d = InvoiceData(items=[LineItem(description="a", amount=Decimal("1.00"))], subtotal=Decimal("10.00"),
                    total_amount=Decimal("10.70"))
    check = next(c for c in run_checks(d) if c.name == "items_sum")
    assert check.passed is False and check.fields == ["items[0].amount", "subtotal", "total_amount"]


def test_failed_items_sum_blames_only_the_items_when_other_checks_confirm_the_totals():
    """1309: รวมเงิน + VAT = ยอดรวม (ยืนยันกันเอง) แต่รายการรวมไม่ตรง → รายการผิด ยอดยัง verified."""
    d = InvoiceData(items=[LineItem(description="a", amount=Decimal("3860.00"))], subtotal=Decimal("3714.95"),
                    vat_amount=Decimal("260.05"), total_amount=Decimal("3975.00"))
    check = next(c for c in run_checks(d) if c.name == "items_sum")
    assert check.passed is False and check.fields == ["items[0].amount"]
    ev = evidence(d, text="a 3860.00\nรวมเงิน 3714.95\nVAT 260.05\nยอดสุทธิ 3975.00")
    assert ev["total_amount"].level == "verified" and ev["subtotal"].level == "verified"
    assert ev["items[0].amount"].level == "conflict"


def test_items_match_the_total_less_vat_when_no_amount_before_vat_was_read():
    """1318: ไม่ได้ยอดก่อน VAT แต่ยอดรวม − VAT = ผลรวมรายการ."""
    d = InvoiceData(items=[LineItem(description="a", amount=Decimal("7500.00"))], vat_amount=Decimal("525.00"),
                    total_amount=Decimal("8025.00"))
    check = next(c for c in run_checks(d) if c.name == "items_sum")
    assert check.passed is True and check.fields == ["items[0].amount", "total_amount", "vat_amount"]
