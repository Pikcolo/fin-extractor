"""extraction/hybrid.py: rules นำ, LLM เติมเฉพาะช่องที่ยังไม่เสร็จ, ค่าต้องผ่านด่านก่อนรับ."""

from decimal import Decimal

import pytest

from fin_extractor.doctypes import GENERIC_PROFILE
from fin_extractor.extraction.hybrid import known_context, merge_llm, top_level, unresolved_keys
from fin_extractor.schema import InvoiceData
from fin_extractor.validation import run_checks

TEXT = "ใบกำกับภาษี\nบริษัท ตัวอย่าง จำกัด\nเลขประจำตัวผู้เสียภาษี 0105559117128\nรวม 100.00\nภาษีมูลค่าเพิ่ม 7.00\nรวมทั้งสิ้น 107.00\nอื่น ๆ 150.00"


def score(d: InvoiceData) -> int:
    checks = run_checks(d)
    return sum(c.passed is True for c in checks) - sum(c.passed is False for c in checks)


def rules_data(**kw) -> InvoiceData:
    return InvoiceData(amount_before_vat="100", vat_amount="7", **kw)


def merge(rules, llm, unresolved, asked=None):
    asked = asked or {top_level(k) for k in unresolved}
    return merge_llm(rules, llm, asked, unresolved, TEXT, GENERIC_PROFILE, score)


@pytest.mark.parametrize("key,top", [("vendor.name", "vendor"), ("items[2].amount", "items"),
                                     ("total_amount", "total_amount"), ("document_fields.total_paid", "document_fields")])
def test_top_level_maps_flat_keys_to_schema_properties(key, top):
    assert top_level(key) == top


def test_unresolved_are_missing_required_and_fields_of_failed_checks():
    d = InvoiceData(subtotal="100", vat_amount="7", total_amount="150")
    keys = unresolved_keys(d, run_checks(d), required=["vendor.name", "total_amount"], not_applicable=[], primary_amount="total_amount")
    assert "vendor.name" in keys                       # ต้องมีแต่ว่าง
    assert "total_amount" in keys                      # อยู่ใน check ที่ไม่ผ่าน
    assert unresolved_keys(d, [], required=["vendor.name"], not_applicable=["vendor.name"], primary_amount=None) == []


def test_known_context_excludes_unresolved_items_and_remark():
    d = rules_data(total_amount="107")
    assert known_context(d, {"total_amount"}) == {"amount_before_vat": "100.00", "vat_amount": "7.00"}


def test_grounded_value_for_a_missing_field_is_accepted_and_checks_improve():
    llm = InvoiceData(total_amount="107")
    merged, accepted, rejected, _ = merge(rules_data(), llm, ["total_amount"])
    assert merged.total_amount == Decimal("107.00") and accepted == {"total_amount"} and rejected == []


def test_value_not_in_the_text_is_rejected():
    llm = InvoiceData(vendor={"name": "Import Star Co., Ltd."})
    merged, accepted, rejected, _ = merge(rules_data(), llm, ["vendor.name"])
    assert merged.vendor.name is None and accepted == set()
    assert rejected == [("vendor.name", "ไม่พบในข้อความ")]


def test_tax_id_must_pass_checksum():
    bad = InvoiceData(vendor={"tax_id": "1234567890123"})
    text_with_bad = TEXT + "\n1234567890123"
    merged, accepted, rejected, _ = merge_llm(rules_data(), bad, {"vendor"}, ["vendor.tax_id"], text_with_bad, GENERIC_PROFILE, score)
    assert merged.vendor.tax_id is None and rejected == [("vendor.tax_id", "เลขภาษีไม่ผ่าน checksum")]
    good = InvoiceData(vendor={"tax_id": "0105559117128"})
    merged, accepted, _, _ = merge(rules_data(), good, ["vendor.tax_id"])
    assert merged.vendor.tax_id == "0105559117128" and accepted == {"vendor.tax_id"}


def test_amount_that_makes_checks_worse_is_dropped_even_if_it_appears_in_the_text():
    llm = InvoiceData(total_amount="150")   # 150.00 อยู่ในข้อความ แต่ 100 + 7 ≠ 150
    merged, accepted, rejected, _ = merge(rules_data(), llm, ["total_amount"])
    assert merged.total_amount is None and accepted == set()
    assert rejected == [("total_amount", "ทำให้ check รวมแย่ลง")]


def test_field_rules_already_have_is_never_overwritten_and_agreement_is_reported():
    rules = rules_data(total_amount="107")
    llm = InvoiceData(amount_before_vat="100", vat_amount="9", total_amount="107")
    merged, accepted, rejected, agree = merge(rules, llm, ["vendor.name"], asked={"amount_before_vat", "vat_amount", "total_amount", "vendor"})
    assert merged.vat_amount == Decimal("7.00") and accepted == set() and rejected == []
    assert agree["total_amount"] is True and agree["vat_amount"] is False


def test_remark_from_llm_is_ignored():
    llm = InvoiceData(remark=[{"label": "หมายเหตุ", "value": "อื่น ๆ"}])
    merged, accepted, _, _ = merge(rules_data(), llm, ["vendor.name"], asked={"remark", "vendor"})
    assert merged.remark == [] and accepted == set()


class _StubLLM:
    name = "llm:stub"

    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls = data, error, []

    def extract(self, text, profile=None, only=None, known=None, focus=None):
        self.calls.append((only, known))
        if self.error:
            raise self.error
        return self.data


def _pipe(llm):
    from fin_extractor.pipeline import FinanceOCRPipeline

    p = object.__new__(FinanceOCRPipeline)  # ไม่โหลด OCR/VLM
    p.cfg = {"validation": {"amount_tolerance": 0.02, "default_vat_rate": 0.07}}
    p.llm, p.strategy = llm, "hybrid"
    p.profiles, p.doc_table, p.core_labels = [], {}, ()
    p.last_rejected = []
    return p


LINES = ["บริษัท ตัวอย่าง จำกัด", "เลขประจำตัวผู้เสียภาษี 0105559117128", "รวม 100.00", "ภาษีมูลค่าเพิ่ม 7.00", "รวมทั้งสิ้น 107.00"]


def test_pipeline_hybrid_skips_the_llm_when_nothing_is_unresolved():
    # โปรไฟล์ที่ไม่บังคับช่องใดและ check ที่มีผ่านหมด → ไม่มีช่องค้าง → ไม่เรียก LLM
    from fin_extractor.doctypes import Profile

    llm = _StubLLM(error=AssertionError("ไม่ควรถูกเรียก"))
    pipe = _pipe(llm)
    pipe.profiles = [Profile(name="any", aliases=("ใบกำกับภาษี",), required=(), primary_amount=None)]
    lines = ["ใบกำกับภาษี", *LINES]
    data, sources, agree = pipe.extract_traced(lines, "\n".join(lines))
    assert pipe.profile_for(data).name == "any"
    assert llm.calls == [] and set(sources.values()) == {"rules"} and agree == {}


def test_pipeline_hybrid_asks_only_unresolved_properties_and_marks_sources():
    pipe = _pipe(_StubLLM(InvoiceData(invoice_no="IV-001")))
    lines = [*LINES, "เลขที่ IV-001"]
    data, sources, _ = pipe.extract_traced(lines, "\n".join(lines))
    (only, known), = pipe.llm.calls
    assert only and "total_amount" not in only and set(only) <= {"vendor", "buyer", "invoice_no", "date", "document_type"}
    assert all(not k.startswith("items[") for k in known)
    assert pipe.last_rejected == [] or all(isinstance(r, tuple) for r in pipe.last_rejected)


def test_pipeline_hybrid_falls_back_to_rules_when_llm_fails():
    pipe = _pipe(_StubLLM(error=ValueError("bad json")))
    data, sources, agree = pipe.extract_traced(LINES, "\n".join(LINES))
    assert set(sources.values()) == {"rules"} and agree == {}
