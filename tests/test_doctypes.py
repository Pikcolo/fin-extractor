"""โปรไฟล์ชนิดเอกสาร: จำแนกชนิด, ฟิลด์ประจำชนิด, remark, และสถานะที่ขึ้นกับโปรไฟล์.

ข้อความตัวอย่างย่อมาจากสแกนจริงใน data/raw/finance_doc_scan (ไม่ใส่ชื่อ/เลขจริง ใช้ค่าสมมติรูปแบบเดียวกับที่ OCR อ่านได้)
"""

import pytest
from test_extraction_validation import BITKUB_LINES

from fin_extractor.config import load_config
from fin_extractor.doctypes import GENERIC_PROFILE, FieldSpec, Profile, classify, load_profiles
from fin_extractor.extraction import extract_fields
from fin_extractor.extraction.llm_extractor import normalize_llm_output, profile_schema
from fin_extractor.extraction.profile_extractor import (
    dedupe_remark, extract_document_fields, extract_remark, normalize_value)
from fin_extractor.schema import ExtraField, InvoiceData
from fin_extractor.validation import decide_status, run_checks
from fin_extractor.validation.evidence import build_field_evidence, flatten

PROFILES = load_profiles(load_config()["document_types"])
BY_NAME = {p.name: p for p in PROFILES}

WITHHOLDING = [
    "หนังสือรับรองการหักภาษี ณ ที่จ่าย เล่มที่ เลขที่ WT2099000001",
    "ตามมาตรา 50 ทวิแห่งประมวลรัษฎากร",
    "รวมเงินที่จ่ายและภาษีที่หักนำส่ง 2,000.00 60.00",
    "รวมเงินภาษีที่หักนำส่ง (ตัวอักษร) (หกสิบบาทถ้วน)",
]
CUSTOMS_EN = [
    "Receipt No. 4",
    "Special Entry Number: B999X990000001",
    "Duty 0.00 % 0.00",
    "VAT 7.00 % 123.45",
    "Total Charges 123.45",
]
CUSTOMS_TH = [
    "ใบเสร็จรับเงิน กรมศุลกากร",
    "เลขที่ใบขนสินค้า B999-X990000001 (1194)",
    "วันที่นำเข้า/ส่งออก 05-11-2569",
    "รวมเงินทั้งสิ้น (บาท) 123.45",
]


# ---------- จำแนกชนิด ----------

@pytest.mark.parametrize("lines,doc_type,expected", [
    (WITHHOLDING, None, "withholding_certificate"),
    (["หนังสือรับรองการหักภาษี ณ ที่จ่าย"], "receipt", "withholding_certificate"),  # ป้ายบนเอกสารชนะ document_type ที่ rules เดา
    (["ตามมาตรา  50  ทวิ แห่งประมวลรัษฎากร"], None, "withholding_certificate"),  # OCR แทรกช่องว่าง
    (CUSTOMS_EN, "receipt", "customs_receipt"),
    (CUSTOMS_TH, "receipt", "customs_receipt"),  # มีคำว่า "ใบเสร็จรับเงิน" แต่ป้ายศุลกากรมีลำดับสำคัญกว่า
    (["ใบสำคัญรับเงิน", "เลขที่ 03-202610001"], None, "payment_voucher"),
    (BITKUB_LINES, "tax_invoice", "generic"),
    (["Tax Invoice"], "tax_invoice", "generic"),
    (["อะไรก็ไม่รู้"], None, "generic"),
])
def test_classify_by_printed_title(lines, doc_type, expected):
    assert classify(PROFILES, "\n".join(lines), doc_type).name == expected


def test_classify_falls_back_to_document_type_name_then_generic():
    table = {"my_type": {"title": "ชนิดของฉัน"}}
    profs = load_profiles(table)
    assert classify(profs, "ไม่มีป้าย", "my_type").name == "my_type"
    assert classify(profs, "ไม่มีป้าย", "ชนิด ของฉัน").name == "my_type"  # เทียบชื่อแบบตัดช่องว่าง
    assert classify(profs, "ไม่มีป้าย", "other") is GENERIC_PROFILE


def test_generic_profile_is_exactly_the_original_invoice_rules():
    assert GENERIC_PROFILE.required == ("vendor.name", "vendor.tax_id", "invoice_no", "date", "total_amount")
    assert GENERIC_PROFILE.primary_amount == "total_amount" and not GENERIC_PROFILE.fields


def test_profile_rejects_unknown_field_type_and_dangling_required():
    with pytest.raises(ValueError, match="type"):
        FieldSpec(key="x", label="x", type="float")
    with pytest.raises(ValueError, match="required"):
        Profile(name="p", required=("document_fields.nope",))
    with pytest.raises(ValueError, match="primary_amount"):
        Profile(name="p", primary_amount="document_fields.nope")


# ---------- ฟิลด์ประจำชนิด (rules) ----------

def test_withholding_fields_are_read_by_label():
    got = extract_document_fields(WITHHOLDING, BY_NAME["withholding_certificate"])
    assert got["total_paid"] == "2000.00"
    assert got["certificate_no"] == "WT2099000001"
    assert got["total_tax_withheld_text"] == "(หกสิบบาทถ้วน)"
    assert got["total_tax_withheld"] == "60.00"  # สองยอดอยู่บรรทัดเดียวกัน ซ้าย = ยอดจ่าย ขวา = ภาษีที่หัก


@pytest.mark.parametrize("line,paid,tax", [
    ("รวมเงินที่จ่ายและภาษีที่หักน้ำส่ง 233,644.86 7,009.35", "233644.86", "7009.35"),  # OCR สะกดส่วนท้ายป้ายเพี้ยน
    ("รวมเงินที่จ่ายและภาษีที่หักน่ำส่ง 106,25000 3,187.50", None, None),   # ยอดแรกขาดจุดทศนิยม: ไม่หยิบ 3,187.50 ไปเป็นยอดจ่าย
    ("รวมเงินที่จ่ายและภาษีที่หักนำส่ง 2,000.00", None, None),              # ยอดเดียว: ไม่รู้ว่าเป็นยอดจ่ายหรือภาษี
])
def test_two_amounts_on_one_line_are_never_swapped(line, paid, tax):
    got = extract_document_fields([line], BY_NAME["withholding_certificate"])
    assert got.get("total_paid") == paid and got.get("total_tax_withheld") == tax


def test_field_spec_rejects_nth_outside_of():
    with pytest.raises(ValueError, match="nth"):
        FieldSpec(key="x", label="x", type="money", nth=2, of=2)


def test_customs_fields_skip_rates_and_read_the_amount():
    got = extract_document_fields(CUSTOMS_EN, BY_NAME["customs_receipt"])
    assert got == {"entry_no": "B999X990000001", "receipt_no": "4", "duty": "0.00", "total_charges": "123.45"}
    th = extract_document_fields(CUSTOMS_TH, BY_NAME["customs_receipt"])
    assert th["entry_no"] == "B999-X990000001" and th["import_date"] == "2026-11-05" and th["total_charges"] == "123.45"


def test_document_fields_absent_label_means_absent_key():
    assert extract_document_fields(["ไม่มีป้ายใด ๆ"], BY_NAME["customs_receipt"]) == {}
    assert extract_document_fields(BITKUB_LINES, GENERIC_PROFILE) == {}


@pytest.mark.parametrize("kind,raw,expected", [
    ("money", "1,940.5", "1940.50"), ("money", "ไม่มี", None),
    ("date", "05-11-2569", "2026-11-05"), ("date", "xx", None),
    ("tax_id", "0-1055-59117-12-8", "0105559117128"), ("tax_id", "123", None),
    ("percent", "7%", "0.07"), ("percent", 7, "0.07"), ("percent", "0.07", "0.07"),
    ("code", "  WT-001 ", "WT-001"), ("code", "ตรอก", None),
    ("code", "ี12345678901 x", "12345678901"),            # อักษรไทยที่ OCR แทรกหน้าตัวเลขถูกตัด
    ("code", "9Z]999G90000000001", "9Z]999G90000000001"),  # ไม่ตัดกลางก้อน (เดิมได้ "1Z" ซึ่งเข้าใจผิด)
    ("text", "  a   b ", "a b"), ("text", "", None),
])
def test_normalize_value_by_type(kind, raw, expected):
    assert normalize_value(kind, raw) == expected


# ---------- remark ----------

def test_remark_keeps_unmodelled_label_value_lines_and_drops_known_ones():
    lines = ["เลขที่ใบสั่งซื้อ: PO-123", "วันที่: 18/05/2568", "เงื่อนไขการชำระเงิน : โอนเงิน",
             "Special Entry Number: B999X990000001", "ไม่มีโคลอน", "ชื่อ: บริษัท ก จำกัด",
             "Weight: 1.20 Kgs Origin : HK8489",          # สองคู่ในบรรทัดเดียว (หลายคอลัมน์) แยกไม่ได้ ไม่เดา
             "Tariff Rate : 0 VAT 7.00 % 123.45"]        # ค่าปนป้ายของช่องอื่น
    data = InvoiceData()
    got = extract_remark(lines, BY_NAME["customs_receipt"], data)
    assert [(e.label, e.value) for e in got] == [("เลขที่ใบสั่งซื้อ", "PO-123"), ("เงื่อนไขการชำระเงิน", "โอนเงิน")]


def test_draft_profile_cannot_auto_approve():
    d = InvoiceData(vendor={"name": "บริษัท ก จำกัด", "tax_id": "0105559117128"}, invoice_no="A1", date="2023-05-18",
                    amount_before_vat="100.00", vat_amount="7.00", total_amount="107.00")
    checks = run_checks(d)
    assert decide_status(d, checks)[1] == "auto_approved"  # generic: กฎเดิม
    assert decide_status(d, checks, can_auto_approve=False)[1] == "needs_review"
    assert Profile(name="x", status="").can_auto_approve and not Profile(name="x", status="Draft: รอยืนยัน").can_auto_approve


def test_remark_drops_entries_that_repeat_a_filled_field():
    d = InvoiceData(invoice_no="INV-1", total_amount="1940.00", document_fields={"entry_no": "B999"})
    remark = [ExtraField(label="รวมเงิน", value="1,940.00"),       # ค่าเท่ายอดที่มีช่องแล้ว (รูปแบบตัวเลขต่างกัน)
              ExtraField(label="เลขที่บิล", value="INV-1"),         # ค่าเท่า invoice_no
              ExtraField(label="Entry", value="B999"),               # ค่าเท่าฟิลด์ประจำชนิด
              ExtraField(label="เงื่อนไข", value="โอนเงิน"),
              ExtraField(label="เงื่อนไข", value="โอน เงิน")]       # ซ้ำกับรายการก่อนหน้า (ตัดช่องว่าง)
    got = dedupe_remark(remark, d, GENERIC_PROFILE)
    assert [(e.label, e.value) for e in got] == [("เงื่อนไข", "โอนเงิน")]


# ---------- สถานะและหลักฐานขึ้นกับโปรไฟล์ ----------

def _status(profile: Profile, data: InvoiceData):
    return decide_status(data, run_checks(data), required=profile.required, primary_amount=profile.primary_amount)


def test_withholding_without_total_amount_is_not_failed_when_its_own_total_exists():
    prof = BY_NAME["withholding_certificate"]
    d = InvoiceData(vendor={"name": "บริษัท ก จำกัด", "tax_id": "0105559117128"}, document_profile=prof.name,
                    document_fields={"total_paid": "2000.00"})
    missing, status = _status(prof, d)
    assert missing == ["document_fields.total_tax_withheld"] and status == "needs_review"
    assert decide_status(d, run_checks(d))[1] == "failed"  # โปรไฟล์ generic เดิม: ไม่มี total_amount → failed
    d.document_fields.clear()
    assert _status(prof, d)[1] == "failed"  # ไม่มียอดหลักของชนิดนี้ → failed


def test_primary_amount_null_never_fails():
    prof = Profile(name="p", primary_amount=None, required=("vendor.name",))
    assert _status(prof, InvoiceData())[1] == "needs_review"


def test_evidence_covers_document_fields_remark_and_profile_required_keys():
    prof = BY_NAME["customs_receipt"]
    text = "\n".join(CUSTOMS_EN)
    d = InvoiceData(document_profile=prof.name, document_fields={"total_charges": "123.45", "import_date": "2026-11-05"},
                    remark=[ExtraField(label="Flight Number", value="SR 603")])
    ev = build_field_evidence(d, run_checks(d), text, required=prof.required, profile=prof)
    assert ev["document_fields.total_charges"].grounded is True and ev["document_fields.total_charges"].level == "unverified"
    assert ev["document_fields.import_date"].grounded is False  # ไม่มีวันนั้นในข้อความ
    assert ev["document_fields.import_date"].level == "conflict"
    assert ev["document_fields.entry_no"].level == "missing"       # required ของโปรไฟล์แต่ไม่มีค่า
    assert "vendor.name" not in ev                                  # ไม่ใช่ required ของชนิดนี้
    assert ev["remark[0]"].grounded is False and set(flatten(d)) == {
        "document_fields.total_charges", "document_fields.import_date", "remark[0]"}


def test_not_applicable_from_profile_is_distinguished_from_missing():
    prof = Profile(name="p", required=("vendor.name", "invoice_no"), not_applicable=frozenset({"invoice_no"}))
    ev = build_field_evidence(InvoiceData(), [], "", not_applicable=prof.not_applicable, required=prof.required)
    assert ev["invoice_no"].level == "not_applicable" and ev["vendor.name"].level == "missing"


# ---------- LLM: schema ตามชนิด และ normalize ----------

def test_llm_schema_has_only_the_profiles_document_fields():
    props = profile_schema(BY_NAME["customs_receipt"])["properties"]
    assert set(props["document_fields"]["properties"]) == {f.key for f in BY_NAME["customs_receipt"].fields}
    assert "document_profile" not in props and "remark" in props
    assert "document_fields" not in profile_schema(GENERIC_PROFILE)["properties"]  # generic: LLM เห็น schema เล็กเท่าเดิม


def test_llm_document_fields_are_normalized_by_type_and_leftovers_go_to_remark():
    prof = BY_NAME["customs_receipt"]
    out = normalize_llm_output({
        "document_fields": {"total_charges": "1,123.45", "import_date": "05/11/2569", "entry_no": "B999-X1",
                            "duty": "ไม่ระบุ", "not_in_profile": "ค่าเกิน", "awb_no": ""},
        "extra_fields": [{"label": "PO", "value": "9"}],
    }, prof)
    assert out["document_fields"] == {"total_charges": "1123.45", "import_date": "2026-11-05", "entry_no": "B999-X1"}
    labels = {(e["label"], e["value"]) for e in out["remark"]}
    assert labels == {("PO", "9"), ("Duty (อากร)", "ไม่ระบุ"), ("not_in_profile", "ค่าเกิน")}  # อ่านไม่ได้/ไม่ประกาศ ไม่ทิ้ง
    assert "document_fields" in InvoiceData.model_validate(out).model_dump()


def test_generic_llm_output_moves_any_document_fields_to_remark():
    out = normalize_llm_output({"document_fields": {"x": "1"}})
    assert out["document_fields"] == {} and out["remark"] == [{"label": "x", "value": "1"}]


# ---------- pipeline: ผลตามโปรไฟล์ ----------

def _pipe():
    from fin_extractor.pipeline import FinanceOCRPipeline

    p = object.__new__(FinanceOCRPipeline)  # ไม่โหลด OCR/VLM — ทดสอบเฉพาะตรรกะสกัด
    p.cfg = {"validation": {"amount_tolerance": 0.02, "default_vat_rate": 0.07}}
    p.llm, p.profiles, p.doc_table, p.core_labels = None, PROFILES, {}, ()
    return p


def test_pipeline_selects_profile_and_fills_document_fields_and_remark():
    p = _pipe()
    data, sources, _ = p.extract_traced([*CUSTOMS_TH, "ผู้รับมอบอำนาจ: นาย ก"], "")
    assert data.document_profile == "customs_receipt"
    assert data.document_fields["total_charges"] == "123.45"
    assert sources["document_fields.total_charges"] == "rules"
    assert [(e.label, e.value) for e in data.remark] == [("ผู้รับมอบอำนาจ", "นาย ก")]
    assert p.profile_for(data).primary_amount == "document_fields.total_charges"


def test_pipeline_invoice_keeps_generic_profile_and_core_result():
    p = _pipe()
    data, _, _ = p.extract_traced(BITKUB_LINES, "")
    assert data.document_profile == "generic" and data.document_fields == {}
    core = extract_fields(BITKUB_LINES)
    for name in ("vendor", "buyer", "invoice_no", "date", "items", "subtotal", "total_amount", "vat_amount"):
        assert getattr(data, name) == getattr(core, name)


def test_old_results_with_extra_fields_still_load():
    d = InvoiceData.model_validate({"extra_fields": [{"label": "PO", "value": "1"}]})
    assert [(e.label, e.value) for e in d.remark] == [("PO", "1")]


def test_remark_from_llm_survives_when_rules_wins_and_has_its_own_remark():
    """rules ชนะการเลือกผลแต่มี remark อยู่แล้ว: remark ของ LLM ต้องไม่หาย (เดิมใช้ฝั่งผู้ชนะฝั่งเดียว)"""
    from fin_extractor.pipeline import fill_missing

    rules = extract_fields(BITKUB_LINES)
    rules.remark = [ExtraField(label="Ref", value="A1")]
    llm = rules.model_copy(deep=True)
    llm.remark = [ExtraField(label="เงื่อนไข", value="โอนเงิน"), ExtraField(label="ref", value="A 1")]  # ซ้ำกับของ rules
    merged = fill_missing(rules, llm)
    assert [(e.label, e.value) for e in merged.remark] == [("Ref", "A1"), ("เงื่อนไข", "โอนเงิน")]


def test_extract_traced_attributes_each_remark_to_its_source():
    from fin_extractor.pipeline import FinanceOCRPipeline

    class Stub:
        def __init__(self, data):
            self.data = data

        def extract(self, text, profile=None):
            return self.data

    p = object.__new__(FinanceOCRPipeline)
    p.cfg = {"validation": {"amount_tolerance": 0.02, "default_vat_rate": 0.07}}
    p.profiles, p.doc_table, p.core_labels = PROFILES, {}, ()
    lines = [*BITKUB_LINES, "เลขที่ใบสั่งซื้อ: PO-9"]
    rules = extract_fields(lines)
    llm = rules.model_copy(deep=True)
    llm.remark = [ExtraField(label="เงื่อนไข", value="โอนเงิน")]
    p.llm = Stub(llm)
    data, sources, agree = p.extract_traced(lines, "\n".join(lines))
    by_value = {e.value: sources[f"remark[{i}]"] for i, e in enumerate(data.remark)}
    assert by_value == {"PO-9": "rules", "โอนเงิน": "llm"} and not any(k.startswith("remark[") for k in agree)


# ---------- check ตามโปรไฟล์ ----------

def test_checks_not_in_profile_are_reported_as_null_not_removed():
    d = InvoiceData(vendor={"name": "A", "tax_id": "0105559117128"}, date="2023-05-18",
                    amount_before_vat="100.00", vat_amount="9.00", total_amount="109.00", vat_rate="0.07")  # VAT ผิด → vat_rate ไม่ผ่าน
    full = {c.name: c for c in run_checks(d)}
    assert full["vat_rate"].passed is False and full["vendor_tax_id_checksum"].passed is True
    only = {c.name: c for c in run_checks(d, applicable=["vendor_tax_id_checksum", "date_range"])}
    assert set(only) == set(full)                                          # ชื่อ check ชุดเดียวกันทุกชนิด
    assert only["vat_rate"].passed is None and only["vat_rate"].detail == "ไม่ใช้กับเอกสารชนิดนี้"
    assert only["vat_rate"].fields == []                                    # ไม่ยืนยัน/ไม่ขัดฟิลด์ใด
    assert only["vendor_tax_id_checksum"].passed is True and only["date_range"].passed == full["date_range"].passed
    assert [c.name for c in run_checks(d, applicable=None)] == list(full)   # ไม่ส่ง = กฎเดิม


def test_inapplicable_failure_no_longer_blocks_but_a_real_failure_still_does():
    d = InvoiceData(vendor={"name": "A", "tax_id": "0105559117128"}, invoice_no="1", date="2023-05-18",
                    amount_before_vat="100.00", vat_amount="9.00", total_amount="109.00")
    assert decide_status(d, run_checks(d))[1] == "needs_review"                                   # vat ผิด → ไม่อนุมัติ
    ok = run_checks(d, applicable=["vendor_tax_id_checksum", "date_range"])
    assert decide_status(d, ok)[1] == "auto_approved"
    bad_id = InvoiceData(vendor={"name": "A", "tax_id": "0105559117129"}, invoice_no="1", date="2023-05-18", total_amount="1.00")
    assert decide_status(bad_id, run_checks(bad_id, applicable=["vendor_tax_id_checksum"]))[1] == "needs_review"


def test_every_check_name_run_checks_emits_is_a_known_check():
    from fin_extractor.doctypes import KNOWN_CHECKS

    d = InvoiceData(vendor={"tax_id": "0105559117128"}, buyer={"tax_id": "0105559117128"}, date="2023-05-18",
                    subtotal="100.00", discount="10.00", amount_before_vat="90.00", vat_amount="6.30", total_amount="96.30",
                    items=[{"description": "x", "amount": "100.00"}])
    assert {c.name for c in run_checks(d)} == set(KNOWN_CHECKS)  # ครบทุกข้อและไม่มีชื่อนอกรายการ


def test_profile_rejects_unknown_check_name():
    with pytest.raises(ValueError, match="check"):
        Profile(name="p", checks=("not_a_check",))


def test_config_profiles_skip_invoice_arithmetic_for_non_invoice_documents():
    for name in ("withholding_certificate", "customs_receipt"):
        assert not {"items_sum", "vat_rate", "before_vat_plus_vat", "discount"} & set(BY_NAME[name].checks)
    assert "items_sum" in BY_NAME["payment_voucher"].checks and "vat_rate" not in BY_NAME["payment_voucher"].checks
    assert GENERIC_PROFILE.checks is None


def test_build_result_applies_the_profiles_checks():
    from types import SimpleNamespace

    from fin_extractor.pipeline import FinanceOCRPipeline

    p = object.__new__(FinanceOCRPipeline)
    p.cfg = {"validation": {"amount_tolerance": 0.02, "default_vat_rate": 0.07}}
    p.llm, p.profiles, p.doc_table, p.core_labels = None, PROFILES, {}, ()
    p.reader = SimpleNamespace(name="t")
    r = p.build_result("x", 1, [*CUSTOMS_EN, "VAT 7.00 % 123.45", "ใบเสร็จรับเงิน กรมศุลกากร"], "", None, [])
    by = {c.name: c for c in r.checks}
    assert r.data.document_profile == "customs_receipt"
    assert by["vat_rate"].passed is None and by["vat_rate"].detail == "ไม่ใช้กับเอกสารชนิดนี้"
    assert set(by) >= {"items_sum", "vat_rate", "date_range"}


def test_llm_schema_omits_fields_the_profile_does_not_have():
    props = profile_schema(BY_NAME["withholding_certificate"])["properties"]
    for name in ("items", "total_amount", "vat_amount", "invoice_no"):
        assert name not in props
    assert "vendor" in props and "buyer" in props and "document_fields" in props and "remark" in props
    assert "total_amount" in profile_schema(GENERIC_PROFILE)["properties"]
    assert "vat_amount" in profile_schema(BY_NAME["customs_receipt"])["properties"] and "total_amount" not in profile_schema(BY_NAME["customs_receipt"])["properties"]
