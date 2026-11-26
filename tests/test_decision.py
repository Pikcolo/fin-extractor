"""decision model แบบ shadow (decision/): ไม่โหลดโมเดลจริง ใช้ backend ปลอมที่บันทึกสิ่งที่ถูกถาม."""

import copy
import json

from fin_extractor.config import load_config
from fin_extractor.decision import Decider, make_decider
from fin_extractor.decision.client import orders
from fin_extractor.decision.questions import NOT_FOUND, build, first_chunk, lines_with
from fin_extractor.doctypes import load_profiles
from fin_extractor.schema import ExtractionResult, InvoiceData

CFG = load_config()
PROFILES = load_profiles(CFG["document_types"])
MD = ("<!-- s00 strip_00 -->\nใบเสร็จรับเงิน\nบริษัท เอ จำกัด\n\n"
      "<!-- s01 strip_01 -->\nเลขที่ INV-001 วันที่ 05/10/2569\nบรรทัดที่ไม่เกี่ยวกับอะไรเลย\nรวมเงิน | 1,940.00\nVAT 7% | 135.80\n\n"
      "<!-- s00 strip_00 supplement openthai-text -->\nข้อความเสริม")


class FakeBackend:
    """choice: ตอบตัวเลือกลำดับแรก (จำลองโมเดลที่ลำเอียงตามลำดับ) · yes/no: ชื่อที่มีคำแนะนำของฟอร์มปน = no บันทึกทุกการเรียก."""

    name = "fake"

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def choice(self, state, instructions, options):
        if self.fail:
            raise RuntimeError("no model")
        self.calls.append(("choice", state, list(options)))
        first = next(iter(options))
        return {k: (0.7 if k == first else 0.3 / max(1, len(options) - 1)) for k in options}, 0.01

    def nouls(self, state, questions):
        self.calls.append(("nouls", state, list(questions)))
        bad = "ให้ระบุ" in json.dumps(state, ensure_ascii=False)
        return {k: 0.2 if bad or "ให้ระบุ" in t else 0.9 for k, t in questions.items()}


def _cfg(**kw):
    return {**CFG["decision"], **kw}


def test_disabled_by_default_and_cli_flag_overrides():
    assert CFG["decision"]["enabled"] is False and make_decider(CFG) is None
    assert isinstance(make_decider(CFG, True), Decider) and make_decider(CFG, False) is None


def test_state_holds_head_and_only_lines_with_extracted_values_never_the_whole_page():
    assert first_chunk(MD) == "ใบเสร็จรับเงิน\nบริษัท เอ จำกัด"  # ไม่รวม s01 และไม่รวมท่อนเสริมของ s00
    data = InvoiceData(vendor={"name": "บริษัท เอ จำกัด"}, invoice_no="INV-001", total_amount="1940.00")
    qs = build(data, MD, PROFILES, _cfg())
    for q in qs:
        dumped = json.dumps(q["state"], ensure_ascii=False)
        assert "บรรทัดที่ไม่เกี่ยวกับอะไรเลย" not in dumped and "VAT 7%" not in dumped  # บรรทัดที่ไม่มีค่าที่สกัดได้ไม่ถูกป้อน
    assert [q["key"] for q in qs if q["task"] == "name_only"] == ["vendor.name"]  # buyer.name ว่าง = ไม่ถาม
    assert qs[-1]["state"] == {"ค่าในช่องชื่อ": "บริษัท เอ จำกัด"}  # คำถามชื่อเห็นแค่ค่า


def test_lines_with_matches_money_by_value_and_text_without_spaces():
    lines = ["รวม | 103,062 | 50", "ยอดสุทธิ 1,940.00", "บริษัท  เอ จำกัด"]
    assert lines_with("1940.00", "money", lines) == ["ยอดสุทธิ 1,940.00"]
    assert lines_with("บริษัท เอ จำกัด", "text", lines) == ["บริษัท  เอ จำกัด"]
    assert lines_with("103062.50", "money", lines) == []  # ยอดที่ rules ประกอบจากช่องบาท|สตางค์ไม่มีในบรรทัดตรง ๆ


def test_yes_no_questions_on_one_state_are_asked_in_one_call():
    fake = FakeBackend()
    state = {"ค่า": "x"}
    qs = [{"task": "t", "key": k, "type": "noul", "state": state, "instructions": f"{k}?"} for k in ("a", "b", "c")]
    decs = Decider(_cfg(), fake).ask(qs)
    assert [c[0] for c in fake.calls] == ["nouls"] and len(fake.calls[0][2]) == 3  # state เดียวกัน = เรียกครั้งเดียว
    assert [d.key for d in decs] == ["a", "b", "c"] and all(d.instructions for d in decs)


def test_name_only_flags_form_instruction_text():
    data = InvoiceData(vendor={"name": "(ให้ระบุว่าเป็น บุคคล นิติบุคคล บริษัท สมาคม หรือคณะบุคคล)"}, buyer={"name": "บริษัท บี จำกัด"})
    decs = Decider(_cfg(), FakeBackend()).ask(build(data, MD, PROFILES, _cfg(tasks=["name_only"])))
    assert {d.key: d.answer for d in decs} == {"vendor.name": "no", "buyer.name": "yes"}


def test_missing_hint_offers_values_from_lines_plus_not_found_and_is_a_suggestion():
    qs = build(InvoiceData(), MD, PROFILES, _cfg(tasks=["missing_hint"]), missing=["total_amount", "invoice_no"])
    (q,) = qs  # เลขที่ (ข้อความ/รหัส) ไม่ถาม: หาค่าที่เป็นไปได้ไม่ได้โดยไม่ตั้งเกณฑ์เอง
    assert q["key"] == "total_amount" and set(q["options"]) == {"1940.00", "135.80", NOT_FOUND}
    (dec,) = Decider(_cfg(), FakeBackend()).ask(qs)
    assert dec.orders == 3 and dec.task == "missing_hint"


def test_choice_is_asked_in_several_orders_and_averaged():
    assert orders(1) == [[0]] and orders(2) == [[0, 1], [1, 0]] and len(orders(8)) == 3
    fake = FakeBackend()
    q = {"task": "doc_type", "key": "document_type", "type": "choice", "state": {"x": 1}, "instructions": "?",
         "options": {"a": None, "b": None, "c": None}}
    (dec,) = Decider(_cfg(), fake).ask([q])
    assert len(fake.calls) == 3 and dec.orders == 3 and dec.agreed is False
    assert abs(sum(dec.probabilities.values()) - 1) < 1e-3 and dec.abstain == 0.01 and dec.model == "fake"


def test_backend_failure_records_nothing_and_does_not_raise():
    q = build(InvoiceData(), MD, PROFILES, _cfg(tasks=["profile"]))
    assert Decider(_cfg(), FakeBackend(fail=True)).ask(q) == []


def test_decisions_do_not_change_the_rest_of_the_result():
    r = ExtractionResult(source_file="x", engine="e", data=InvoiceData(vendor={"name": "บริษัท เอ จำกัด"}), status="needs_review")
    before = r.model_dump_json()
    assert "decisions" not in json.loads(before)  # ว่าง = ไม่มีคีย์ ผลเดิมเหมือนเดิมทุกไบต์
    r2 = copy.deepcopy(r)
    r2.decisions = Decider(_cfg(), FakeBackend()).ask(build(r2.data, MD, PROFILES, _cfg()))
    after = json.loads(r2.model_dump_json())
    assert len(after.pop("decisions")) >= 3 and after == json.loads(before)


def test_missing_hint_is_confirmed_only_by_checks_that_pass_with_the_suggested_value():
    from fin_extractor.pipeline import FinanceOCRPipeline
    from fin_extractor.schema import Decision

    pipe = FinanceOCRPipeline(CFG, engine="openthai")
    raw = ("<!-- s00 -->\nใบกำกับภาษี\nเลขที่ INV-1 วันที่ 01/10/2026\n"
           "มูลค่าก่อนภาษี 100.00\nภาษีมูลค่าเพิ่ม 7.00\n")
    r = pipe.process_raw("x.pdf", 1, raw)
    assert r.data.total_amount is None
    hint = lambda v: Decision(task="missing_hint", key="total_amount", type="choice", answer=v, probabilities={v: 1.0}, model="fake")
    right, wrong, none = hint("107.00"), hint("100.00"), hint(NOT_FOUND)
    pipe.confirm_hints(r, [right])
    pipe.confirm_hints(r, [wrong, none])
    assert "before_vat_plus_vat" in right.confirmed_by
    assert wrong.confirmed_by == [] and none.confirmed_by == []
    assert r.data.total_amount is None  # shadow: ผลจริงไม่เปลี่ยน


def _hint(key, v):
    from fin_extractor.schema import Decision

    return Decision(task="missing_hint", key=key, type="choice", answer=v, probabilities={v: 1.0}, model="fake")


def _receipt(pipe, extra=""):
    return pipe.process_raw("x.pdf", 1, "<!-- s00 -->\nใบกำกับภาษี\nเลขที่ INV-1 วันที่ 01/10/2026\n"
                                        "มูลค่าก่อนภาษี 100.00\nภาษีมูลค่าเพิ่ม 7.00\n" + extra)


def test_fill_writes_only_check_confirmed_values_with_source_decision_and_never_auto_approves():
    from fin_extractor.pipeline import FinanceOCRPipeline

    pipe = FinanceOCRPipeline(CFG, engine="openthai")
    r = _receipt(pipe, "ชำระ 107.00\n")  # ค่าที่เสนอมาจากบรรทัด OCR เสมอ (ตัวเลือกของ missing_hint)
    assert r.data.total_amount is None
    right = _hint("total_amount", "107.00")
    pipe.confirm_hints(r, [right])
    out = pipe.apply_hints(r, [right])
    assert str(out.data.total_amount) == "107.00" and right.filled
    assert out.review["total_amount"].source == "decision" and out.review["total_amount"].level == "verified"
    assert out.status != "auto_approved"
    assert r.data.total_amount is None  # ผลเดิมไม่ถูกแก้ในที่

    wrong = _hint("total_amount", "100.00")
    pipe.confirm_hints(r, [wrong])
    assert pipe.apply_hints(r, [wrong]) is r and not wrong.filled


def test_tax_id_hint_is_never_filled_even_when_the_checksum_passes():
    from fin_extractor.pipeline import FinanceOCRPipeline

    pipe = FinanceOCRPipeline(CFG, engine="openthai")
    r = _receipt(pipe)
    tid = _hint("vendor.tax_id", "0105549019026")  # ผ่าน checksum แต่ไม่บอกว่าเป็นของฝ่ายไหน
    pipe.confirm_hints(r, [tid])
    assert tid.confirmed_by == [] and pipe.apply_hints(r, [tid]) is r


def test_fill_is_dropped_when_it_makes_another_check_fail():
    from fin_extractor.pipeline import FinanceOCRPipeline
    from fin_extractor.schema import ValidationCheck

    pipe = FinanceOCRPipeline(CFG, engine="openthai")
    r = _receipt(pipe)
    h = _hint("total_amount", "107.00")
    h.confirmed_by = ["before_vat_plus_vat"]
    # จำลอง check ที่ผ่านอยู่เดิม: ถ้าเติมแล้ว check นี้กลายเป็นไม่ผ่าน ต้องไม่เติม
    r.checks = [*r.checks, ValidationCheck(name="items_sum", passed=True, detail="")]
    orig = pipe.assess

    def assess(data, *a, **k):
        checks, *rest = orig(data, *a, **k)
        return ([c if c.name != "items_sum" else ValidationCheck(name="items_sum", passed=False, detail="") for c in checks], *rest)

    pipe.assess = assess
    assert pipe.apply_hints(r, [h]) is r and not h.filled
