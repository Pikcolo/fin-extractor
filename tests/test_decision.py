"""decision model แบบ shadow (decision/): ไม่โหลดโมเดลจริง ใช้ backend ปลอมที่บันทึกสิ่งที่ถูกถาม."""

import copy
import json

from fin_extractor.config import load_config
from fin_extractor.decision import Decider, make_decider
from fin_extractor.decision.client import orders
from fin_extractor.decision.questions import build, first_chunk
from fin_extractor.doctypes import load_profiles
from fin_extractor.schema import ExtractionResult, InvoiceData

CFG = load_config()
PROFILES = load_profiles(CFG["document_types"])
MD = ("<!-- s00 strip_00 -->\nใบเสร็จรับเงิน\nบริษัท เอ จำกัด\n\n"
      "<!-- s01 strip_01 -->\nรวมเงิน 100.00\n\n<!-- s00 strip_00 supplement openthai-text -->\nข้อความเสริม")


class FakeBackend:
    """ตอบตามตัวเลือกที่อยู่ลำดับแรกเสมอ (จำลองโมเดลที่ลำเอียงตามลำดับ) บันทึกทุกคำถาม."""

    name = "fake"

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def choice(self, state, instructions, options):
        if self.fail:
            raise RuntimeError("no model")
        self.calls.append(("choice", state, list(options)))
        first = next(iter(options))
        return {k: (0.7 if k == first else 0.3 / (len(options) - 1)) for k in options}, 0.01

    def noul(self, state, instructions):
        self.calls.append(("noul", state, None))
        return 0.2 if "ให้ระบุ" in state["ค่าในช่องชื่อ"] else 0.9


def _cfg(**kw):
    return {**CFG["decision"], **kw}


def test_disabled_by_default_and_cli_flag_overrides():
    assert CFG["decision"]["enabled"] is False and make_decider(CFG) is None
    assert isinstance(make_decider(CFG, True), Decider) and make_decider(CFG, False) is None


def test_state_uses_only_the_first_chunk_and_extracted_fields_never_the_whole_page():
    assert first_chunk(MD) == "ใบเสร็จรับเงิน\nบริษัท เอ จำกัด"  # ไม่รวม s01 และไม่รวมท่อนเสริมของ s00
    data = InvoiceData(vendor={"name": "บริษัท เอ จำกัด"})
    qs = build(data, MD, PROFILES, _cfg())
    for q in qs:
        dumped = json.dumps(q["state"], ensure_ascii=False)
        assert "รวมเงิน 100.00" not in dumped and "ข้อความเสริม" not in dumped
    assert [q["task"] for q in qs] == ["doc_type", "profile", "party_name"]  # buyer.name ว่าง = ไม่ถาม
    prof = next(q for q in qs if q["task"] == "profile")
    assert "generic" in prof["options"] and "withholding_certificate" in prof["options"]


def test_choice_is_asked_in_several_orders_and_averaged():
    assert orders(1) == [[0]] and orders(2) == [[0, 1], [1, 0]] and len(orders(8)) == 3
    fake = FakeBackend()
    d = Decider(_cfg(), fake)
    q = {"task": "doc_type", "key": "document_type", "type": "choice", "state": {"x": 1}, "instructions": "?",
         "options": {"a": None, "b": None, "c": None}}
    (dec,) = d.ask([q])
    assert len(fake.calls) == 3 and dec.orders == 3
    assert dec.agreed is False  # โมเดลปลอมเลือกตัวแรกเสมอ ลำดับต่างกันจึงได้คำตอบต่างกัน
    assert abs(sum(dec.probabilities.values()) - 1) < 1e-3 and dec.abstain == 0.01 and dec.model == "fake"


def test_party_name_flags_form_instruction_text():
    data = InvoiceData(vendor={"name": "(ให้ระบุว่าเป็น บุคคล นิติบุคคล บริษัท สมาคม หรือคณะบุคคล)"}, buyer={"name": "บริษัท บี จำกัด"})
    decs = Decider(_cfg(tasks=["party_name"]), FakeBackend()).ask(build(data, MD, PROFILES, _cfg(tasks=["party_name"])))
    assert {d.key: d.answer for d in decs} == {"vendor.name": "no", "buyer.name": "yes"}


def test_backend_failure_records_nothing_and_does_not_raise():
    q = build(InvoiceData(), MD, PROFILES, _cfg(tasks=["profile"]))
    assert Decider(_cfg(), FakeBackend(fail=True)).ask(q) == []


def test_decisions_do_not_change_the_rest_of_the_result():
    r = ExtractionResult(source_file="x", engine="e", data=InvoiceData(), status="needs_review")
    before = r.model_dump_json()
    assert "decisions" not in json.loads(before)  # ว่าง = ไม่มีคีย์ ผลเดิมเหมือนเดิมทุกไบต์
    r2 = copy.deepcopy(r)
    r2.decisions = Decider(_cfg(tasks=["profile"]), FakeBackend()).ask(build(r2.data, MD, PROFILES, _cfg(tasks=["profile"])))
    after = json.loads(r2.model_dump_json())
    assert len(after.pop("decisions")) == 1 and after == json.loads(before)
