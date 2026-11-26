"""remark ของขั้นสกัด: ข้อมูลสำคัญที่ไม่มีช่องไม่ถูกทิ้ง (kind/source_chunk) และค่าที่ LLM เสนอแต่ถูกปัดตกถูกเก็บไว้ใน ``rejected``."""

from fin_extractor.extraction import lines_with_chunks, markdown_to_lines
from fin_extractor.extraction.profile_extractor import UNMAPPED_LABEL, extract_remark, extract_unmapped
from fin_extractor.schema import ExtractionResult, InvoiceData, RejectedValue

MD = """<!-- s00 text -->
บริษัท ตัวอย่าง จำกัด
เลขที่: INV-001

<!-- s03 table 1/2 -->
<table><tr><td>ยอดชำระ Total</td><td>1,940.00</td></tr></table>
<table><tr><td>รวมเงิน</td><td>2,000.00</td></tr></table>
"""


def test_lines_with_chunks_equals_markdown_to_lines_and_labels_each_line_with_its_section():
    pairs = lines_with_chunks(MD)
    assert [ln for ln, _ in pairs] == markdown_to_lines(MD)  # ไม่เปลี่ยนเนื้อหา/ลำดับบรรทัดที่ rules เห็น
    assert dict(pairs)["เลขที่: INV-001"] == "s00" and dict(pairs)["ยอดชำระ Total | 1,940.00"] == "s03"
    assert [c for _ln, c in lines_with_chunks("ข้อความทั้งหน้า ไม่มีคอมเมนต์")] == [None]


def test_unmapped_keeps_lines_with_an_amount_date_or_valid_tax_id_that_no_field_holds():
    data = InvoiceData(total_amount="2000.00")
    lines = [ln for ln, _ in lines_with_chunks(MD)]
    chunks = [c for _ln, c in lines_with_chunks(MD)]
    got = extract_unmapped(lines, data, chunks)
    assert [(e.value, e.source_chunk, e.kind, e.label) for e in got] == [("ยอดชำระ Total | 1,940.00", "s03", "unmapped", UNMAPPED_LABEL)]
    # ยอด 2,000.00 มีช่องรับแล้ว (total_amount) / ชื่อบริษัท ไม่มีสัญญาณยอด-วันที่-เลขภาษี จึงไม่ถูกเก็บ


def test_unmapped_ignores_13_digit_codes_that_fail_the_tax_id_checksum():
    got = extract_unmapped(["8851295511045 ปากกา", "0105549019026 บริษัท"], InvoiceData(), None)  # ตัวแรกรหัสสินค้า ตัวหลังเลขภาษีที่ checksum ผ่าน
    assert [e.value for e in got] == ["0105549019026 บริษัท"] and got[0].source_chunk is None


def test_pair_remark_carries_its_source_chunk():
    from fin_extractor.doctypes import GENERIC_PROFILE

    lines = ["หมายเหตุพิเศษ : ส่งของภายใน 7 วัน"]
    got = extract_remark(lines, GENERIC_PROFILE, InvoiceData(), ["s07"])
    assert got and got[0].source_chunk == "s07" and got[0].kind == "pair"


def test_result_schema_has_rejected_values():
    r = ExtractionResult(source_file="a.pdf", engine="e", data=InvoiceData(), status="check", rejected=[RejectedValue(key="vendor.tax_id", reason="checksum ไม่ผ่าน")])
    assert r.model_dump()["rejected"] == [{"key": "vendor.tax_id", "reason": "checksum ไม่ผ่าน"}]


def test_build_result_stores_what_hybrid_rejected():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_hybrid import LINES, _pipe  # noqa: PLC0415

    p = _pipe(None)
    p.llm, p.strategy = None, "hybrid"  # ไม่มี LLM จริง: ทดสอบเฉพาะการส่ง last_rejected ลงผล
    p.reader = type("R", (), {"name": "fake"})()
    p.last_rejected = [("vendor.tax_id", "checksum ไม่ผ่าน")]
    r = p.build_result("a.pdf", 1, LINES, "\n".join(LINES), None, [])
    assert [(x.key, x.reason) for x in r.rejected] == [("vendor.tax_id", "checksum ไม่ผ่าน")]
