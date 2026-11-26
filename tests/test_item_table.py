"""รายการสินค้าจากตาราง HTML และเลขที่เอกสาร — ตาราง/บรรทัดแต่ละกรณีมาจากผล OCR ของสแกนจริง (เลขเอกสารในชื่อ test)."""

from decimal import Decimal

from fin_extractor.extraction import extract_fields
from fin_extractor.extraction.item_table import table_items
from fin_extractor.extraction.rule_extractor import markdown_to_lines


def table(*rows: str) -> str:
    return "<table>" + "".join(f"<tr>{r}</tr>" for r in rows) + "</table>"


def td(*cells: str) -> str:
    return "".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in cells)


def rows(raw: str) -> list[tuple]:
    return [(i.quantity, i.unit_price, i.amount, i.description) for i in table_items(raw, 30)]


def test_columns_are_read_by_header_and_unit_of_measure_is_not_a_price_1310():
    raw = table(td("Item ลำดับ", "Article Number รหัสสินค้า", "Article Description รายการ", "Qty จำนวน", "Unit หน่วย",
                   "Price/Unit ราคา/หน่วย", "Discount ส่วนลด", "Vat ภาษี", "Amount จำนวนเงิน"),
                td("1", "8851295511045", "ปากกาเขียนไวท์บอร์ด สีเขียว", "1.00", "ด้าม", "20.00", "5%", "V", "19.00"))
    assert rows(raw) == [(Decimal("1.00"), Decimal("20.00"), Decimal("19.00"), "ปากกาเขียนไวท์บอร์ด สีเขียว")]


def test_summary_rows_after_the_items_are_not_items_1308():
    head = td("รหัสสินค้า Product Code", "รายละเอียด Description", "จำนวน Quantity", "ราคา/หน่วย Unit price",
              "ส่วนลด Discount", "จำนวนเงิน Amount")
    part1 = table(head, td("A0174103", "UPS 2200VA APC", "4", "6,190.00", "", "24,760.00"))
    part2 = table(head, td("จำนวนเงินทั้งสิ้น NET TOTAL WORDS", "จำนวนเงินทั้งสิ้น / NET TOTAL", "24,760.00", ""),
                  td("", "รวมราคาสินค้า / SUB TOTAL", "23,140.19", ""))
    assert rows(part1 + part2) == [(Decimal(4), Decimal("6190.00"), Decimal("24760.00"), "UPS 2200VA APC")]


def test_amount_in_words_row_ends_the_items_1312():
    raw = table(td("ลำดับ", "รหัสสินค้า", "ชื่อสินค้า", "ราคา", "จำนวน", "รวม"),
                td("1", "03A03", "ESP32", "359.00", "11", "3,949.00"),
                td("", "", "ค่าบริการจัดส่งสินค้า", "0.00", "1", "0.00"),
                td("สามพันเก้าร้อยสี่สิบเก้าบาทถ้วน", "รวมเป็นเงิน จำนวนภาษีมูลค่าเพิ่ม 7%", "3,949.00 258.35"))
    assert [r[:3] for r in rows(raw)] == [(Decimal(11), Decimal("359.00"), Decimal("3949.00")),
                                          (Decimal(1), Decimal("0.00"), Decimal("0.00"))]


def test_two_items_in_one_row_split_when_every_number_column_agrees_1309():
    raw = table(td("รหัสสินค้า Product Code", "รายละเอียด Description", "จำนวน Quantity", "ราคา/หน่วย Unit price",
                   "ส่วนลด Discount", "จำนวนเงิน Amount"),
                td("A0158095 A0147468", "Cable Micro SD", "1 2", "115.00 1,930.00", "", "115.00 3,860.00"))
    assert [r[:3] for r in rows(raw)] == [(Decimal(1), Decimal("115.00"), Decimal("115.00")),
                                          (Decimal(2), Decimal("1930.00"), Decimal("3860.00"))]


def test_baht_satang_header_spans_two_columns_and_total_row_is_not_an_item_1320():
    raw = table(td("ลำดับที่", "รายการ", '<td colspan="2">จำนวนเงิน (บาท)</td>', "หมายเหตุ"),
                td("1", "ค่าปรับปรุง Machine Learning", "103,062", "50", ""),
                td("", "รวมทั้งสิ้น", "103,062", "50", ""))
    assert rows(raw) == [(None, None, Decimal("103062.50"), "ค่าปรับปรุง Machine Learning")]


def test_quantity_with_unit_and_price_per_unit_column_1317():
    raw = table(td("รหัสสินค้า Product Code", "รายละเอียด Description", "จำนวน Quantity", "หน่วยละ Unit",
                   "ส่วนลด Discount", "จำนวนเงิน Amount"),
                td("PD0001", "ค่ามัดจำสินค้า 30%", "1.00 SET", "7,500.00", "", "7,500.00"))
    assert rows(raw) == [(Decimal("1.00"), Decimal("7500.00"), Decimal("7500.00"), "ค่ามัดจำสินค้า 30%")]


def test_tables_without_an_item_header_give_no_items_and_plain_text_falls_back():
    assert table_items(table(td("Non-Taxable Charges", "149.16"), td("7% VAT", "17.57")), 30) == []
    assert table_items("รายการ\nกาแฟ 45.00", 30) is None


def no(text: str) -> str | None:
    return extract_fields(markdown_to_lines(text), text).invoice_no


def test_invoice_no_label_beats_account_no_1313():
    assert no("Account No. :8TH013E199\nInvoice No. : 000002780808") == "000002780808"


def test_address_after_bare_lekthi_is_not_a_document_number_1308():
    assert no("ที่อยู: Address | เลขที่ 49/12 ซอย 6 ถนน นิพัทธ์สงเคราะห์ 1 | SO26040857") is None


def test_value_under_a_header_cell_1312():
    assert no(table(td("เลขที่", "ออกเมื่อ"), td("BI8461-101069", "10 ตุลาคม 2569"))) == "BI8461-101069"


def test_value_before_the_english_label_1306():
    assert no("เลขที่ผู้เสียภาษี Tax ID | 0-9055-64002-79-4 | ใบวางบิล : IN-691005003 Invoice No") == "IN-691005003"


def test_a_date_is_not_a_document_number_1322():
    assert no("ใบสำคัญรับเงิน\nเลขที่ 12/10/2026\nวันที่ 12/10/2026") is None


def test_value_in_the_next_cell_when_the_label_cell_has_only_captions_1308():
    raw = table(td("วันที่ : Date", "15/10/2026"), td("เลขที่ใบกำกับภาษี : Invoice No.", "BAUNS00054022610000155"),
                td("เลขที่คำสั่งซื้อ : Order No.", "SO26040857"))
    assert no(raw) == "BAUNS00054022610000155"
    assert no(table(td("เลขที่ No.", "รายการ Description"), td("1", "ค่าบริการทำบัญชี"))) is None  # หัวตารางรายการ ไม่ใช่เลขที่เอกสาร
