"""แยกยอดก่อน VAT / VAT / ยอดรวม จากบรรทัดตาราง — ข้อความแต่ละกรณีมาจากผล OCR ของสแกนจริง (เลขเอกสารในชื่อ test)."""

from decimal import Decimal

from fin_extractor.extraction import extract_fields
from fin_extractor.extraction.rule_extractor import markdown_to_lines


def amounts(text: str) -> dict:
    d = extract_fields(markdown_to_lines(text))
    return {k: getattr(d, k) for k in ("subtotal", "amount_before_vat", "vat_amount", "total_amount")
            if getattr(d, k) is not None}


def test_labels_in_one_cell_pair_in_order_with_amounts_on_the_next_line_1312():
    text = ("<table><tr>\n<td>สามพันเก้าร้อยสี่สิบเก้าบาทถ้วน</td>\n<td>รวมเป็นเงิน จำนวนภาษีมูลค่าเพิ่ม 7% รวมมูลค่าสินค้า</td>\n"
            "<td>3,949.00 258.35 3,690.65</td>\n</tr></table>")
    assert amounts(text) == {"subtotal": Decimal("3949.00"), "vat_amount": Decimal("258.35"),
                             "amount_before_vat": Decimal("3690.65")}


def test_vat_exempt_goods_value_is_not_vat_1310():
    lines = ["ผู้รับสินค้า | ผู้ส่งสินค้า | มูลค่าสินค้าที่ยกเว้นภาษีมูลค่าเพิ่ม", "737.29", "ยอดสุทธิ 788.90"]
    d = extract_fields(lines)
    assert d.vat_amount is None and d.total_amount == Decimal("788.90")


def test_grand_total_with_money_word_is_the_total_not_the_subtotal_1316():
    assert amounts("ค่าภาษีมูลค่าเพิ่ม | 149.16\nรวมเงินทั้งสิ้น (บาท) | 149.16") == {
        "vat_amount": Decimal("149.16"), "total_amount": Decimal("149.16")}


def test_subtotal_beside_the_amount_in_words_and_dotted_vat_label_1317():
    lines = markdown_to_lines("(แปดพันยี่สิบห้าบาทถ้วน) | รวมเงิน Sub Total | 7,500:00\n"
                              "ชำระเงินโดย : ( ) เงินสด | ภาษีมูลค่ำเพิ่ม V.A.T. | 525.00\n"
                              "ยอดเงินสุทธิ Net Total | 8,025.00")
    d = extract_fields(lines)
    assert (d.subtotal, d.vat_amount, d.total_amount) == (Decimal("7500.00"), Decimal("525.00"), Decimal("8025.00"))


def test_baht_and_satang_columns_join_1320():
    assert amounts("รวมทั้งสิ้น | 103,062 | 50") == {"total_amount": Decimal("103062.50")}


def test_one_label_with_several_values_is_left_empty_50_tawi():
    assert amounts("รวมเงินที่จ่ายและภาษีที่หักนำส่ง | 106,250.00 | 3,187.50") == {}


def test_colon_is_read_as_a_decimal_point_only_with_thousands_separators():
    assert markdown_to_lines("7,500:00 เวลา 10:30") == ["7,500.00 เวลา 10:30"]
