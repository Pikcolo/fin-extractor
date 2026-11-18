"""ขั้น 3 ของ flow: แบ่งหน้าเป็นส่วน (``layout``) → ภาพรายส่วนก่อน OCR (``images``) → ตัดภาพตามผัง (``crops``). ภาพล้วน ไม่เรียก OCR."""

from .crops import iter_section_crops, layout_path, load_final_image
from .images import VARIANTS, binarize_inverted, drop_empty_rows_by_table, flatten_background, gray_value_channel
from .layout import crop_full, layout_sections, resize_to_work

__all__ = ["VARIANTS", "binarize_inverted", "crop_full", "drop_empty_rows_by_table", "flatten_background", "gray_value_channel",
           "iter_section_crops", "layout_path", "layout_sections", "load_final_image", "resize_to_work"]
