"""อ่าน/เขียนภาพและ PDF.

ใช้ ``np.fromfile`` + ``cv2.imdecode`` แทน ``cv2.imread`` เพราะ cv2.imread บน Windows
อ่าน path ที่มีภาษาไทยไม่ได้
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}


def read_image(path: str | Path) -> np.ndarray:
    """คืนภาพ BGR (uint8)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"อ่านภาพไม่ได้: {path}")
    return img


def save_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"encode ภาพไม่ได้: {path}")
    buf.tofile(str(path))


def _render_pdf(path: Path, dpi: int) -> list[np.ndarray]:
    import pypdfium2 as pdfium

    pages = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for page in pdf:
            pil = page.render(scale=dpi / 72).to_pil().convert("RGB")
            pages.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
    finally:
        pdf.close()
    return pages


def load_pages(path: str | Path, pdf_dpi: int = 300) -> list[np.ndarray]:
    """โหลดไฟล์เป็น list ของหน้า (ภาพ 1 ไฟล์ = 1 หน้า, PDF = หลายหน้า)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _render_pdf(path, pdf_dpi)
    if ext in IMAGE_EXTS:
        return [read_image(path)]
    raise ValueError(f"ไม่รองรับไฟล์ {ext}")


def page_count(path: str | Path) -> int:
    """จำนวนหน้าของไฟล์โดยไม่เรนเดอร์ (ภาพ = 1 หน้า)."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return 1
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def iter_input_files(path: str | Path) -> Iterator[Path]:
    """รับได้ทั้งไฟล์เดี่ยวหรือโฟลเดอร์."""
    path = Path(path)
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in SUPPORTED_EXTS:
            yield p
