"""Preprocess lab: ชุดเครื่องมือสำหรับ notebook รายไฟล์ ใช้เทียบ edge detection / threshold / การแบ่ง section
เพื่อดูว่าภาพแบบไหนเหมาะกับวิธีไหน (ข้อมูลตั้งต้นของ router ตอนปรับ input)

หลักการ
- ทำงานบนสำเนาขนาดคงที่ (ด้านยาว ``WORK`` พิกเซล) ให้ตัวเลขเทียบข้ามไฟล์ได้ ค่าที่กวาด (kernel, gap, threshold) เป็น **ค่าทดลอง**
  ไม่ใช่ค่าตั้งต้นของระบบ router ต้องอ้างผลรวมจากทุกไฟล์ ไม่ใช่ค่าที่ตั้งไว้ที่นี่
- ผลอยู่ที่ ``data/preprocess/<dataset>/<profile>/sections/features/<ไฟล์>.json`` และ ``.../clean/<ไฟล์>/`` (พาธมาจาก ``DataPaths`` ใน paths.py
  ที่เดียว ใต้ data/ ไม่ถูก commit เพราะเป็นข้อมูลภาษีจริง)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import load_pages, read_image  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of  # noqa: E402
from fin_extractor.preprocessing import binarize as bin_ops  # noqa: E402
from fin_extractor.preprocessing import enhance, geometry  # noqa: E402

from matplotlib import font_manager  # noqa: E402

# ชื่อภาพภาษาไทยในกราฟ: font ตั้งต้นของ matplotlib ไม่มีอักษรไทย ใช้ Sarabun ที่มากับโปรเจกต์ (assets/fonts, SIL OFL)
for _f in (ROOT / "assets" / "fonts").glob("Sarabun-*.ttf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams["font.family"] = ["Sarabun", "DejaVu Sans"]

WORK = 1600  # ด้านยาวของสำเนาที่วิเคราะห์ (ตรงกับ preprocessing.min_long_side ใน config)
PATHS = DataPaths(load_config())  # พาธข้อมูลถามจาก paths.py ที่เดียว ไม่ฝังเอง
SHOW = 760  # ด้านยาวของภาพที่แสดงใน notebook (ย่อให้ไฟล์ไม่บวม)

# ค่ากวาด (ทดลอง): ขนาด kernel ขยายหมึก (กว้าง, สูง) ที่ WORK=1600 ไล่จากระดับตัวอักษร → คำ → บรรทัด → บล็อก → ส่วนใหญ่
KERNELS = [(9, 3), (25, 5), (60, 9), (140, 20), (300, 45)]
BAND_GAPS = [8, 16, 32]  # ความสูงขั้นต่ำของแถวว่างที่ถือเป็นรอยตัดแนวนอน (พิกเซล)


# --------------------------------------------------------------------------- โหลด / แสดงผล
STATE: dict = {}  # ผลของ load() ครั้งล่าสุด (ชุดข้อมูล, ที่มาของภาพ, มุมเอียงก่อน/หลังปรับ) ใช้ต่อใน features() และตอนบันทึก


def _resize_to_work(img: np.ndarray) -> tuple[np.ndarray, float]:
    scale = min(1.0, WORK / max(img.shape[:2]))
    return (cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img), scale


def load(path: str | Path, dpi: int = 300, straighten: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """(ภาพเต็ม, สำเนาทำงาน, สัดส่วนที่ย่อ) หน้าแรกของไฟล์ ภาพ BGR

    ``straighten``: ปรับภาพให้ตรงก่อนวิเคราะห์ (แก้ทิศ 0/90/180/270 + deskew) ใช้ภาพที่ pipeline preprocess เก็บไว้ถ้ามี
    (ผ่านทั้งสองขั้นแล้ว) ไม่งั้นทำเอง และบันทึกมุมเอียงก่อน/หลังไว้ใน ``STATE``
    หมายเหตุ: ตัวจำแนกทิศของ Paddle ในเครื่องนี้โหลด cuDNN ไม่ได้ จึงคืนมุม 0 เงียบ ๆ (ดู warning ของ geometry) ภาพที่ทำเองจะไม่ถูกหมุนทิศ
    """
    raw = load_pages(Path(path), dpi)[0]
    raw_work, _ = _resize_to_work(raw)
    STATE.clear()
    STATE["dataset"] = dataset_of(path)
    STATE["skew_before_deg"] = geometry.estimate_skew(raw_work)
    full, STATE["source"] = raw, "ภาพดิบ (ไม่ปรับ)"
    if straighten:
        cached = load_cached_preprocessed(Path(path).stem)
        if cached is not None:
            full, STATE["source"] = cached, "ภาพที่ pipeline preprocess เก็บไว้ (แก้ทิศ + deskew แล้ว)"
        else:
            full, angle = geometry.deskew(raw)
            STATE["source"] = f"ปรับเอง: deskew {angle:+.1f}° (ไม่ได้แก้ทิศ)"
    work, scale = _resize_to_work(full)
    STATE["skew_after_deg"] = geometry.estimate_skew(work)
    return full, work, scale


def load_cached_preprocessed(stem: str, profile: str = "default", dataset: str | None = None) -> np.ndarray | None:
    """ภาพที่ pipeline preprocess เก็บไว้ (ถ้ามี) ไว้เทียบกับผลของ lab."""
    for p in PATHS.preprocess_final(dataset or STATE["dataset"], profile).rglob(f"{stem}_p1.png"):
        return read_image(p)
    return None


def _show_img(img: np.ndarray) -> np.ndarray:
    s = min(1.0, SHOW / max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def grid(items: dict[str, np.ndarray], cols: int = 3, notes: dict[str, str] | None = None, title: str = "") -> None:
    """แสดงภาพหลายใบเป็นตาราง ``notes`` = ข้อความใต้ชื่อภาพ (เช่น ตัวเลขที่วัดได้)."""
    notes = notes or {}
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 6.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, img) in zip(axes.ravel(), items.items()):
        im = _show_img(img)
        if im.ndim == 3:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name + (f"\n{notes[name]}" if name in notes else ""), fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.show()


def table(rows: dict[str, dict]) -> None:
    import pandas as pd

    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 160):
        from IPython.display import display

        display(pd.DataFrame(rows).T)


# --------------------------------------------------------------------------- 1) คุณลักษณะภาพ
def _ink(gray: np.ndarray) -> tuple[float, np.ndarray]:
    t, inv = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(t), inv


def line_masks(binary_black_on_white: np.ndarray, divisor: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """เส้นตารางแนวนอน/แนวตั้งด้วย morphological opening (ความยาว kernel = ด้านภาพ / ``divisor`` ค่าทดลอง)."""
    ink = 255 - binary_black_on_white
    h, w = ink.shape
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // divisor, 3), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // divisor, 3))))
    return horiz, vert


def features(work: np.ndarray) -> dict[str, float]:
    """ตัวเลขบรรยายภาพ ที่ router น่าจะใช้ตัดสิน (ทุกตัวคำนวณจากภาพล้วน ไม่มีเกณฑ์ตัดสิน)."""
    gray = enhance.to_gray(work)
    h, w = gray.shape
    otsu_t, inv = _ink(gray)
    p5, p95 = np.percentile(gray, [5, 95])
    sat = float(cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[..., 1].mean()) if work.ndim == 3 else 0.0
    noise = float(np.mean(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    touch = (stats[1:, 0] == 0) | (stats[1:, 1] == 0) | (stats[1:, 0] + stats[1:, 2] >= w) | (stats[1:, 1] + stats[1:, 3] >= h)
    bw = bin_ops.otsu(gray)
    horiz, vert = line_masks(bw)
    nh = cv2.connectedComponents(horiz)[0] - 1
    nv = cv2.connectedComponents(vert)[0] - 1
    return {
        "width": w, "height": h, "aspect": round(w / h, 3),
        "background_median": float(np.median(gray)), "contrast_p95_p5": float(p95 - p5),
        "otsu_threshold": otsu_t, "ink_ratio": float((inv > 0).mean()),
        "color_saturation": sat, "noise_median_diff": noise,
        "skew_deg": STATE.get("skew_after_deg", geometry.estimate_skew(work)),
        **({"skew_before_deg": STATE["skew_before_deg"]} if "skew_before_deg" in STATE else {}),
        "components": int(n - 1), "speck_components_le4px": int((areas <= 4).sum()),
        "border_touching_ink_ratio": float(areas[touch].sum() / max(areas.sum(), 1)),
        "hlines": int(nh), "vlines": int(nv),
    }


# --------------------------------------------------------------------------- 2) edge detection
def edges(gray: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """หลายวิธีหา edge (ขาว = edge) + ความหนาแน่น edge ของแต่ละวิธี."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    med = float(np.median(blur))

    def canny(lo: float, hi: float) -> np.ndarray:
        return cv2.Canny(blur, lo, hi)

    def norm_otsu(m: np.ndarray) -> np.ndarray:
        m8 = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.threshold(m8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    out = {
        "canny median-based (0.67·med, 1.33·med)": canny(max(0, 0.67 * med), min(255, 1.33 * med)),
        "canny otsu-based (0.5·T, T)": canny(0.5 * otsu_t, otsu_t),
        "canny 50/150": canny(50, 150),
        "canny 100/200": canny(100, 200),
        "sobel magnitude → otsu": norm_otsu(cv2.magnitude(sx, sy)),
        "laplacian → otsu": norm_otsu(np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))),
        "morph gradient 3×3 → otsu": norm_otsu(cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)),
    }
    stats = {k: {"edge_density": float((v > 0).mean())} for k, v in out.items()}
    return out, stats


# --------------------------------------------------------------------------- 3) threshold
def thresholds(gray: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """หลายวิธี threshold (ตัวอักษรดำ พื้นขาว) + ตัวเลขที่ใช้เทียบ: หมึกกี่ %, กี่ชิ้นส่วน, ชิ้นเล็ก (≤4px) กี่ %."""
    clahe = enhance.clahe(gray)
    out = {
        "otsu (blur 3)": bin_ops.otsu(gray),
        "otsu + CLAHE": bin_ops.otsu(clahe),
        "adaptive gaussian 31/15 (ค่าของ binarize.py)": bin_ops.adaptive(gray),
        "adaptive gaussian 51/10": bin_ops.adaptive(gray, 51, 10),
        "sauvola 25/0.2 (ค่าของ binarize.py)": bin_ops.sauvola(gray),
        "sauvola 51/0.2": bin_ops.sauvola(gray, 51, 0.2),
    }
    stats = {}
    for k, b in out.items():
        ink = 255 - b
        n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        areas = st[1:, cv2.CC_STAT_AREA]
        stats[k] = {"ink_ratio": float((ink > 0).mean()), "components": int(n - 1),
                    "speck_ratio_le4px": float((areas <= 4).mean()) if len(areas) else 0.0}
    return out, stats


# --------------------------------------------------------------------------- 4) contour / section
def text_ink(binary_black_on_white: np.ndarray, drop_lines: bool = True) -> np.ndarray:
    """หมึกสำหรับแบ่ง section: ตัดเส้นตารางยาว ๆ ออกก่อน (``drop_lines``)

    กรอบตารางของฟอร์มเป็นชิ้นเดียวต่อกันทั้งหน้า ถ้าไม่ตัดออก contour ภายนอกจะได้แค่ "กรอบทั้งหน้า" และเส้นตั้งที่พาดทุกแถว
    ทำให้ไม่มีแถวว่างให้ตัดแถบ (เจอจริงกับ 001307 รอบแรกของ lab)
    """
    ink = 255 - binary_black_on_white
    if not drop_lines:
        return ink
    horiz, vert = line_masks(binary_black_on_white)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(lines))


def boxes_by_dilation(binary_black_on_white: np.ndarray, kernel: tuple[int, int], min_area_ratio: float = 0.0002,
                      drop_lines: bool = True) -> list[tuple[int, int, int, int]]:
    """ขยายหมึกด้วย kernel (กว้าง, สูง) แล้วหา contour ภายนอก → กรอบ (x, y, w, h) ``min_area_ratio`` = สัดส่วนพื้นที่หน้าขั้นต่ำ (ทดลอง)."""
    ink = text_ink(binary_black_on_white, drop_lines)
    dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * ink.shape[0] * ink.shape[1]
    boxes = [cv2.boundingRect(c) for c in cnts]
    return sorted((b for b in boxes if b[2] * b[3] >= min_area), key=lambda b: (b[1], b[0]))


def bands_by_gaps(binary_black_on_white: np.ndarray, min_gap: int, empty_ratio: float = 0.002,
                  drop_lines: bool = True) -> list[tuple[int, int]]:
    """ตัดหน้าเป็นแถบแนวนอนตามแถวว่าง (แถวที่มีหมึก < ``empty_ratio`` ของความกว้าง ต่อเนื่อง ≥ ``min_gap`` พิกเซล)."""
    ink = text_ink(binary_black_on_white, drop_lines) > 0
    row = ink.mean(axis=1)
    empty = row < empty_ratio
    bands, start, gap = [], None, 0
    for y, e in enumerate(empty):
        if not e:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                bands.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        bands.append((start, len(empty)))
    return bands


def cells_from_lines(binary_black_on_white: np.ndarray, min_ratio: float = 0.0002, max_ratio: float = 0.5) -> list[tuple[int, int, int, int]]:
    """ช่องของตารางจากเส้นแนวนอน+แนวตั้ง: ส่วนว่างที่ล้อมด้วยเส้น (ขนาดระหว่าง min_ratio–max_ratio ของพื้นที่หน้า ทดลอง)."""
    horiz, vert = line_masks(binary_black_on_white)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    free = cv2.bitwise_not(lines)
    n, _, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    area = lines.shape[0] * lines.shape[1]
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n) if min_ratio * area <= st[i, 4] <= max_ratio * area]


def draw_boxes(img: np.ndarray, boxes, color=(0, 0, 255), width: int = 2) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, width)
    return out


def draw_bands(img: np.ndarray, bands, color=(0, 140, 255)) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y0, y1 in bands:
        cv2.rectangle(out, (0, y0), (out.shape[1] - 1, y1), color, 2)
    return out


# --------------------------------------------------------------------------- 5) แบ่งตามตาราง → ภาพย่อย
def table_regions(binary_black_on_white: np.ndarray, min_area_ratio: float = 0.02) -> list[tuple[int, int, int, int]]:
    """บริเวณตาราง = กลุ่มเส้นแนวนอน+แนวตั้งที่เชื่อมกัน มีทั้งสองทิศ และกรอบใหญ่ ≥ ``min_area_ratio`` ของหน้า (ทดลอง)."""
    horiz, vert = line_masks(binary_black_on_white)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    h_img, w_img = binary_black_on_white.shape
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i, :4])
        box = (slice(y, y + h), slice(x, x + w))
        if w * h >= min_area_ratio * h_img * w_img and horiz[box].any() and vert[box].any():
            out.append((x, y, w, h))
    return sorted(out, key=lambda b: (b[1], b[0]))


def free_boxes(tables: list[tuple[int, int, int, int]], w_img: int, h_img: int) -> list[tuple[int, int, int, int]]:
    """พื้นที่ของหน้าที่อยู่นอกตารางทั้งหมด เป็นกล่องสี่เหลี่ยม (x, y, w, h) — รวมข้อความ **ข้างตาราง** (ซ้าย/ขวา) ด้วย

    ตัดหน้าเป็นแถบแนวนอนที่ขอบบน/ล่างของตารางทุกก้อน แต่ละแถบเอาช่วงแนวนอนที่ไม่ถูกตารางบัง แล้วต่อแถบที่ติดกันและช่วงเดียวกันเข้าด้วยกัน
    (ไม่มีตารางเลย = ทั้งหน้าเป็นกล่องเดียว) ผู้เรียกตัดสินเองว่ากล่องไหนมีหมึก: นอกตารางไม่ได้แปลว่าไม่ใช่ข้อมูล
    """
    ys = sorted({0, h_img, *(min(max(v, 0), h_img) for t in tables for v in (t[1], t[1] + t[3]))})
    open_boxes: dict[tuple[int, int], list[int]] = {}
    out: list[tuple[int, int, int, int]] = []
    for y0, y1 in zip(ys, ys[1:]):
        spans = sorted((t[0], t[0] + t[2]) for t in tables if t[1] < y1 and t[1] + t[3] > y0)
        free, cur = [], 0
        for a, b in spans:
            if a > cur:
                free.append((cur, a))
            cur = max(cur, b)
        if cur < w_img:
            free.append((cur, w_img))
        carried: dict[tuple[int, int], list[int]] = {}
        for iv in free:
            if iv in open_boxes and open_boxes[iv][1] == y0:
                carried[iv] = open_boxes.pop(iv)
                carried[iv][1] = y1
            else:
                carried[iv] = [y0, y1]
        out += [(iv[0], r[0], iv[1] - iv[0], r[1] - r[0]) for iv, r in open_boxes.items()]
        open_boxes = carried
    out += [(iv[0], r[0], iv[1] - iv[0], r[1] - r[0]) for iv, r in open_boxes.items()]
    return out


FRAME_INSIDE = 0.9  # ส่วนของตารางในที่อยู่ในกรอบอย่างน้อยเท่านี้จึงนับว่า "อยู่ข้างใน" (ค่าทดลอง)
MAX_PIECE_FRAC = 0.3  # ส่วนที่สูงกว่านี้ (สัดส่วนของความสูงหน้า) ถูกตัดเป็นกลุ่มแถว วัดบน 9 ใบ (OpenThai luma, rules ล้วน ไม่มีเฉลย): 0.5 ขาด 9 ช่อง · 0.3 ขาด 7 (ดีสุด กู้วันที่ 1310, วันที่/เลขที่ 1320 แต่เสีย invoice_no 1317) · 0.2 ขาด 9 และ 1307 ตก failed (ท่อนเล็กเกิน)


def drop_frames(tables: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """ตัดตารางที่เป็น "กรอบครอบ" ออก: กรอบของฟอร์มทั้งใบเชื่อมเส้นทุกอย่างเป็นก้อนเดียว ถ้าเก็บไว้จะกลายเป็นทั้งหน้าซ้ำกับตารางข้างใน

    ตารางที่มีตารางอื่นอยู่ข้างในถือเป็นกรอบ (ข้อมูล 18 ใบ: ตารางซ้อนกัน 4 คู่ เป็นกรอบฟอร์มทั้งหมด พื้นที่ตารางในเป็น 0.73–0.89 ของกรอบ
    ไม่มีกรณีตารางจริงที่มีตารางย่อย — เอกสารชนิดอื่นอาจต่าง: ถ้ามี หัวคอลัมน์จะถูกแยกจากแถวภายในกรอบนั้น)
    ตารางที่ไม่มีตารางอื่นข้างในไม่ถูกแตะ ส่วนนอกตารางที่เหลือเป็นส่วนข้อความตามปกติ (``free_boxes``)
    """
    def inside(inner, outer) -> bool:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        w = min(ix + iw, ox + ow) - max(ix, ox)
        h = min(iy + ih, oy + oh) - max(iy, oy)
        return iw * ih < ow * oh and max(w, 0) * max(h, 0) >= FRAME_INSIDE * iw * ih

    return [t for t in tables if not any(u is not t and inside(u, t) for u in tables)]


def split_rows(ink: np.ndarray, box: tuple[int, int, int, int], max_h: float,
               empty_ratio: float = 0.002) -> list[tuple[int, int, int, int]]:
    """ตัดกล่องที่สูงกว่า ``max_h`` เป็นกลุ่มแถว (ท่อนติดกัน ไม่ซ้อนกัน) ที่ **แถวว่างระหว่างบรรทัด** ไม่ตัดกลางบรรทัด

    จำนวนท่อน = ceil(สูง / max_h) ตัดที่ช่องว่างที่ใกล้ตำแหน่งแบ่งเท่า ๆ กันที่สุด (ตรงกลางช่วงแถวว่าง) ``ink`` = หมึกของหน้า (ตัดเส้นตารางออกแล้ว)
    ช่วงแถวว่าง = แถวที่หมึก < ``empty_ratio`` ของความกว้าง (ค่าเดียวกับ ``bands_by_gaps``) ไม่เหลือช่องว่างให้ตัดแล้วก็หยุดตัด (ท่อนสูงกว่าเกณฑ์ได้ ไม่ตัดผ่ากลางบรรทัด)
    """
    x, y, w, h = box
    n = int(np.ceil(h / max_h)) if max_h > 0 else 1
    if n <= 1:
        return [box]
    empty = (ink[y:y + h, x:x + w] > 0).mean(axis=1) < empty_ratio
    runs, start = [], None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            runs.append((start + i) // 2)
            start = None
    if start is not None:
        runs.append((start + len(empty)) // 2)
    cuts: list[int] = []
    for k in range(1, n):
        ideal = h * k / n
        pool = [r for r in runs if (not cuts or r > cuts[-1]) and 0 < r < h]
        if not pool:  # ไม่มีช่องว่างเหลือให้ตัด: ปล่อยท่อนสูงกว่าเกณฑ์ ดีกว่าตัดผ่ากลางบรรทัด
            break
        cuts.append(int(min(pool, key=lambda r: abs(r - ideal))))
    edges = [0, *sorted(set(cuts)), h]
    return [(x, y + a, w, b - a) for a, b in zip(edges, edges[1:]) if b > a]


RULE_COVER = 0.8  # แถวพิกเซลที่เป็นเส้นแนวนอนอย่างน้อยเท่านี้ของความกว้างตารางจึงนับเป็น "เส้นแบ่งแถว" (ค่าทดลอง)
EMPTY_ROW = 0.002  # แถวที่หมึก < เท่านี้ของความกว้างหน้าถือว่าว่าง (ค่าเดียวกับ ``split_rows`` / ``bands_by_gaps``)


def table_rules(horiz: np.ndarray, table: tuple[int, int, int, int], min_cover: float = RULE_COVER) -> list[int]:
    """ตำแหน่ง y (พิกัดหน้า) ของเส้นแนวนอนยาวเต็มตาราง (เส้นแบ่งแถว) ``horiz`` = ผลของ ``line_masks``"""
    x, y, w, h = table
    cov = (horiz[y:y + h, x:x + w] > 0).mean(axis=1)
    groups: list[list[int]] = []
    for v in np.where(cov >= min_cover)[0]:
        if groups and v - groups[-1][-1] <= 3:
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    return [y + int(round(float(np.mean(g)))) for g in groups]


def band_cuts(ink: np.ndarray, candidates: set[int]) -> list[int]:
    """จุดตัดแนวนอนที่ใช้ได้จริง: ขอบหน้า + ตำแหน่งผู้สมัครที่ **แถวนั้นว่างตลอดความกว้างหน้า** (ไม่ตัดผ่านข้อความข้างตาราง)"""
    h_img = ink.shape[0]
    empty = (ink > 0).mean(axis=1) < EMPTY_ROW
    ok = {0, h_img} | {y for y in candidates if 0 < y < h_img and empty[y]}
    return sorted(ok)


def layout_sections(binary_black_on_white: np.ndarray, min_table_ratio: float = 0.02,
                    min_component_px: int = 8, table_mode: str = "rows",
                    max_piece_frac: float | None = MAX_PIECE_FRAC) -> list[dict]:
    """แบ่งหน้าเป็นภาพย่อยตามตาราง แล้วเรียงตามลำดับอ่าน (บนลงล่าง ซ้ายไปขวา)

    - ``table_mode="rows"`` (ค่าตั้งต้น): **แถบแนวนอนเต็มความกว้างหน้า** ตัดที่ขอบตารางและเส้นแนวนอนยาวเต็มตาราง (``table_rules``) เฉพาะแถวที่ **ว่างตลอดความกว้างหน้า**
      (``band_cuts``) ไม่ตัดผ่านข้อความข้างตาราง ป้ายข้างตาราง (เช่น ป้ายยอดรวมทางซ้ายของกล่องตัวเลข) จึงอยู่ภาพเดียวกับค่าของแถวนั้น
      ``"bands"`` ตัดเฉพาะที่ขอบตาราง (วัดแล้วแย่กว่า: ป้าย–ค่าอยู่บรรทัดเดียวกัน 10/22 เทียบ rows 18/22)
      ผลข้างเคียงที่รู้: แถบที่มีแบนเนอร์ใหญ่กับกล่องเล็กอยู่ข้างกัน OpenThai อาจหลอน (1312) และแถบหัวคอลัมน์ถูกตัดแยกบาง ๆ จากแถวรายการ
    - ``table_mode="whole"`` เก็บ **ทั้งก้อนเป็นส่วนเดียว** หัวคอลัมน์กับค่าในแถวจึงอยู่ภาพเดียวกัน (ป้ายข้างตารางเป็นอีกภาพ)
      ตัวอ่านที่รู้จักตารางคืนโครงสร้างแถว–คอลัมน์ให้ ไม่ต้องประกอบกลับจากตำแหน่ง
      กรอบของฟอร์มที่ครอบตารางอื่นถูกตัดออกก่อน (``drop_frames``) ไม่งั้นกลายเป็นทั้งหน้า
      - ส่วนที่สูงกว่า ``max_piece_frac`` ของหน้า (ตารางหรือข้อความ) ถูกตัดเป็นกลุ่มแถวที่แถวว่างระหว่างบรรทัด (``split_rows``) แต่ละท่อนมี
      ``part``/``parts`` บอกลำดับ ``None`` = ไม่ตัด ท่อนหลัง ๆ ไม่มีหัวคอลัมน์ของตารางเดิมซ้ำ (ยังไม่ได้ทำ)
    ``table_mode="cells"`` แตกเป็น **ช่อง (cell)** ตามเส้น (ไม่เจอช่องเลยเก็บทั้งตารางเป็นส่วนเดียว) ช่องแยกหัวออกจากค่า
    - นอกตาราง → **ส่วนข้อความ (text)** ทุกบริเวณที่ไม่ถูกตารางบัง รวมข้อความ **ข้างตาราง** ด้วย (``free_boxes``) ไม่ใช่แค่แถบคั่นระหว่างตาราง
      เพราะป้ายของช่องยอดรวมมักอยู่ซ้ายของตารางตัวเลข ส่วนที่ไม่มีหมึกเลยจะติดธง ``keep=False``
    - ส่วนที่ **ไม่มีชิ้นหมึกเลย** (นับหลังตัดเส้นตารางออก ไม่นับชิ้นเล็กกว่า ``min_component_px`` พิกเซลที่เป็นจุดรบกวน)
      ติดธง ``keep=False`` (ตัดทิ้ง) ใช้จำนวนชิ้น ไม่ใช้สัดส่วนหมึกต่อพื้นที่ เพราะช่องสูงยาวที่มีตัวเลขสองตัว (เช่น คอลัมน์จำนวน)
      จะดูเหมือนว่างและถูกตัดทิ้งผิด (เจอจริงกับ 1309 รอบแรก)
    ค่า ``min_table_ratio`` / ``min_component_px`` เป็นค่าทดลอง ดูตารางสรุปประกอบภาพก่อนเชื่อ
    """
    if table_mode not in ("whole", "cells", "bands", "rows"):
        raise ValueError(f"table_mode ต้องเป็น 'whole', 'cells', 'bands' หรือ 'rows': {table_mode!r}")
    h_img, w_img = binary_black_on_white.shape
    ink = text_ink(binary_black_on_white)
    tables = table_regions(binary_black_on_white, min_table_ratio)
    if table_mode != "cells":
        tables = drop_frames(tables)  # โหมดช่องแตกกรอบเป็นช่องอยู่แล้ว
    secs: list[dict] = []

    def add(kind: str, box: tuple[int, int, int, int], table: int | None, part: tuple[int, int] | None = None) -> None:
        pieces = split_rows(ink, box, max_piece_frac * h_img) if max_piece_frac and part is None else [box]
        if len(pieces) > 1:
            for k, piece in enumerate(pieces, start=1):
                add(kind, piece, table, (k, len(pieces)))
            return
        x, y, w, h = box
        crop = ink[y:y + h, x:x + w]
        ratio = float((crop > 0).mean()) if crop.size else 0.0
        n, _, st, _ = cv2.connectedComponentsWithStats(crop, connectivity=8) if crop.size else (1, None, np.zeros((1, 5), int), None)
        comps = int((st[1:, cv2.CC_STAT_AREA] >= min_component_px).sum())
        secs.append({"kind": kind, "bbox": [x, y, w, h], "table": table, "ink_ratio": ratio, "components": comps, "keep": comps > 0,
                     **({"part": part[0], "parts": part[1]} if part else {})})

    if table_mode in ("bands", "rows"):
        # แถบแนวนอนเต็มความกว้าง: ป้ายข้างตารางอยู่ภาพเดียวกับตัวเลขของแถวนั้นเสมอ (ไม่แยกซ้าย/ขวาเป็นคนละภาพ)
        cands: set[int] = set()
        horiz = line_masks(binary_black_on_white)[0] if table_mode == "rows" else None
        for tx, ty, tw, th in tables:
            cands |= {ty, ty + th}
            if horiz is not None:
                cands |= set(table_rules(horiz, (tx, ty, tw, th)))
        cuts = band_cuts(ink, cands)
        for y0, y1 in zip(cuts, cuts[1:]):
            over = [max(0, min(y1, ty + th) - max(y0, ty)) for tx, ty, tw, th in tables]
            best = int(np.argmax(over)) if over else None
            is_table = best is not None and over[best] >= 0.5 * (y1 - y0)
            add("table" if is_table else "text", (0, y0, w_img, y1 - y0), best if is_table else None)
        tables = []  # ใช้หมดแล้ว ข้ามเส้นทางตารางก้อน/ช่อง และ free_boxes (แถบคลุมทั้งหน้าอยู่แล้ว)
        free = []
    else:
        free = free_boxes([t for t in tables], w_img, h_img)
    for t, (tx, ty, tw, th) in enumerate(tables):
        cells = cells_from_lines(binary_black_on_white[ty:ty + th, tx:tx + tw]) if table_mode == "cells" else []
        for cx, cy, cw, ch in cells:
            add("cell", (tx + cx, ty + cy, cw, ch), t)
        if not cells:
            add("table", (tx, ty, tw, th), t)
    for x, y, w, h in free:
        add("text", (x, y, w, h), None)
    secs.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    for i, s in enumerate(secs):
        s["id"] = f"s{i:02d}"
    return secs


def draw_sections(img: np.ndarray, secs: list[dict]) -> np.ndarray:
    """เขียว = ช่องตารางที่เก็บ, ส้ม = ข้อความนอกตาราง, น้ำเงิน = ตารางทั้งก้อน, แดงหนา = ว่าง (ตัดทิ้ง)."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = {"cell": (0, 160, 0), "text": (0, 140, 255), "table": (200, 90, 0)}
    for s in secs:
        x, y, w, h = s["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), colors[s["kind"]] if s["keep"] else (0, 0, 220), 2 if s["keep"] else 4)
        cv2.putText(out, s["id"][1:], (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 0, 90), 1, cv2.LINE_AA)
    return out


def crop_full(full: np.ndarray, work: np.ndarray, sec: dict) -> np.ndarray:
    """ตัดส่วนนั้นจากภาพเต็ม (พิกัดของ section อยู่บนสำเนาทำงาน จึงคูณสัดส่วนกลับ)."""
    fx, fy = full.shape[1] / work.shape[1], full.shape[0] / work.shape[0]
    x, y, w, h = sec["bbox"]
    return full[int(y * fy):int((y + h) * fy), int(x * fx):int((x + w) * fx)]


def variants(crop: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """ภาพย่อยหนึ่งส่วนก่อนเข้า OCR: สีเดิม / เทา / ลด noise สามแบบ / CLAHE + ตัวเลขที่วัดได้
    (noise = ค่าเฉลี่ยผลต่างจาก median 3×3, contrast = p95−p5, ink = สัดส่วนหมึกตาม Otsu)
    ตัวเลขบอกลักษณะภาพ ไม่ได้บอกว่า OCR อ่านดีขึ้น (ต้องวัดด้วยการอ่านจริงอีกขั้น)."""
    gray = enhance.to_gray(crop)
    imgs = {
        "สีเดิม": crop, "เทา": gray,
        "เทา + median": enhance.denoise(gray, "median"),
        "เทา + nlmeans": enhance.denoise(gray, "nlmeans"),
        "เทา + bilateral": enhance.denoise(gray, "bilateral"),
        "เทา + CLAHE": enhance.clahe(gray),
    }
    stats = {}
    for k, im in imgs.items():
        g = enhance.to_gray(im)
        p5, p95 = np.percentile(g, [5, 95])
        stats[k] = {
            "noise": float(np.mean(np.abs(g.astype(np.int16) - cv2.medianBlur(g, 3).astype(np.int16)))),
            "contrast": float(p95 - p5), "ink": float((_ink(g)[1] > 0).mean()),
        }
    return imgs, stats


def section_features_dir() -> Path:
    if "dataset" not in STATE:
        raise RuntimeError("เรียก L.load(...) ก่อน เพื่อให้รู้ว่าเป็นชุดข้อมูลไหน")
    return PATHS.section_features(STATE["dataset"])


def gray_value_channel(crop: np.ndarray) -> np.ndarray:
    """เทาแบบ "ช่องสว่างสุด" (max ของ B,G,R = ช่อง V ของ HSV) แทนสูตร luma

    หมึกดำมืดทุกช่อง จึงยังมืด ส่วนสีที่ไม่ใช่ข้อความ (ลายน้ำฟ้า, ตราประทับแดง/น้ำเงิน, แถบสี, เส้นตารางสี) จะมีอย่างน้อยหนึ่งช่องสว่าง
    เมื่อเอาค่าสว่างสุดมา จึงกลายเป็นเกือบขาว (luma จะถ่วงสีให้เทาเข้มและยังเห็นอยู่)
    """
    return crop.max(axis=2) if crop.ndim == 3 else crop


def flatten_background(gray: np.ndarray, frac: float = 0.04) -> np.ndarray:
    """ทำพื้นหลังให้ขาวเรียบ: หารด้วยภาพพื้นหลังที่ประมาณจาก morphological closing (ลบเงา/สีพื้น) แล้วยืดคอนทราสต์
    ``frac`` = ขนาด kernel ต่อด้านสั้นของภาพ (ทดลอง ต้องใหญ่กว่าตัวอักษรเพื่อให้ closing ลบตัวอักษรออกจากภาพพื้นหลัง)."""
    k = max(15, int(min(gray.shape[:2]) * frac) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    norm = cv2.divide(gray, bg, scale=255)
    lo = np.percentile(norm, 1)
    return np.clip((norm.astype(np.float32) - lo) * (255.0 / max(255.0 - lo, 1.0)), 0, 255).astype(np.uint8)


BG_SCALE = 4  # ย่อภาพก่อนประมาณสีพื้น (ประหยัดเวลา median) ค่าทดลอง
BG_KERNEL = 31  # หน้าต่าง median บนภาพย่อ (= 124 px บนภาพจริง) ต้องกว้างกว่าเส้นหนาสุดของตัวอักษรหัวเรื่อง (~30 px)
BIN_FLOOR = 28  # ระยะสีขั้นต่ำ (Lab) ที่ถือว่าต่างจากพื้น กันรอยยับของกระดาษเมื่อ Otsu ได้ค่าต่ำ ค่าทดลอง (Otsu จริงบน 1310 ได้ 86–102)
MIN_RULE_H = 80  # เส้นตั้งของตารางต้องยาวอย่างน้อยเท่านี้ (พิกเซล ≈ 3 บรรทัด ค่าทดลอง) บน 1307 ความสูงแถว ≈ 50 px
MIN_RULES = 4  # ช่วงแนวตั้งที่นับเป็น "ตารางมีช่องค่า" ต้องมีเส้นตั้งพร้อมกันอย่างน้อยเท่านี้ (ซ้าย, ป้าย|ค่า, คอลัมน์ค่าอีก ≥1, ขวา) กันกรอบของกล่องข้อความ
MAX_GLYPH_H = 40  # ชิ้นหมึกที่สูงกว่านี้ไม่ใช่ตัวอักษร (เศษเส้นตั้งที่เอียงจนไม่ถูกจับเป็นเส้น) วัดบน 1307: ตัวอักษรสูงสุด 30 px เศษเส้น 56–181 px
MIN_GLYPH_AREA = 60  # ชิ้นหมึกที่เล็กกว่านี้ (พิกเซล²) หรือบางกว่า ``MIN_GLYPH_W`` คือเศษเส้นตาราง วัดบน 1307: เศษเส้น กว้าง 1–2 px หรือ 44 px² ส่วนสระ/วรรณยุกต์เล็กสุด 86 px²
MIN_GLYPH_W = 3
DOT_MAX_H = 10  # ชิ้นหมึกที่เตี้ยกว่านี้ (พิกเซลของภาพ section) = จุดไข่ปลา/จุดทศนิยม ไม่นับเป็นข้อความ วัดบน 1307: จุด 4–5 px ตัวอักษร p75 = 21 px


def binarize_inverted(crop: np.ndarray) -> np.ndarray:
    """ขาว-ดำ แบบ **ตัวอักษรขาว กระดาษดำ** ที่ไม่ขึ้นกับสีพื้น/สีตัวอักษร

    ตัดสินจากระยะสี (Lab) ของแต่ละพิกเซลจากสีพื้นรอบข้าง (median หน้าต่างใหญ่) ไม่ใช่จากความสว่าง: ตัวดำบนกระดาษขาว ตัวขาวบนแถบสีเขียว/น้ำเงิน
    และตัวสีบนพื้นสี ได้ผลแบบเดียวกัน (วัดดูด้วยตาบน 1306 s07/s20, 1310, 1312 s02, 1317 s03, 1318 s03, 1319 s02) จุดอ่อน: ขอบของแถบสีที่เป็นมุมโค้งเกิดรอยเส้นบาง ๆ
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    small = cv2.resize(lab, None, fx=1 / BG_SCALE, fy=1 / BG_SCALE, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, BG_KERNEL), (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
    dist = np.linalg.norm(lab.astype(np.float32) - bg.astype(np.float32), axis=2)
    otsu, _ = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((dist > max(otsu, BIN_FLOOR)) * 255).astype(np.uint8)


def _runs(mask_1d: np.ndarray, gap: int = 0) -> list[tuple[int, int]]:
    """ช่วง [a, b) ที่ mask เป็นจริงต่อเนื่อง (ช่องว่างสั้นกว่าหรือเท่ากับ ``gap`` ถือว่าต่อกัน)."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def table_columns(b: np.ndarray) -> dict | None:
    """โครงคอลัมน์ของตารางในภาพ ``binarize_inverted`` จากเส้นตั้ง: ``ruled`` = ช่วงแนวตั้งที่มีเส้นตั้งพร้อมกัน ≥ ``MIN_RULES`` เส้น
    ``boundary`` = ขอบขวาของเซลล์ที่กว้างที่สุด (= คอลัมน์ป้าย) ขวามือคือช่องค่า ไม่เจอตาราง = None."""
    h, _w = b.shape
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, MIN_RULE_H)))
    ruled = np.array([len(_runs(vert[y] > 0, gap=3)) >= MIN_RULES for y in range(h)])
    if not ruled.any():
        return None
    y0, y1 = int(np.flatnonzero(ruled)[0]), int(np.flatnonzero(ruled)[-1]) + 1
    xs = [int(round((a + z - 1) / 2)) for a, z in _runs((vert[y0:y1] > 0).any(axis=0), gap=3)]
    widest = int(np.argmax(np.diff(xs)))
    return {"ruled_y": [y0, y1], "boundary": xs[widest + 1], "left": xs[widest]}


def drop_empty_rows(b: np.ndarray, boundary: int, left: int = 0, span: tuple[int, int] | None = None) -> tuple[np.ndarray, list[list[int]]]:
    """ทาดำแถวที่ **ช่องค่า (ขวาของ ``boundary``) ไม่มีตัวอักษร** เหลือแถวที่มีค่ากับบรรทัดป้ายที่ซ้อนทับ ภายในช่วงแนวตั้ง ``span`` (ค่าตั้งต้น = ทั้งภาพ)
    ตัวอักษรคือชิ้นหมึกที่ไม่ใช่จุดไข่ปลา/เศษเส้น (ดู ``DOT_MAX_H`` ``MAX_GLYPH_H`` ``MIN_GLYPH_*``) ใช้ภาพล้วน ไม่ใช้ OCR
    คืน (ภาพใหม่ รูปทรงเดิม, ช่วงแนวตั้ง [a, b] ที่เก็บไว้)."""
    h, w = b.shape
    y0, y1 = span or (0, h)
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, MIN_RULE_H)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_and(b, cv2.bitwise_not(cv2.dilate(vert, np.ones((1, 7), np.uint8)))))
    text = np.zeros_like(b)
    for i in range(1, n):
        _x, _y, cw, ch, area = st[i]
        if DOT_MAX_H <= ch <= MAX_GLYPH_H and MIN_GLYPH_W <= cw < w // 3 and area >= MIN_GLYPH_AREA:
            text[lab == i] = 255
    val = (text[:, boundary + 3:] > 0).any(axis=1)
    lbl = (text[:, left + 3:boundary - 3] > 0).any(axis=1)
    keep = np.zeros(h, bool)
    for a, z in _runs(val, gap=DOT_MAX_H):
        keep[a:z] = True
        for la, lz in _runs(lbl, gap=DOT_MAX_H // 2):
            if la < z and lz > a:
                keep[la:lz] = True
    keep = cv2.dilate(keep.astype(np.uint8)[:, None], np.ones((2 * DOT_MAX_H + 1, 1), np.uint8))[:, 0] > 0
    out = b.copy()
    drop = ~keep
    drop[:y0] = False
    drop[y1:] = False
    out[drop] = 0
    return out, [list(r) for r in _runs(keep[y0:y1])]


def drop_empty_rows_by_table(items: list[tuple[dict, np.ndarray]]) -> list[tuple[np.ndarray, dict]]:
    """ใช้ ``drop_empty_rows`` กับทุกส่วนของตารางเดียวกัน (``sec['table']`` เท่ากัน) ด้วยคอลัมน์ป้ายชุดเดียวกัน
    ส่วนที่ถูกตัดจากตารางมักเห็นเส้นตั้งไม่ครบ (แถวยอดรวมที่ช่องรวมกัน) จึงยืมขอบคอลัมน์จากส่วนที่เห็นโครงตารางชัดสุด (ช่วงเส้นตั้งยาวสุด)
    ส่วนข้อความ (ไม่มีตาราง) ไม่แตะ ตารางที่ไม่มีค่าเลยสักส่วนก็ไม่แตะ (ไม่ใช่ฟอร์มที่เข้าใจ) ภาพที่ทาดำหมดคืน info['blank'] = True."""
    out = [(b, {"applied": False}) for _s, b in items]
    for table in {s["table"] for s, _ in items if s["table"] is not None}:
        members = [i for i, (s, _b) in enumerate(items) if s["table"] == table]
        cols = {i: table_columns(items[i][1]) for i in members}
        found = {i: c for i, c in cols.items() if c}
        if not found:
            continue
        ref = max(found.values(), key=lambda c: c["ruled_y"][1] - c["ruled_y"][0])
        done = {}
        for i in members:
            span = tuple(cols[i]["ruled_y"]) if cols[i] else None
            done[i] = drop_empty_rows(items[i][1], ref["boundary"], ref["left"], span)
        if not any(kept for _o, kept in done.values()):
            continue
        for i, (img, kept) in done.items():
            out[i] = (img, {"applied": True, "label_right_x": ref["boundary"], "kept_rows": kept, "blank": not img.any()})
    return out


def erase_table_lines(gray: np.ndarray) -> np.ndarray:
    """ลบเส้นตารางยาว ๆ (แนวนอน/แนวตั้ง) ออกจากภาพเทา โดยทาทับด้วยสีขาว — เส้นที่ติดขอบภาพย่อยเกิดจากการตัดตามช่องตาราง."""
    bw = bin_ops.otsu(gray)
    horiz, vert = line_masks(bw)
    mask = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8))
    out = gray.copy()
    out[mask > 0] = 255
    return out


def clean_variants(crop: np.ndarray) -> dict[str, np.ndarray]:
    """วิธีเตรียมภาพย่อยก่อนเข้า OCR ที่จะเทียบกัน (ผลทั้งหมดเป็นภาพเทา 1 ช่อง ยกเว้น 'สีเดิม')."""
    luma = enhance.to_gray(crop)
    v = gray_value_channel(crop)
    flat = flatten_background(v)
    return {
        "สีเดิม": crop,
        "เทา luma": luma,
        "เทา V (กำจัดสี)": v,
        "luma + พื้นหลังเรียบ": flatten_background(luma),
        "V + พื้นหลังเรียบ": flat,
        "V + พื้นหลังเรียบ + median": cv2.medianBlur(flat, 3),
        "V + พื้นหลังเรียบ + ลบเส้นตาราง": erase_table_lines(flat),
    }


def section_manifest(secs: list[dict]) -> list[dict]:
    """ผังแบ่ง section สำหรับเก็บลง JSON ภาพย่อย **ไม่เก็บเป็นไฟล์** (ซ้ำกับภาพทั้งหน้า 200MB) ตัดจากภาพทั้งหน้าตอนใช้ด้วย
    ``iter_section_crops`` พิกัดใน manifest ทำให้ได้ภาพเดียวกันทุกครั้ง."""
    return [dict(s) for s in secs]


def iter_section_crops(stem: str, dataset: str = "finance_doc_scan", keep_only: bool = True):
    """(section, ภาพ BGR ความละเอียดเต็ม) ของทุกส่วนตามผังใน features ตัดจากภาพทั้งหน้าที่ pipeline preprocess เก็บไว้ (ตรงแล้ว สี)."""
    full = load_cached_preprocessed(stem, dataset=dataset)
    if full is None:
        raise FileNotFoundError(f"ไม่พบภาพที่ preprocess แล้วของ {stem} ใน {PATHS.preprocess_final(dataset)}")
    work, _ = _resize_to_work(full)
    layout = json.loads((PATHS.section_features(dataset) / f"{stem}.json").read_text(encoding="utf-8"))["layout_sections"]
    for s in layout:
        if keep_only and not s["keep"]:
            continue
        yield s, crop_full(full, work, s)


# --------------------------------------------------------------------------- บันทึก / เทียบข้ามไฟล์
def save(stem: str, **sections) -> Path:
    """รวมผลของไฟล์นี้ลง JSON (เขียนทับส่วนที่ส่งมา คงส่วนอื่นไว้) — ให้ router / สคริปต์สรุปอ่านต่อ."""
    d = section_features_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data.update(sections)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def compare_with_others(stem: str, keys: list[str] | None = None) -> None:
    """ค่า features ของไฟล์นี้เทียบกับไฟล์อื่นที่รัน notebook ไปแล้ว (อันดับเป็นเปอร์เซ็นไทล์ใน 100)."""
    import pandas as pd
    from IPython.display import display

    rows = {}
    for p in sorted(section_features_dir().glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "features" in d:
            rows[p.stem] = d["features"]
    if stem not in rows:
        print("ยังไม่มี features ของไฟล์นี้ ให้รันเซลล์ก่อนหน้า")
        return
    df = pd.DataFrame(rows).T
    cols = keys or list(df.columns)
    pct = df[cols].rank(pct=True) * 100
    view = pd.DataFrame({"ไฟล์นี้": df.loc[stem, cols], "มัธยฐานทุกไฟล์": df[cols].median(), "เปอร์เซ็นไทล์": pct.loc[stem]})
    print(f"เทียบกับ {len(df)} ไฟล์ที่มีผลแล้ว")
    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 160):
        display(view)
