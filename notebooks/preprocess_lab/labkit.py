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
from fin_extractor.io import load_pages  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of  # noqa: E402
from fin_extractor.preprocessing import binarize as bin_ops  # noqa: E402
from fin_extractor.preprocessing import enhance, geometry  # noqa: E402

from matplotlib import font_manager  # noqa: E402

# โค้ดแบ่ง section / เตรียมภาพรายส่วนอยู่ใน package (fin_extractor.sections) ที่นี่ส่งต่อให้ notebook ใช้ชื่อเดิมได้
from fin_extractor.sections import crops as _crops  # noqa: E402
from fin_extractor.sections.images import (  # noqa: E402,F401
    BG_KERNEL, BG_SCALE, BIN_FLOOR, DOT_MAX_H, MAX_GLYPH_H, MIN_GLYPH_AREA, MIN_GLYPH_W, MIN_RULE_H, MIN_RULES, _runs, binarize_inverted,
    drop_empty_rows, drop_empty_rows_by_table, flatten_background, gray_value_channel, table_columns)
from fin_extractor.sections.layout import (  # noqa: E402,F401
    EMPTY_ROW, FRAME_INSIDE, MAX_PIECE_FRAC, RULE_COVER, WORK, band_cuts, cells_from_lines, crop_full, draw_sections, drop_frames, free_boxes,
    layout_sections, line_masks, split_rows, table_regions, table_rules, text_ink)
from fin_extractor.sections.layout import resize_to_work as _resize_to_work  # noqa: E402

# ชื่อภาพภาษาไทยในกราฟ: font ตั้งต้นของ matplotlib ไม่มีอักษรไทย ใช้ Sarabun ที่มากับโปรเจกต์ (assets/fonts, SIL OFL)
for _f in (ROOT / "assets" / "fonts").glob("Sarabun-*.ttf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams["font.family"] = ["Sarabun", "DejaVu Sans"]

PATHS = DataPaths(load_config())  # พาธข้อมูลถามจาก paths.py ที่เดียว ไม่ฝังเอง
SHOW = 760  # ด้านยาวของภาพที่แสดงใน notebook (ย่อให้ไฟล์ไม่บวม)

# ค่ากวาด (ทดลอง): ขนาด kernel ขยายหมึก (กว้าง, สูง) ที่ WORK=1600 ไล่จากระดับตัวอักษร → คำ → บรรทัด → บล็อก → ส่วนใหญ่
KERNELS = [(9, 3), (25, 5), (60, 9), (140, 20), (300, 45)]
BAND_GAPS = [8, 16, 32]  # ความสูงขั้นต่ำของแถวว่างที่ถือเป็นรอยตัดแนวนอน (พิกเซล)


# --------------------------------------------------------------------------- โหลด / แสดงผล
STATE: dict = {}  # ผลของ load() ครั้งล่าสุด (ชุดข้อมูล, ที่มาของภาพ, มุมเอียงก่อน/หลังปรับ) ใช้ต่อใน features() และตอนบันทึก




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
    return _crops.load_final_image(PATHS, dataset or STATE["dataset"], stem, profile)


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
    """(section, ภาพ BGR ความละเอียดเต็ม) ของทุกส่วนตามผังที่เก็บไว้ (ตัวจริงอยู่ที่ fin_extractor.sections.crops ที่นี่ใช้ PATHS ของ lab)."""
    return _crops.iter_section_crops(PATHS, stem, dataset, keep_only)


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
