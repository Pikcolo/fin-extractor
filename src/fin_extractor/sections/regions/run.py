"""ลำดับงานของโหมด regions ต่อหนึ่งหน้า แล้วเขียนภาพตรวจ + crop + manifest.json (ลำดับเดียวกับ pipeline/runner.py ของ preprocess_finance)

    normalize → mask → clean → regions → tables → export

รับภาพ final ของขั้น 2 (ตั้งตรง แก้เอียงแล้ว สี render 300 dpi) ค่า px ของ configs/regions/ วัดที่ 200 dpi
จึงตรวจทุกอย่างบนสำเนาที่ย่อเป็น ``ref_dpi`` (แบบเดียวกับสำเนาทำงาน WORK ของผัง section เดิม) แล้วตัดภาพจริงจากภาพ final สีความละเอียดเต็ม

ผลที่ ``03_sections/<dataset>/<profile>/regions/<ไฟล์>/`` (ล้างก่อนเขียนทุกครั้ง ไฟล์จึงตรงกับ manifest เสมอ)
    01_normalize.png … 05_tables.png         ภาพแต่ละขั้น (บนสำเนา ref_dpi) 02/03 มีเฉพาะเมื่อขั้นนั้นทำอะไรจริง
    crops/<ลำดับ>_<region>[_tNN]__<วิธี>.png  ภาพแต่ละ region (ท่อน tNN เมื่อสูงเกิน max_piece_frac)
    crops/tables/t<T>_r<RR>_c<CC>__<วิธี>.png ช่องตาราง r00 = หัวตาราง (key)
    manifest.json                            ทุกค่าที่ตรวจได้ + รายการไฟล์ (กล่องทั้งพิกัดสำเนาและพิกัดภาพ final) + kv_pairs + QR ที่ถอดได้
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ...config import PROJECT_ROOT
from ...preprocessing import clean as clean_ops
from ...preprocessing import enhance
from .. import images as I
from ..layout import split_rows
from . import codes, doc_types, draw, geometry, lines, split, tables
from .masks import ink_mask, text_mask

CONFIG_DIR = PROJECT_ROOT / "configs" / "regions"
STAGES = ("normalize", "mask", "clean", "regions", "tables", "export")

VARIANT_FUNCS = {  # ชื่อเดียวกับภาพ section (sections/images.py) ทุกตัวรับภาพสี BGR ความละเอียดเต็ม
    "color": lambda c: c,
    "luma": enhance.to_gray,
    "flat": lambda c: I.flatten_background(I.gray_value_channel(c)),
    "bin": I.binarize_inverted,
}

CLEAN_METHODS = {  # วิธีลบพื้น/ลายน้ำบนสำเนา (ตาม preprocess_finance/preprocess/clean.py) ภาพ variant ปกติตัดจากภาพที่ไม่ clean เสมอ
    "none": lambda img: img,
    "suppress_color": clean_ops.suppress_color,
    "max_channel": lambda img: cv2.cvtColor(I.gray_value_channel(img), cv2.COLOR_GRAY2BGR),
}


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class RegionConfig:
    """configs/regions/: pipeline.yaml + doc_types.yaml + templates/<ชนิด>.yaml (อ่านครั้งเดียว)."""

    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.dir = Path(config_dir)
        self.cfg = load_yaml(self.dir / "pipeline.yaml")
        self.doc_types = load_yaml(self.dir / "doc_types.yaml")
        self._templates: dict[str, dict | None] = {}

    def template(self, doc_type: str) -> dict | None:
        if doc_type not in self._templates:
            p = self.dir / "templates" / f"{doc_type}.yaml"
            self._templates[doc_type] = load_yaml(p) if p.exists() else None
        return self._templates[doc_type]


def run_page(full: np.ndarray, stem: str, src_dpi: float, rc: RegionConfig, page_dir: Path, debug_images: bool = True) -> dict:
    """แบ่ง region/ตารางของหน้าเดียว เขียนผลลง ``page_dir`` คืน manifest."""
    t0 = time.perf_counter()
    cfg = rc.cfg
    for old in [*page_dir.glob("[0-9][0-9]_*.png"), page_dir / "crops"]:
        shutil.rmtree(old) if old.is_dir() else old.unlink(missing_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    def snap(name: str, img: np.ndarray) -> None:
        if debug_images:
            draw.imwrite(page_dir / f"{name}.png", img)

    doc_type, overrides = doc_types.classify(stem, rc.doc_types)
    tpl = rc.template(doc_type)
    opts = {**cfg["defaults"], **((tpl or {}).get("options") or {}), **overrides}

    # สำเนาที่ ref_dpi (ค่า px ทุกตัววัดที่ความละเอียดนี้) — ไม่ขยายภาพที่เล็กกว่า
    scale = min(1.0, cfg["ref_dpi"] / src_dpi)
    work = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else full

    # normalize: ตัดเหลือขอบเขตเนื้อหา + ขอบขาว ให้ ratio ของ template ไม่ขึ้นกับขอบสแกน
    cb = geometry.content_bbox(work)
    nbox = geometry.pad_box(cb, work.shape, int(cfg["normalize"]["margin_frac"] * work.shape[1]))
    img = geometry.crop(work, nbox)
    ox, oy = nbox[0], nbox[1]
    content = (cb[0] - ox, cb[1] - oy, cb[2] - ox, cb[3] - oy)
    snap("01_normalize", img)

    def to_page(box) -> list[int]:
        """กล่องบนสำเนา normalize → พิกัดภาพ final ความละเอียดเต็ม."""
        x0, y0, x1, y1 = box
        H, W = full.shape[:2]
        return [max(0, int((x0 + ox) / scale)), max(0, int((y0 + oy) / scale)),
                min(W, int(np.ceil((x1 + ox) / scale))), min(H, int(np.ceil((y1 + oy) / scale)))]

    m: dict = {"key": stem, "doc_id": stem, "doc_type": doc_type, "options": opts,
               "source": {"page_size": [int(full.shape[1]), int(full.shape[0])], "src_dpi": src_dpi,
                          "ref_dpi": cfg["ref_dpi"], "scale": round(scale, 6), "normalize_offset": [int(ox), int(oy)]},
               "stages": {"normalize": {"content_bbox": [int(v) for v in cb], "crop_box": [int(v) for v in nbox],
                                        "size": [int(img.shape[1]), int(img.shape[0])]}}}

    anchor = opts.get("anchor", "content")
    frame = content
    if anchor == "frame":
        fb = geometry.frame_bbox(img, cfg["regions"]["frame_min_area_frac"])
        frame = fb if fb is not None else content
        anchor = "frame" if fb is not None else "content(fallback)"

    # mask: QR/บาร์โค้ด (ถอดรหัสเก็บไว้) + พื้นที่ ignore ของ template (ลายเซ็น/ตรายาง) ระบายขาวก่อนหา region
    found = codes.find_codes(img, cfg["mask"]) if opts.get("mask_codes", True) else []
    ignored = [{"name": g["name"], "box": split.ratio_to_px(g["box"], frame)} for g in ((tpl or {}).get("ignore") or [])]
    blank_boxes = [c["box"] for c in found] + [g["box"] for g in ignored]
    page = full
    if blank_boxes:
        img = codes.blank(img, blank_boxes)
        page = full.copy()
        pad = int(round(6 / scale))  # codes.blank ขยายกล่อง 6 px บนสำเนา
        for b in blank_boxes:
            x0, y0, x1, y1 = to_page(b)
            page[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 255
        snap("02_mask", img)
    m["stages"]["mask"] = {"codes": found, "ignored": ignored}

    # clean: สำเนาสำหรับ variant ``clean`` เท่านั้น การตรวจโครงหน้าใช้ภาพที่ไม่ clean
    method = opts.get("clean", "none")
    if method not in CLEAN_METHODS:
        raise ValueError(f"clean ไม่รู้จัก {method!r} เลือกจาก {list(CLEAN_METHODS)}")
    cleaned_page = CLEAN_METHODS[method](page) if method != "none" else None
    if cleaned_page is not None:
        snap("03_clean", CLEAN_METHODS[method](img))
    m["stages"]["clean"] = {"method": method}

    # regions: เส้นนอนใช้ทั้งดูดขอบ template และตัดแถบ generic
    lc = cfg["lines"]
    hlines = lines.detect_hlines(img, lc["min_len_frac"]) if opts.get("use_lines", True) else []
    if tpl is not None and opts.get("mode", "template") == "template":
        regs = split.template_regions(img.shape, frame, tpl, cfg["regions"]["pad_frac"], hlines, lc["snap_tol_frac"], lc["snap_min_len_frac"])
    else:
        regs = split.generic_regions(img, frame, {**cfg["regions"], **cfg["generic"]}, hlines)
    overlay = draw.draw_regions(img, regs, frame, hlines)
    snap("04_regions", overlay)
    m["stages"]["regions"] = {"anchor": anchor, "frame": list(frame), "hlines": hlines,
                              "regions": [{**r, "box": list(r["box"])} for r in regs]}

    # tables: เส้นตั้งแบ่งแถบเป็นช่อง แถวแรก = หัว (key)
    text = text_mask(ink_mask(img))
    ex = cfg["export"]

    def trim(box):
        return geometry.trim_box(text, box, ex["trim_pad_px"]) if ex["trim"] else box

    vlines = lines.detect_vlines(img, lc["v_min_len_frac"]) if opts.get("use_lines", True) else []
    tbls = tables.build_tables(img, text, regs, vlines, cfg["tables"], trim) if regs and regs[0]["mode"] == "generic:lines" and opts.get("tables", True) else []
    if tbls or vlines:
        snap("05_tables", draw.draw_tables(overlay, tbls, vlines))
    m["stages"]["tables"] = {"vlines": vlines, "tables": tbls}

    # export: ตัดจากภาพ final สีความละเอียดเต็ม (ระบาย QR แล้ว)
    crops_dir = page_dir / "crops"
    files, skipped = [], []

    def write(name: str, box, meta: dict, variants: list[str]) -> str:
        pb = to_page(box)
        sub = page[pb[1]:pb[3], pb[0]:pb[2]]
        out = {v: VARIANT_FUNCS[v](sub) for v in variants}
        if cleaned_page is not None:
            out["clean"] = cleaned_page[pb[1]:pb[3], pb[0]:pb[2]]
        first = ""
        for v, vimg in out.items():
            p = crops_dir / f"{name}__{v}.png"
            draw.imwrite(p, vimg)
            rel = p.relative_to(page_dir).as_posix()
            first = first or rel
            files.append({**meta, "variant": v, "box": [int(x) for x in box], "box_page": pb, "path": rel})
        return first

    max_h = ex.get("max_piece_frac") or 0
    for i, r in enumerate(regs):
        box = trim(r["box"]) if r.get("trim", True) else r["box"]
        if box is None or float((ink_mask(geometry.crop(img, box)) > 0).mean()) < ex["min_ink_frac"]:
            skipped.append({"region": r["name"]})  # ไม่เหลืออะไร (เช่น แถบที่มีแต่ QR ที่ระบายแล้ว)
            continue
        x0, y0, x1, y1 = box
        pieces = [(y0, y1)]
        if max_h and r.get("tile", True):
            pieces = [(py, py + ph) for _x, py, _w, ph in split_rows(text, (x0, y0, x1 - x0, y1 - y0), max_h * img.shape[0])]
        for j, (a, b) in enumerate(pieces):
            name = f"{i:02d}_{r['name']}" + (f"_t{j:02d}" if len(pieces) > 1 else "")
            write(name, (x0, a, x1, b), {"region": r["name"], "tile": j}, ex["variants"])

    # ช่องตาราง + คู่ key/value (เก็บพาธ ข้อความมาหลัง OCR)
    kv = []
    for t in tbls:
        for ri, row in enumerate(t["rows"]):
            for c in row["cells"]:
                if c["empty"]:
                    continue
                c["crop"] = write(f"tables/t{t['id']}_r{ri:02d}_c{c['col']:02d}", c["trim"],
                                  {"table": t["id"], "row": ri, "col": c["col"], "kind": row["kind"]}, ex["cell_variants"])
        header = {c["col"]: c.get("crop") for c in t["rows"][0]["cells"]}
        for ri, row in enumerate(t["rows"][1:], start=1):
            by_col = {c["col"]: c.get("crop") for c in row["cells"]}
            for c in row["cells"]:
                if not c["empty"]:
                    kv.append({"table": t["id"], "layout": t["layout"], "row": ri, "col": c["col"], "value_crop": c["crop"],
                               "header_key_crop": header.get(c["key_col"]), "left_key_crop": by_col.get(c["left_col"])})
    m["stages"]["export"] = {"n_files": len(files), "files": files, "skipped_regions": skipped, "kv_pairs": kv}
    m["elapsed_s"] = round(time.perf_counter() - t0, 3)
    (page_dir / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    m["_overlay"] = overlay
    return m


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def run_dataset(paths, dataset: str, cfg: dict, profile: str, only: list[str] | None = None, debug_images: bool = True) -> list[dict]:
    """ทุกภาพ final ของ ``02_preprocess/<dataset>/<profile>/`` → ``03_sections/<dataset>/<profile>/regions/``
    เขียน ``summary.json`` และ ``contact_sheet.png`` (overlay ทุกใบในภาพเดียว เริ่ม debug ที่นี่) เมื่อรันทั้งชุด."""
    from ...io import read_image
    from ...stages import read_meta
    from ..build import page_images

    final_dir = paths.preprocess_final(dataset, profile)
    pages = page_images(final_dir) if final_dir.is_dir() else {}
    if not pages:
        raise FileNotFoundError(f"ไม่พบภาพที่ preprocess แล้วใน {final_dir} — รัน: python -m fin_extractor sections <input> --to preprocess")
    stems = [s for s in pages if not only or any(s.endswith(o) for o in only)]
    if not stems:
        raise FileNotFoundError(f"ไม่มีไฟล์ที่ลงท้ายด้วย {only} ใน {final_dir}")
    rc = RegionConfig()
    root = paths.section_regions(dataset, profile=profile)
    summary, sheet = [], []
    for stem in stems:
        src_dpi = cfg["loader"]["pdf_dpi"] * float(read_meta(pages[stem]).get("scale", 1.0))
        m = run_page(read_image(pages[stem]), stem, src_dpi, rc, root / stem, debug_images)
        ex, tb = m["stages"]["export"], m["stages"]["tables"]["tables"]
        regions = [r["name"] for r in m["stages"]["regions"]["regions"]]
        codes_found = [c["kind"] + (f"={c['data']}" if c["data"] else "") for c in m["stages"]["mask"]["codes"]]
        summary.append({"key": stem, "doc_type": m["doc_type"], "mode": m["stages"]["regions"]["regions"][0]["mode"] if regions else None,
                        "regions": regions, "tables": len(tb), "cells": sum(1 for t in tb for r in t["rows"] for c in r["cells"] if not c["empty"]),
                        "kv_pairs": len(ex["kv_pairs"]), "codes": codes_found, "manifest": f"{stem}/manifest.json"})
        sheet.append((stem[-4:], m.pop("_overlay")))
        s = summary[-1]
        print(f"{stem[-4:]}: {m['doc_type']} · {s['mode']} · region {len(regions)} · ตาราง {s['tables']} (ช่อง {s['cells']}, kv {s['kv_pairs']})"
              + (f" · {', '.join(codes_found)}" if codes_found else ""), flush=True)
    if not only:
        (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        draw.imwrite(root / "contact_sheet.png", draw.contact_sheet(sheet))
    print(f"รวม {len(stems)} ไฟล์ → {root}")
    return summary
