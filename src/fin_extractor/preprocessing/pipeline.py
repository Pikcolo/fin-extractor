"""รวมขั้นตอน preprocess ตามค่าใน config (``preprocessing``).

ลำดับ: (remove_streaks) → upscale → orientation (0/90/180/270) → deskew → crop → denoise → CLAHE → binarize
ค่าตั้งต้นเปิดแค่ upscale / orientation / deskew และคงสีไว้ (วัดแล้วใน docs/02-preprocessing.md)
เก็บภาพทุกขั้นไว้ใน ``steps`` เพื่อ debug ว่าขั้นไหนทำให้ภาพแย่ลง
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..io import save_image
from . import binarize as bin_ops
from . import clean, crop, enhance, orient
from .deskew import deskew


@dataclass
class PreprocessResult:
    image: np.ndarray
    steps: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def save_steps(self, out_dir: str | Path, stem: str) -> list[Path]:
        out_dir = Path(out_dir)
        paths = []
        for i, (name, img) in enumerate(self.steps.items()):
            p = out_dir / f"{stem}_{i:02d}_{name}.png"
            save_image(p, img)
            paths.append(p)
        return paths


def preprocess(img: np.ndarray, cfg: dict[str, Any], keep_steps: bool = False, orientation: int | None = None) -> PreprocessResult:
    """``orientation`` = มุมที่สั่งเอง (0/90/180/270 ตามเข็ม) ใช้แทนตัวตรวจอัตโนมัติ ``meta["orientation_source"] = "manual"``
    ตัวตรวจอัตโนมัติล้ม → ``meta["orientation_error"]`` (มุมที่ใช้คือ 0 จึงอาจตะแคงอยู่ ตรวจ manifest)"""
    if orientation not in (None, 0, 90, 180, 270):
        raise ValueError(f"orientation ต้องเป็น 0/90/180/270: {orientation}")
    steps: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}

    def record(name: str, im: np.ndarray) -> np.ndarray:
        if keep_steps:
            steps[name] = im
        return im

    record("original", img)
    out = img if cfg.get("keep_color") else enhance.to_gray(img)
    out = record("gray" if out.ndim == 2 else "color", out)

    if cfg.get("remove_streaks"):  # ต้องก่อนหมุน/deskew: เกณฑ์ "แตะทั้งสองขอบ" ใช้ได้กับภาพดิบเท่านั้น
        out, meta["streaks_removed"] = clean.remove_streaks(out)
        if meta["streaks_removed"]:
            record("streaks", out)

    out, meta["scale"] = enhance.upscale(out, cfg.get("min_long_side", 0))
    if meta["scale"] != 1.0:
        record("upscale", out)

    if orientation is not None:
        meta["orientation"], meta["orientation_source"] = orientation, "manual"
    elif cfg.get("fix_orientation"):
        meta["orientation"], err = orient.detect_orientation_checked(out)
        if err:
            meta["orientation_error"] = err
    if orientation is not None or cfg.get("fix_orientation"):
        if meta["orientation"]:
            out = record("orientation", orient.rotate_orthogonal(out, meta["orientation"]))

    if cfg.get("deskew"):
        out, meta["skew_angle"] = deskew(out, cfg.get("max_skew_angle", 15.0))
        if meta["skew_angle"]:
            record("deskew", out)

    if cfg.get("remove_borders"):
        out = record("crop", crop.crop_to_content(out))

    if cfg.get("denoise", "none") != "none":
        out = record("denoise", clean.denoise(out, cfg["denoise"]))

    if cfg.get("clahe"):
        out = record("clahe", enhance.clahe(out))

    if cfg.get("binarize", "none") != "none":
        out = record("binarize", bin_ops.binarize(out, cfg["binarize"]))

    return PreprocessResult(image=out, steps=steps, meta=meta)
