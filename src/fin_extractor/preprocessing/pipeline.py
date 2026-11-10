"""รวมขั้นตอน preprocess ตามค่าใน config (``preprocessing``).

ลำดับ: upscale → orientation (0/90/180/270) → deskew → crop → denoise → CLAHE → binarize
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
from . import enhance, geometry


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


def preprocess(img: np.ndarray, cfg: dict[str, Any], keep_steps: bool = False) -> PreprocessResult:
    steps: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}

    def record(name: str, im: np.ndarray) -> np.ndarray:
        if keep_steps:
            steps[name] = im
        return im

    record("original", img)
    out = img if cfg.get("keep_color") else enhance.to_gray(img)
    out = record("gray" if out.ndim == 2 else "color", out)

    out, meta["scale"] = enhance.upscale(out, cfg.get("min_long_side", 0))
    if meta["scale"] != 1.0:
        record("upscale", out)

    if cfg.get("fix_orientation"):
        meta["orientation"] = geometry.detect_orientation(out)
        if meta["orientation"]:
            out = record("orientation", geometry.rotate_orthogonal(out, meta["orientation"]))

    if cfg.get("deskew"):
        out, meta["skew_angle"] = geometry.deskew(out, cfg.get("max_skew_angle", 15.0))
        if meta["skew_angle"]:
            record("deskew", out)

    if cfg.get("remove_borders"):
        out = record("crop", geometry.crop_to_content(out))

    if cfg.get("denoise", "none") != "none":
        out = record("denoise", enhance.denoise(out, cfg["denoise"]))

    if cfg.get("clahe"):
        out = record("clahe", enhance.clahe(out))

    if cfg.get("binarize", "none") != "none":
        out = record("binarize", bin_ops.binarize(out, cfg["binarize"]))

    return PreprocessResult(image=out, steps=steps, meta=meta)
