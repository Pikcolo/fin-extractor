"""ขั้นที่ต้องโหลดโมเดลของหน้า OCR demo (``ocr_demo_app.py``) รันแยก process ทีละขั้น พิมพ์ความคืบหน้าเป็น JSON ทีละบรรทัดทาง stdout

    python -X utf8 scripts/demo/ocr_demo_worker.py preprocess <job_dir> [--orientation 0|90|180|270]
    python -X utf8 scripts/demo/ocr_demo_worker.py ocr <job_dir> --engine openthai|typhoon|paddle

แยก process เพราะตัวตรวจทิศ (Paddle) กับ OpenThai (torch) ใช้ cuDNN คนละรุ่น อยู่ process เดียวกันไม่ได้ (ดู preprocessing/orient.py)
และถ้าโมเดลล้ม/VRAM เต็ม เซิร์ฟเวอร์ของหน้าเว็บไม่ล้มตาม
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import load_pages, save_image  # noqa: E402
from fin_extractor.preprocessing import preprocess  # noqa: E402

VIEW_SIDE = 1400  # ด้านยาวของภาพที่แสดงบนหน้าเว็บ (ภาพจริงที่ใช้ต่อเก็บความละเอียดเต็มแยกไว้)


def emit(**event) -> None:
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def save_view(path: Path, img: np.ndarray) -> list[int]:
    """ภาพย่อสำหรับแสดง (JPEG) คืนขนาดภาพจริง [กว้าง, สูง]."""
    s = min(1.0, VIEW_SIDE / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    save_image(path, small)
    return [int(img.shape[1]), int(img.shape[0])]


def cmd_preprocess(job: Path, orientation: int | None) -> None:
    cfg = load_config()
    src = next(job.glob("source.*"))  # ไฟล์ที่อัปโหลด ไม่ใช่ input.jpg (ภาพย่อสำหรับแสดง)
    t = time.perf_counter()
    pages = load_pages(src, cfg["loader"]["pdf_dpi"])
    emit(event="loaded", pages=len(pages), seconds=round(time.perf_counter() - t, 2))
    t = time.perf_counter()
    res = preprocess(pages[0], cfg["preprocessing"], keep_steps=True, orientation=orientation)
    steps = []
    for i, (name, img) in enumerate(res.steps.items()):
        size = save_view(job / f"pre_{i:02d}_{name}.jpg", img)
        steps.append({"name": name, "image": f"pre_{i:02d}_{name}.jpg", "size": size})
    save_image(job / "final.png", res.image)  # ความละเอียดเต็ม ขั้นถัดไปตัดภาพรายส่วนจากภาพนี้
    emit(event="done", steps=steps, meta=res.meta, final_size=[int(res.image.shape[1]), int(res.image.shape[0])],
         settings={k: cfg["preprocessing"].get(k) for k in ("min_long_side", "fix_orientation", "deskew", "max_skew_angle",
                                                             "denoise", "clahe", "binarize", "keep_color")},
         seconds=round(time.perf_counter() - t, 2))


def cmd_ocr(job: Path, engine: str) -> None:
    from fin_extractor.readers import TORCH_ENGINES, make_reader

    cfg = load_config()
    index = json.loads((job / "sections" / "index.json").read_text(encoding="utf-8"))["sections"]
    reader = make_reader(engine, cfg)
    t = time.perf_counter()
    if hasattr(reader, "_load"):  # typhoon เรียก Ollama ไม่มีโมเดลให้โหลดในเครื่อง
        reader._load()
    emit(event="loaded", reader=reader.name, seconds=round(time.perf_counter() - t, 2))
    torch = None
    if engine in TORCH_ENGINES:
        import torch
    for n, s in enumerate(index, start=1):
        emit(event="start", id=s["id"], n=n, total=len(index))
        gray = cv2.imdecode(np.fromfile(str(job / "sections" / s["file"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        t = time.perf_counter()
        res = reader.recognize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        if torch is not None:
            torch.cuda.empty_cache()  # เหมือน sections/read.py: ไม่ให้หน่วยความจำสะสมจนภาพถัดไปขอเพิ่มไม่ได้
        emit(event="section", id=s["id"], text=res.text, truncated=res.truncated, confidence=res.confidence,
             seconds=round(time.perf_counter() - t, 2))
    emit(event="done", reader=reader.name)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preprocess", "ocr"])
    ap.add_argument("job", type=Path)
    ap.add_argument("--orientation", type=int, choices=[0, 90, 180, 270])
    ap.add_argument("--engine", default="openthai")
    args = ap.parse_args()
    try:
        if args.stage == "preprocess":
            cmd_preprocess(args.job, args.orientation)
        else:
            cmd_ocr(args.job, args.engine)
    except Exception as e:  # noqa: BLE001 — ส่งข้อความกลับไปแสดงบนหน้าเว็บ
        emit(event="error", error=f"{type(e).__name__}: {str(e)[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
