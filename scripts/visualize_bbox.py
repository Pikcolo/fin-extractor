"""วาดกรอบ (bbox) ที่ Paddle ตรวจเจอลงบนภาพหลัง preprocess เป็นไฟล์ภาพ (ไม่ทำ HTML)

ต่อเอกสารได้ 2 ภาพที่ ``result_ocr/paddle/<dataset>/visualize/<ไฟล์>/``
  words.jpg   กรอบของทุกคำ สีตามความมั่นใจที่ Paddle รายงาน (ไล่ต่อเนื่องจากส้มแดง=ต่ำ ไปน้ำเงิน=สูง ไม่มีเกณฑ์ตัดสี)
  lines.jpg   กรอบของบรรทัดที่จัดกลุ่มแล้ว พร้อมเลขลำดับที่ขอบซ้าย
แถบบนของภาพบอกจำนวนคำ/บรรทัด ความมั่นใจต่ำสุด-เฉลี่ย-สูงสุด และแถบสี

Paddle ไม่ได้เก็บกรอบไว้ในผล extract (เก็บแค่ข้อความ) จึงต้องอ่านภาพใหม่ เฉพาะไฟล์ที่ระบุ ทำทีละไฟล์ (หนัก)
ผลระดับคำเก็บที่ ``ocr_words.json`` ข้าง ๆ ภาพ รอบหลังวาดใหม่จากไฟล์นี้โดยไม่โหลด Paddle (ใส่ --redo เพื่ออ่านใหม่)
Typhoon ไม่มีกรอบ (VLM อ่านทั้งหน้าแล้วคืน Markdown)

    .venv/Scripts/python scripts/visualize_bbox.py 001312 001313 001315        # ส่วนท้ายของชื่อไฟล์
    .venv/Scripts/python scripts/visualize_bbox.py --cached                    # วาดซ้ำเฉพาะไฟล์ที่มี ocr_words.json แล้ว
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.ocr.base import OCRWord, group_into_lines  # noqa: E402
from fin_extractor.paths import DEFAULT_PROFILE, DataPaths, font_path  # noqa: E402

MAX_SIDE = 1800
LOW, HIGH = (213, 94, 0), (0, 114, 178)  # ความมั่นใจต่ำ → สูง (Okabe-Ito แยกได้แม้ตาบอดสี)
HEADER = 64


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(font_path("Sarabun-Regular.ttf")), size)
    except OSError:
        return ImageFont.load_default()


def conf_color(c: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * c) for a, b in zip(LOW, HIGH))


def with_header(img: Image.Image, text: str, bar: tuple[float, float] | None = None) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + HEADER), "white")
    canvas.paste(img, (0, HEADER))
    d, f = ImageDraw.Draw(canvas), font(24)
    d.text((12, 6), text, fill="black", font=f)
    if bar:
        lo, hi = bar
        x0, x1, y = 12, min(420, img.width - 24), 42
        for x in range(x0, x1):
            t = (x - x0) / max(1, x1 - x0 - 1)
            d.line([(x, y), (x, y + 14)], fill=conf_color(t))
        d.text((x1 + 10, y - 8), f"ความมั่นใจ {lo:.2f} (ซ้าย) → {hi:.2f} (ขวา) ไล่สีต่อเนื่อง", fill="black", font=font(20))
    return canvas


def shrink(img: Image.Image) -> tuple[Image.Image, float]:
    s = min(1.0, MAX_SIDE / max(img.size))
    return (img.resize((round(img.width * s), round(img.height * s))) if s < 1 else img.copy()), s


def draw_words(page: Image.Image, words: list[OCRWord], title: str) -> Image.Image:
    img, s = shrink(page)
    d = ImageDraw.Draw(img, "RGBA")
    confs = [w.confidence for w in words]
    lo, hi = min(confs), max(confs)
    for w in words:
        c = conf_color((w.confidence - lo) / (hi - lo) if hi > lo else 1.0)  # สีไล่ตามช่วงที่พบในหน้านี้
        d.rectangle([v * s for v in w.bbox], fill=(*c, 40), outline=c, width=2)
    return with_header(img, f"{title} · {len(words)} คำ · ความมั่นใจ ต่ำสุด {lo:.3f} เฉลี่ย {sum(confs) / len(confs):.3f} สูงสุด {hi:.3f}", (lo, hi))


def draw_lines(page: Image.Image, words: list[OCRWord], title: str) -> Image.Image:
    lines = group_into_lines(words)
    img, s = shrink(page)
    d, f = ImageDraw.Draw(img, "RGBA"), font(22)
    for i, ln in enumerate(lines, 1):
        x0, y0, x1, y1 = (v * s for v in ln.bbox)
        d.rectangle([x0, y0, x1, y1], outline=(*HIGH, 220), width=3)
        label = str(i)
        w = d.textlength(label, font=f)
        d.rectangle([x0 - w - 10, y0, x0 - 2, y0 + 26], fill=HIGH)
        d.text((x0 - w - 6, y0), label, fill="white", font=f)
    return with_header(img, f"{title} · {len(lines)} บรรทัด (เลขที่ขอบซ้าย = ลำดับบรรทัด)")


def read_words(cache: Path) -> list[OCRWord]:
    return [OCRWord(w["text"], w["conf"], tuple(w["bbox"])) for w in json.loads(cache.read_text(encoding="utf-8"))["words"]]


def run_paddle(image_path: Path, cache: Path, engine) -> None:
    from fin_extractor.io import read_image

    res = engine.recognize(read_image(image_path))
    words = [w for line in res.lines for w in line.words]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"image": image_path.name, "words": [{"text": w.text, "conf": w.confidence, "bbox": list(w.bbox)} for w in words]},
                                ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="*", help="ส่วนท้ายของชื่อไฟล์ เช่น 001312")
    ap.add_argument("--cached", action="store_true", help="วาดซ้ำเฉพาะไฟล์ที่มี ocr_words.json แล้ว (ไม่โหลด Paddle)")
    ap.add_argument("--redo", action="store_true", help="ให้ Paddle อ่านใหม่แม้มีผลเก็บไว้")
    ap.add_argument("--dataset", default="finance_doc_scan")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    args = ap.parse_args()

    P = DataPaths()
    out_root, final = P.visualize("paddle", args.dataset), P.preprocess_final(args.dataset, args.profile)
    images = sorted(final.rglob("*_p1.png"))
    if args.cached:
        images = [p for p in images if (out_root / p.stem.removesuffix("_p1") / "ocr_words.json").exists()]
    elif args.docs:
        images = [p for p in images if any(p.stem.removesuffix("_p1").endswith(d) for d in args.docs)]
    else:
        sys.exit("ระบุไฟล์ (เช่น 001312) หรือใช้ --cached")
    if not images:
        sys.exit("ไม่พบไฟล์ที่ตรงกัน")

    engine = None
    for img_path in images:
        name = img_path.stem.removesuffix("_p1")
        out = out_root / name
        cache = out / "ocr_words.json"
        if args.redo or not cache.exists():
            if engine is None:  # โหลด Paddle เมื่อจำเป็นเท่านั้น และโหลดครั้งเดียว
                from fin_extractor.config import load_config
                from fin_extractor.ocr import create_engine

                engine = create_engine("paddle", load_config())
            run_paddle(img_path, cache, engine)
        words = read_words(cache)
        with Image.open(img_path) as im:
            page = im.convert("RGB")
        draw_words(page, words, name).convert("RGB").save(out / "words.jpg", quality=88)
        draw_lines(page, words, name).convert("RGB").save(out / "lines.jpg", quality=88)
        print(out / "words.jpg")


if __name__ == "__main__":
    main()
