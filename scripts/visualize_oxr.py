"""วาดภาพแต่ละขั้นของ OXR จากผลดิบที่เก็บไว้ (ไม่เรียก OXR/GPU) แล้วรวมเป็นรายงาน HTML ในเครื่อง

อินพุต: ผลเต็มของ OXR ใน ``result_ocr/oxr/<dataset>/raw_data/`` สองแบบ
  ``<ไฟล์>_p1.oxr.json``      รอบที่ส่งภาพที่ผ่าน preprocess (ภาพ = preprocess/<dataset>/default/final/)  ← scripts/extract_oxr.py
  ``<ไฟล์>_p1.pdf.oxr.json``  รอบที่ส่ง PDF ดิบ (ภาพ = PDF เรนเดอร์ 150 dpi เหมือนที่ OXR เรนเดอร์เอง)
ผล: ``result_ocr/oxr/<dataset>/visualize/<เลขไฟล์>_<pdf|preprocessed>/index.html`` พร้อมภาพประกอบ (เปิดในเบราว์เซอร์ ลบทั้งโฟลเดอร์ได้)
**ไม่ publish ขึ้นบริการภายนอก** เอกสารเป็นข้อมูลจริง

ขั้นที่แสดง
  1 ภาพนำเข้า      PDF เรนเดอร์ 150 dpi (ขนาดเท่าที่ OXR เห็น พิกัดตรงกันเป๊ะ)
  2 preprocess     ผลของโปรเจกต์ที่เก็บไว้ (ต้นฉบับ → คงสี → ปรับเอียง) **รอบ OXR ที่ใช้วาดไม่ได้ผ่านขั้นนี้** ส่ง PDF ดิบ
  3 layout         กรอบบล็อกและชนิด (Title/Text/Table/Picture) จาก D-FINE
  4 ลำดับการอ่าน   เลข idx เรียงตามลำดับ + เส้นเชื่อม
  5 อ่านเนื้อหา    ภาพตัดแต่ละบล็อก คู่กับข้อความที่ OXR อ่านได้
  6 ผลสุดท้าย      Markdown

สิ่งที่ผลดิบไม่มี (จึงวาดไม่ได้): กรอบก่อนตัดบล็อกซ้ำ/ซ้อน, header/footer ที่ถูกกรองทิ้ง, เวลาของแต่ละขั้น
bbox/ลำดับ ในภาพคือผลสุดท้ายหลัง postprocess

    .venv/Scripts/python scripts/visualize_oxr.py 001312 001313 001315
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.paths import DataPaths, find_dir, font_path  # noqa: E402

P = DataPaths()
DATASET = "finance_doc_scan"
PREFIX = "BRWAC50DE3689DA_"
OXR_DIR = P.raw_data("oxr", DATASET)
PDF_DIR = P.raw(DATASET)
PRE_DIR = P.preprocess(DATASET, "default")
OUT_ROOT = P.visualize("oxr", DATASET)
OXR_DPI = 150  # ค่าตายตัวใน oxr/utils/pdf.py

# สีเข้มพอบนกระดาษขาว แยกกันได้แม้ตาบอดสี (Okabe-Ito) ชนิดอื่นเป็นเทา
COLORS = {"Title": (213, 94, 0), "Text": (0, 114, 178), "Table": (0, 158, 115), "Picture": (204, 121, 167)}
OTHER = (110, 110, 110)
FONT_PATH = str(font_path("Sarabun-Regular.ttf"))  # ฟอนต์ในโปรเจกต์ (assets/fonts)
ALLOWED_TAG = re.compile(r"<(/?)(table|thead|tbody|tr|td|th|br)((?:\s+(?:colspan|rowspan)=\"\d+\")*)\s*/?>", re.I)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def color(t: str) -> tuple[int, int, int]:
    return COLORS.get(t, OTHER)


def oxr_json(stem: str, source: str) -> Path:
    name = f"{PREFIX}{stem}"
    return find_dir(OXR_DIR, name) / f"{name}_p1{'.pdf' if source == 'pdf' else ''}.oxr.json"


def pick_source(stem: str, source: str) -> str:
    """auto = ใช้รอบที่ผ่าน preprocess ถ้ามี ไม่งั้นรอบ PDF ดิบ."""
    if source == "auto":
        return "preprocessed" if oxr_json(stem, "preprocessed").exists() else "pdf"
    return source


def input_page(stem: str, source: str) -> Image.Image:
    if source == "pdf":
        return render_page(stem)
    name = f"{PREFIX}{stem}"
    return Image.open(find_dir(PRE_DIR / "final", name) / f"{name}_p1.png").convert("RGB")


def render_page(stem: str) -> Image.Image:
    pdf = pdfium.PdfDocument(str(PDF_DIR / f"{PREFIX}{stem}.pdf"))
    return pdf[0].render(scale=OXR_DPI / 72).to_pil().convert("RGB")


def tag(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill, f) -> None:
    x, y = xy
    w = draw.textlength(text, font=f)
    draw.rectangle([x, y - 22, x + w + 8, y], fill=fill)
    draw.text((x + 4, y - 21), text, fill="white", font=f)


def layout_image(page: Image.Image, blocks: list[dict]) -> Image.Image:
    img = page.copy()
    d = ImageDraw.Draw(img, "RGBA")
    f = font(17)
    for b in blocks:
        x1, y1, x2, y2 = b["bbox"]
        c = color(b["type"])
        d.rectangle([x1, y1, x2, y2], fill=(*c, 38), outline=c, width=3)
    for b in blocks:  # ป้ายวาดทีหลัง กันถูกกรอบอื่นทับ
        tag(d, (b["bbox"][0], max(b["bbox"][1], 24)), b["type"], color(b["type"]), f)
    return img


def order_image(page: Image.Image, blocks: list[dict]) -> Image.Image:
    img = page.copy()
    d = ImageDraw.Draw(img, "RGBA")
    f = font(20)
    ordered = sorted(blocks, key=lambda b: b["idx"])
    centers = [((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2) for b in ordered]
    for b in ordered:
        d.rectangle(b["bbox"], outline=(*color(b["type"]), 200), width=2)
    for (x1, y1), (x2, y2) in zip(centers, centers[1:]):
        d.line([x1, y1, x2, y2], fill=(30, 30, 30, 170), width=3)
        dx, dy = x2 - x1, y2 - y1
        n = max((dx * dx + dy * dy) ** 0.5, 1)
        ux, uy = dx / n, dy / n  # หัวลูกศรที่ปลายเส้น
        d.polygon([(x2, y2), (x2 - 16 * ux + 7 * uy, y2 - 16 * uy - 7 * ux), (x2 - 16 * ux - 7 * uy, y2 - 16 * uy + 7 * ux)],
                  fill=(30, 30, 30, 200))
    for b in ordered:
        x, y = b["bbox"][0], b["bbox"][1]
        label = str(b["idx"])
        w = d.textlength(label, font=f)
        d.ellipse([x - 4, y - 4, x + w + 14, y + 26], fill=color(b["type"]), outline="white", width=2)
        d.text((x + 5, y), label, fill="white", font=f)
    return img


def preprocess_strip(stem: str) -> tuple[Image.Image | None, str]:
    names = [("original", "ต้นฉบับ"), ("color", "คงสี"), ("deskew", "ปรับเอียง")]
    imgs, labels = [], []
    for key, th in names:
        name = f"{PREFIX}{stem}"
        p = find_dir(PRE_DIR / "steps", name) / f"{name}_p1_{'00_original' if key == 'original' else '01_color' if key == 'color' else '02_deskew'}.png"
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
            labels.append(key)
    meta = ""
    mf = PRE_DIR / "manifest.json"
    if mf.exists():
        for r in json.loads(mf.read_text(encoding="utf-8")).get("files", []):
            if r["file"].endswith(f"{stem}.pdf"):
                meta = json.dumps(r["meta"], ensure_ascii=False)
    if not imgs:
        return None, meta
    h = 560
    thumbs = [im.resize((max(1, round(im.width * h / im.height)), h)) for im in imgs]
    canvas = Image.new("RGB", (sum(t.width for t in thumbs) + 12 * (len(thumbs) + 1), h + 44), "white")
    d, f, x = ImageDraw.Draw(canvas), font(20), 12
    for t, lab in zip(thumbs, labels):
        canvas.paste(t, (x, 36))
        d.text((x, 8), lab, fill="black", font=f)
        x += t.width + 12
    return canvas, meta


def sanitize_table(src: str) -> str:
    """แสดงตารางที่โมเดลคืนมา: ปล่อยเฉพาะแท็กตารางพื้นฐาน ที่เหลือ escape ทั้งหมด."""
    out, last = [], 0
    for m in ALLOWED_TAG.finditer(src):
        out.append(html.escape(src[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(html.escape(src[last:]))
    return "".join(out)


def build(stem: str, source: str) -> Path:
    data = json.loads(oxr_json(stem, source).read_text(encoding="utf-8"))
    result = data["data"]["result"]["outputs"]
    page_json = result["json"]["pages"][0]
    blocks = page_json["blocks"]
    page = input_page(stem, source)
    assert page.size == (page_json["page_size"]["w"], page_json["page_size"]["h"]), "ขนาดภาพไม่ตรงกับที่ OXR ใช้"

    out = OUT_ROOT / f"{stem}_{source}"
    if source == "pdf":
        input_note = f"PDF เรนเดอร์ที่ 150 dpi ขนาด {page.width}×{page.height} px (OXR ตั้ง dpi ตายตัว ไม่ใช้ 300 เหมือนโปรเจกต์) พิกัดทุกอย่างข้างล่างอิงภาพนี้"
        pre_note = "รอบ OXR ที่ใช้วาดนี้ส่ง PDF ดิบ ไม่ผ่านขั้นนี้ ที่แสดงคือสิ่งที่ pipeline ของโปรเจกต์ทำกับไฟล์นี้ก่อนส่งให้ OCR ทุกตัว"
    else:
        input_note = f"ภาพสุดท้ายหลัง preprocess ของโปรเจกต์ (preprocess/final) ขนาด {page.width}×{page.height} px ส่งให้ OXR ตามนี้ พิกัดทุกอย่างข้างล่างอิงภาพนี้"
        pre_note = "รอบ OXR ที่ใช้วาดนี้ได้ภาพจากขั้นนี้ ที่แสดงคือสิ่งที่ pipeline ทำกับไฟล์ก่อนส่งให้ OCR ทุกตัว"
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    page.save(out / "01_input.png")
    strip, meta = preprocess_strip(stem)
    if strip:
        strip.save(out / "02_preprocess.png")
    layout_image(page, blocks).save(out / "03_layout.png")
    order_image(page, blocks).save(out / "04_reading_order.png")

    counts = Counter(b["type"] for b in blocks)
    legend = "".join(
        f'<span class="chip" style="background:rgb{color(t)}">{t} × {n}</span>' for t, n in counts.most_common())
    rows = []
    for b in sorted(blocks, key=lambda b: b["idx"]):
        x1, y1, x2, y2 = (round(v) for v in b["bbox"])
        crop = page.crop((max(0, x1), max(0, y1), min(page.width, x2), min(page.height, y2)))
        name = f"blocks/{b['idx']:02d}_{b['type']}.png"
        crop.save(out / name)
        content = b["content"]
        shown = (f'<div class="rendered">{sanitize_table(content)}</div><details><summary>HTML ที่โมเดลตอบ</summary>'
                 f'<pre>{html.escape(content)}</pre></details>') if b["type"] == "Table" else (
            f"<pre>{html.escape(content) if content.strip() else '(ไม่มีข้อความ — ภาพถูกเก็บเป็นไฟล์)'}</pre>")
        rows.append(
            f'<tr><td class="n"><span class="chip" style="background:rgb{color(b["type"])}">{b["idx"]}</span><br>{b["type"]}'
            f'<br><small>{x1},{y1}<br>{x2},{y2}</small></td>'
            f'<td><img src="{name}" loading="lazy"></td><td>{shown}</td></tr>')

    pre_html = (f'<img src="02_preprocess.png"><p class="note">meta ของ preprocess: {html.escape(meta)}</p>'
                if strip else "<p class=\"note\">ไม่มีผล preprocess ของไฟล์นี้</p>")
    page_html = f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OXR ทีละขั้น — {stem}</title><style>
:root{{--bg:#fff;--fg:#1b1b1b;--muted:#5a5a5a;--line:#d8d8d8;--card:#f6f6f6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--muted:#a0a0a0;--line:#3a3d44;--card:#20232a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:16px}} h1{{font-size:1.4rem}} h2{{font-size:1.15rem;margin-top:2rem;border-top:1px solid var(--line);padding-top:1rem}}
img{{max-width:100%;height:auto;border:1px solid var(--line)}} .note,.muted{{color:var(--muted);font-size:.9rem}}
.chip{{display:inline-block;color:#fff;border-radius:4px;padding:1px 8px;margin-right:6px;font-size:.85rem}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid var(--line);padding:6px;vertical-align:top}}
.n{{width:90px;font-size:.85rem}} pre{{white-space:pre-wrap;word-break:break-word;margin:0;font:inherit;background:var(--card);padding:6px}}
.rendered table td,.rendered table th{{border:1px solid var(--line);padding:2px 6px;font-size:.85rem}}
.flow{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}} .flow div{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 10px}}
</style></head><body><main>
<h1>OXR ทีละขั้น — ไฟล์ {stem}</h1>
<p class="muted">สร้างจากผลดิบที่เก็บไว้ (ไม่ได้รัน OXR ใหม่) · เอกสารจริง เก็บในเครื่องเท่านั้น</p>
<div class="flow"><div>1 นำเข้า<br><small>PDF → ภาพ 150 dpi</small></div>→<div>2 preprocess<br><small>(โปรเจกต์ ไม่ใช่ของ OXR)</small></div>→<div>3 layout<br><small>D-FINE แบ่งบล็อก</small></div>→
<div>4 ลำดับการอ่าน<br><small>OXR VLM / XY-Cut++</small></div>→<div>5 อ่านเนื้อหา<br><small>OXR VLM ทีละบล็อก</small></div>→<div>6 รวมเป็น Markdown/JSON</div></div>

<h2>1 ภาพนำเข้า</h2><p class="note">{input_note}</p><img src="01_input.png">

<h2>2 preprocess (ของโปรเจกต์)</h2><p class="note">{pre_note}</p>{pre_html}

<h2>3 layout: แบ่งบล็อกและชนิด</h2><p>{legend}</p><img src="03_layout.png">
<p class="note">เป็นโมเดลแยกต่างหากจาก OXR (D-FINE) ได้กรอบ + ชนิด + คะแนน ยังไม่ได้อ่านตัวอักษร · รวม {len(blocks)} บล็อก (หลังตัดซ้ำซ้อน ไม่รวม header/footer ที่ถูกกรอง)</p>

<h2>4 ลำดับการอ่าน</h2><img src="04_reading_order.png">
<p class="note">เลขในวงกลม = <code>idx</code> ลำดับที่ OXR ตัดสิน (ถามโมเดลให้เรียงกรอบ ถ้าตอบไม่ครบใช้ XY-Cut++ แทน) เส้นเชื่อมตามลำดับ</p>

<h2>5 อ่านเนื้อหาทีละบล็อก</h2><p class="note">ตัดภาพตามกรอบแล้วถามโมเดลด้วย prompt ตามชนิด: Text → "Recognize the content of the image." · Table → "...into html format." · Picture ไม่ OCR</p>
<table><tr><th>ลำดับ / ชนิด / พิกัด</th><th>ภาพที่ส่งให้โมเดล</th><th>ข้อความที่โมเดลตอบ</th></tr>{''.join(rows)}</table>

<h2>6 ผลสุดท้าย (Markdown)</h2><pre>{html.escape(result['markdown'])}</pre>
</main></body></html>"""
    (out / "index.html").write_text(page_html, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="เลขไฟล์ เช่น 001312")
    ap.add_argument("--source", choices=["auto", "preprocessed", "pdf"], default="auto",
                    help="ผล OXR รอบไหน: preprocessed = ส่งภาพที่ผ่าน preprocess, pdf = ส่ง PDF ดิบ, auto = รอบแรกถ้ามี")
    args = ap.parse_args()
    for s in args.stems:
        source = pick_source(s, args.source)
        if not oxr_json(s, source).exists():
            sys.exit(f"ไม่มีผล OXR ของ {s} ({source}) ที่ {oxr_json(s, source)}")
        print("->", build(s, source))


if __name__ == "__main__":
    main()
