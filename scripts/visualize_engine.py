"""รายงาน HTML แสดงทีละขั้นว่า Paddle / Typhoon ทำอะไรกับเอกสาร (เหมือน visualize_oxr.py แต่ของ engine หลักของโปรเจกต์)

อ่านผลที่เก็บไว้แล้ว ไม่เรียก VLM/LLM ส่วนกรอบคำของ Paddle ไม่อยู่ในผลที่เก็บ (เก็บแค่ข้อความ) จึงต้องให้ Paddle อ่านภาพใหม่
เฉพาะไฟล์ที่ระบุใน ``--ocr`` แล้วเก็บผลไว้ที่ ``visualize/<ไฟล์>/ocr_words.json`` รอบหลังวาดจากไฟล์นี้ ไม่โหลด Paddle อีก

ขั้นที่แสดง (ทุกไฟล์ใน visualize/<ไฟล์>/index.html)
  1 ภาพนำเข้า    หน้าที่เรนเดอร์จาก PDF/ภาพต้นฉบับ
  2 preprocess    ต้นฉบับ → คงสี → ปรับเอียง (ภาพทีละขั้นของ ``preprocess --steps`` + meta)
  Paddle
  3 ตรวจจับคำ     กรอบทุกคำที่ Paddle เจอ สีตามความมั่นใจ (ไล่ต่อเนื่อง ไม่มีเกณฑ์) + คำที่มั่นใจต่ำสุดเรียงให้
  4 จัดเป็นบรรทัด  เลขลำดับบรรทัดและกรอบบรรทัด
  5 อ่านทีละบรรทัด ภาพตัดแต่ละบรรทัดคู่กับข้อความที่อ่านได้
  Typhoon
  3 อ่านทั้งหน้า   VLM ได้ภาพทั้งหน้าและคืน Markdown (ไม่มีกรอบคำและไม่มีความมั่นใจรายคำ ตารางแสดงเป็นตาราง)
  ทั้งคู่
  ท้าย            ค่าที่สกัดแต่ละฟิลด์ + ระดับหลักฐาน + check ของแต่ละ extractor (rules / llm) และข้อความดิบ

    .venv/Scripts/python scripts/visualize_engine.py [paddle typhoon] [--ocr 001312 001313 001315] [--dataset finance_doc_scan]

ผล: result_ocr/<engine>/<dataset>/visualize/index.html (สรุปทุกไฟล์) + <ไฟล์>/index.html ต่อเอกสาร เปิดด้วยเบราว์เซอร์ ลบทั้งโฟลเดอร์ได้
**ไม่ publish ขึ้นบริการภายนอก** เอกสารเป็นข้อมูลจริง
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.ocr.base import OCRWord, group_into_lines  # noqa: E402
from fin_extractor.paths import DEFAULT_PROFILE, DataPaths, find_dir, font_path, iter_results  # noqa: E402

MAX_SIDE = 1800
RAW_EXT = {"paddle": "txt", "typhoon": "md"}
LEVELS = {  # ลำดับ = ลำดับแสดง (ต้องดูก่อน → ท้ายสุด) ป้ายมีทั้งสัญลักษณ์และข้อความ ไม่พึ่งสีอย่างเดียว
    "conflict": ("✗", "ขัดแย้ง ต้องแก้หรือยืนยัน"),
    "missing": ("?", "หาไม่เจอ"),
    "unverified": ("~", "ไม่มีตัวยืนยัน ดูเทียบภาพ"),
    "verified": ("✓", "มี check ยืนยัน"),
    "not_applicable": ("–", "ไม่มีในเอกสารชนิดนี้"),
}
STATUS_TH = {"auto_approved": "อนุมัติอัตโนมัติ", "needs_review": "ต้องตรวจ", "failed": "ล้มเหลว"}
STEPS = [("00_original", "ต้นฉบับ"), ("01_color", "คงสี"), ("02_deskew", "ปรับเอียง")]
LOW, HIGH = (213, 94, 0), (0, 114, 178)  # ความมั่นใจต่ำ (ส้มแดง) → สูง (น้ำเงิน) ไล่ต่อเนื่อง Okabe-Ito
ALLOWED_TAG = re.compile(r"<(/?)(table|thead|tbody|tr|td|th|br)((?:\s+(?:colspan|rowspan)=\"\d+\")*)\s*/?>", re.I)

CSS = """
:root{--bg:#fff;--fg:#1d2433;--muted:#5b6475;--card:#f5f6f8;--line:#d9dce3;
--conflict:#d55e00;--missing:#a8467f;--unverified:#8a6100;--verified:#00785a;--not_applicable:#5b6475;
--auto_approved:#00785a;--needs_review:#8a6100;--failed:#d55e00}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171f;--fg:#e6e8ee;--muted:#a0a7b7;--card:#1e222c;--line:#343a48;
--conflict:#ff9a57;--missing:#e6a0c4;--unverified:#e6b84d;--verified:#4fd1a5;--not_applicable:#a0a7b7;
--auto_approved:#4fd1a5;--needs_review:#e6b84d;--failed:#ff9a57}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.4rem;margin:.4em 0} h2{font-size:1.15rem;margin-top:2rem;border-top:1px solid var(--line);padding-top:1rem}
h3{font-size:1rem;margin:1.2rem 0 .3rem}
.note,.muted{color:var(--muted);font-size:.9rem} a{color:inherit}
img{max-width:100%;height:auto;border:1px solid var(--line)}
.tw{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top} th{background:var(--card)}
td.n{width:70px;font-size:.85rem} td img{max-width:420px}
.chip{display:inline-block;border:1.5px solid currentColor;border-radius:999px;padding:0 9px;font-size:.82rem;white-space:nowrap}
pre{white-space:pre-wrap;word-break:break-word;margin:0;background:var(--card);padding:8px;font:.85rem/1.5 ui-monospace,Consolas,monospace}
pre.raw{max-height:520px;overflow:auto}
.sum span{display:inline-block;margin-right:16px}
.bar{height:12px;max-width:340px;border-radius:6px;background:linear-gradient(90deg,rgb(213,94,0),rgb(0,114,178));margin:4px 0}
.bar-l{display:flex;justify-content:space-between;max-width:340px;font-size:.8rem;color:var(--muted)}
.rendered table td,.rendered table th{border:1px solid var(--line);padding:2px 6px;font-size:.85rem}
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def chip(kind: str, symbol: str, text: str) -> str:
    return f'<span class="chip" style="color:var(--{kind})">{symbol} {esc(text)}</span>'


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(font_path("Sarabun-Regular.ttf")), size)
    except OSError:
        return ImageFont.load_default()


def page_html(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def sanitize_table(src: str) -> str:
    """แสดงตารางที่โมเดลคืนมา: ปล่อยเฉพาะแท็กตารางพื้นฐาน ที่เหลือ escape ทั้งหมด."""
    out, last = [], 0
    for m in ALLOWED_TAG.finditer(src):
        out.append(html.escape(src[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(html.escape(src[last:]))
    return "".join(out).replace("\n", "<br>")


def conf_color(c: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * c) for a, b in zip(LOW, HIGH))


def shrink(img: Image.Image) -> tuple[Image.Image, float]:
    s = min(1.0, MAX_SIDE / max(img.size))
    return (img.resize((round(img.width * s), round(img.height * s))) if s < 1 else img.copy()), s


def save_jpg(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=85)


def doc_name(stem: str) -> str:
    return re.sub(r"_p\d+$", "", stem)


def load_results(final_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in iter_results(final_dir, "*_p*.json")}


# ---------- ขั้นที่ 1–2: ภาพนำเข้ากับ preprocess ----------

def step_image(pre: Path, name: str, key: str) -> Path:
    return find_dir(pre / "steps", name) / f"{name}_p1_{key}.png"


def preprocess_strip(pre: Path, name: str) -> tuple[Image.Image | None, str]:
    imgs, labels = [], []
    for key, label in STEPS:
        p = step_image(pre, name, key)
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
            labels.append(label)
    meta = ""
    mf = pre / "manifest.json"
    if mf.exists():
        for r in json.loads(mf.read_text(encoding="utf-8")).get("files", []):
            if Path(r["file"]).stem == name:
                meta = json.dumps(r.get("meta", {}), ensure_ascii=False)
    if not imgs:
        return None, meta
    h = 560
    thumbs = [im.resize((max(1, round(im.width * h / im.height)), h)) for im in imgs]
    canvas = Image.new("RGB", (sum(t.width for t in thumbs) + 12 * (len(thumbs) + 1), h + 44), "white")
    d, f, x = ImageDraw.Draw(canvas), font(22), 12
    for t, lab in zip(thumbs, labels):
        canvas.paste(t, (x, 38))
        d.text((x, 8), lab, fill="black", font=f)
        x += t.width + 12
    return canvas, meta


# ---------- Paddle: ผลอ่านระดับคำ ----------

def read_words(cache: Path) -> list[OCRWord] | None:
    if not cache.exists():
        return None
    return [OCRWord(w["text"], w["conf"], tuple(w["bbox"])) for w in json.loads(cache.read_text(encoding="utf-8"))["words"]]


def run_paddle(image_path: Path, cache: Path, engine) -> None:
    from fin_extractor.io import read_image

    res = engine.recognize(read_image(image_path))
    words = [w for line in res.lines for w in line.words]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"image": str(image_path.name), "words": [{"text": w.text, "conf": w.confidence, "bbox": list(w.bbox)} for w in words]},
                                ensure_ascii=False), encoding="utf-8")


def detect_image(page: Image.Image, words: list[OCRWord]) -> Image.Image:
    img, s = shrink(page)
    d = ImageDraw.Draw(img, "RGBA")
    for w in words:
        c = conf_color(w.confidence)
        d.rectangle([v * s for v in w.bbox], fill=(*c, 40), outline=c, width=2)
    return img


def lines_image(page: Image.Image, lines) -> Image.Image:
    img, s = shrink(page)
    d, f = ImageDraw.Draw(img, "RGBA"), font(22)
    for i, ln in enumerate(lines, 1):
        x0, y0, x1, y1 = (v * s for v in ln.bbox)
        d.rectangle([x0, y0, x1, y1], outline=(0, 114, 178, 220), width=3)
        label = str(i)
        w = d.textlength(label, font=f)
        d.rectangle([x0 - w - 10, y0, x0 - 2, y0 + 26], fill=(0, 114, 178))
        d.text((x0 - w - 6, y0), label, fill="white", font=f)
    return img


def paddle_sections(out: Path, page: Image.Image, words: list[OCRWord]) -> str:
    lines = group_into_lines(words)
    save_jpg(detect_image(page, words), out / "03_detect.jpg")
    save_jpg(lines_image(page, lines), out / "04_lines.jpg")
    confs = [w.confidence for w in words]
    low = sorted(words, key=lambda w: w.confidence)[:15]
    low_rows = "".join(f"<tr><td>{w.confidence:.3f}</td><td>{esc(w.text)}</td><td class='muted'>{w.bbox}</td></tr>" for w in low)
    rows = []
    (out / "lines").mkdir(exist_ok=True)
    for i, ln in enumerate(lines, 1):
        x0, y0, x1, y1 = ln.bbox
        crop = page.crop((max(0, x0 - 6), max(0, y0 - 6), min(page.width, x1 + 6), min(page.height, y1 + 6)))
        crop.thumbnail((900, 200))
        name = f"lines/{i:03d}.png"
        crop.save(out / name)
        cs = [w.confidence for w in ln.words]
        rows.append(f'<tr><td class="n">{i}</td><td><img src="{name}" loading="lazy"></td><td><pre>{esc(ln.text)}</pre></td>'
                    f'<td class="muted">{len(ln.words)} คำ<br>ต่ำสุด {min(cs):.3f}<br>เฉลี่ย {sum(cs) / len(cs):.3f}</td></tr>')
    return (f'<h2>3 · ตรวจจับคำ</h2><p class="note">Paddle ตรวจจับกรอบของทุกคำแล้วอ่านแต่ละกรอบ รวม {len(words)} คำ '
            f'สีของกรอบ = ความมั่นใจที่ Paddle รายงาน (ไล่ต่อเนื่อง ไม่มีเกณฑ์ตัดสี) ความมั่นใจเฉลี่ย {sum(confs) / len(confs):.3f}</p>'
            f'<div class="bar"></div><div class="bar-l"><span>ต่ำ {min(confs):.2f}</span><span>สูง {max(confs):.2f}</span></div>'
            f'<img src="03_detect.jpg"><h3>15 คำที่ Paddle มั่นใจน้อยที่สุด</h3>'
            f'<div class="tw"><table><tr><th>ความมั่นใจ</th><th>ข้อความที่อ่านได้</th><th>กรอบ</th></tr>{low_rows}</table></div>'
            f'<h2>4 · จัดเป็นบรรทัด</h2><p class="note">คำที่ความสูงซ้อนกันเกินครึ่งถูกจัดเป็นบรรทัดเดียว เรียงจากบนลงล่าง ซ้ายไปขวา '
            f'ได้ {len(lines)} บรรทัด (ตัวเลขที่ขอบซ้ายของกรอบ = ลำดับบรรทัด)</p><img src="04_lines.jpg">'
            f'<h2>5 · อ่านทีละบรรทัด</h2><div class="tw"><table><tr><th>#</th><th>ภาพตัด</th><th>ข้อความ</th><th>ความมั่นใจ</th></tr>{"".join(rows)}</table></div>')


def typhoon_section(raw: str, meta: dict) -> str:
    facts = ", ".join(f"{k}={meta[k]}" for k in ("engine", "seconds", "truncated") if k in meta)
    return ('<h2>3 · อ่านทั้งหน้าด้วย VLM</h2><p class="note">Typhoon OCR ได้ภาพทั้งหน้าและคืน Markdown ในครั้งเดียว '
            'ไม่มีขั้นตรวจจับกรอบ ไม่มีกรอบคำ และไม่มีความมั่นใจรายคำ (ต่างจาก Paddle) จึงดูได้แค่ข้อความที่ได้เทียบกับภาพ '
            f'ตอบไม่จบ/ถูกตัด: {esc(meta.get("truncated"))} · {esc(facts)}</p>'
            f'<div class="rendered">{sanitize_table(raw)}</div><details><summary>Markdown ดิบ</summary><pre class="raw">{esc(raw)}</pre></details>')


# ---------- ค่าที่สกัด ----------

def fields_table(r: dict) -> str:
    order = list(LEVELS)
    rows = sorted(r["field_evidence"].items(), key=lambda kv: (order.index(kv[1]["level"]), kv[0]))
    out = []
    for key, ev in rows:
        sym, label = LEVELS[ev["level"]]
        notes = [*(f"✓ {c}" for c in ev["checks_passed"]), *(f"✗ {c}" for c in ev["checks_failed"])]
        if ev.get("sources_agree") is False:
            notes.append("rules ≠ llm")
        if ev.get("grounded") is False:
            notes.append("ไม่พบในข้อความ")
        out.append(f"<tr><td><code>{esc(key)}</code></td><td>{esc(ev['value'])}</td><td>{chip(ev['level'], sym, label)}</td>"
                   f"<td>{esc(ev.get('source') or '')}</td><td>{esc(', '.join(notes))}</td></tr>")
    return ('<div class="tw"><table><tr><th>ฟิลด์</th><th>ค่า</th><th>ระดับหลักฐาน</th><th>ตัวสกัด</th><th>ตรวจ</th></tr>'
            + "".join(out) + "</table></div>")


def checks_table(r: dict) -> str:
    outcome = {True: ("verified", "✓", "ผ่าน"), False: ("failed", "✗", "ไม่ผ่าน"), None: ("not_applicable", "·", "ไม่ได้ตรวจ")}
    rows = "".join(f"<tr><td>{esc(c['name'])}</td><td>{chip(*outcome[c['passed']])}</td><td>{esc(c['detail'])}</td></tr>" for c in r["checks"])
    return f'<div class="tw"><table><tr><th>check</th><th>ผล</th><th>รายละเอียด</th></tr>{rows}</table></div>'


# ---------- ประกอบรายงาน ----------

def build(engine: str, dataset: str, profile: str, ocr_docs: list[str], paddle) -> Path:
    P = DataPaths()
    base, raw_dir = P.engine(engine, dataset), P.raw_data(engine, dataset)
    extractors = sorted(d.name for d in (base / "final").glob("*") if d.is_dir())
    if not extractors:
        raise SystemExit(f"ไม่มีผลสกัดของ {engine}/{dataset} ที่ {base / 'final'}")
    results = {x: load_results(P.final(engine, dataset, x)) for x in extractors}
    stems = sorted({s for r in results.values() for s in r})
    out_root, pre = P.visualize(engine, dataset), P.preprocess(dataset, profile)
    ext = RAW_EXT[engine]

    rows = []
    for stem in stems:
        name = doc_name(stem)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        final_png = find_dir(P.preprocess_final(dataset, profile), name) / f"{stem}.png"
        orig = step_image(pre, name, "00_original")
        steps_html = ""
        if orig.exists():
            with Image.open(orig) as im:
                save_jpg(shrink(im.convert("RGB"))[0], out / "01_input.jpg")
            steps_html += ('<h2>1 · ภาพนำเข้า</h2><p class="note">หน้าที่เรนเดอร์จากไฟล์ต้นฉบับ (PDF ที่ 300 dpi)</p><img src="01_input.jpg">')
        strip, meta_pre = preprocess_strip(pre, name)
        if strip:
            save_jpg(strip, out / "02_preprocess.jpg")
            steps_html += (f'<h2>2 · preprocess</h2><p class="note">ทำครั้งเดียวและใช้ร่วมกันทุก engine · meta: {esc(meta_pre)}</p><img src="02_preprocess.jpg">')
        elif not orig.exists():
            steps_html += '<h2>1–2 · ภาพนำเข้าและ preprocess</h2><p class="note">ไม่มีภาพทีละขั้น (รัน <code>preprocess --steps</code> เพื่อสร้าง)</p>'

        raw_path = find_dir(raw_dir, name) / f"{stem}.{ext}"
        raw = raw_path.read_text(encoding="utf-8") if raw_path.exists() else "(ไม่พบข้อความดิบ)"
        meta_path = raw_path.with_name(f"{stem}.meta.json")
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        if engine == "paddle":
            cache = out / "ocr_words.json"
            if any(name.endswith(d) for d in ocr_docs) and not cache.exists() and final_png.exists():
                run_paddle(final_png, cache, paddle())
            words = read_words(cache)
            if words and final_png.exists():
                with Image.open(final_png) as im:
                    steps_html += paddle_sections(out, im.convert("RGB"), words)
            else:
                steps_html += ('<h2>3–5 · ผลระดับคำของ Paddle</h2><p class="note">ยังไม่มีผลระดับคำของไฟล์นี้ (ผลที่เก็บไว้มีแค่ข้อความ) '
                               f'รัน <code>visualize_engine.py paddle --ocr {esc(name.rsplit("_", 1)[-1])}</code> เพื่อให้ Paddle อ่านภาพนี้ใหม่</p>')
        else:
            steps_html += typhoon_section(raw, raw_meta)

        sections = []
        for x in extractors:
            r = results[x].get(stem)
            if r:
                sections.append(f'<h2>ค่าที่สกัด · {esc(x)} {chip(r["status"], "●", STATUS_TH.get(r["status"], r["status"]))}</h2>'
                                f'<p class="note">ขาด: {esc(", ".join(r["missing_fields"]) or "-")} · engine: {esc(r["engine"])}</p>'
                                f'{fields_table(r)}<h3>checks</h3>{checks_table(r)}')
        body = (f'<p><a href="../index.html">← ทุกเอกสาร</a></p><h1>{esc(name)} <span class="muted">{esc(engine)}</span></h1>{steps_html}'
                f'{"".join(sections)}<h2>ข้อความดิบ ({esc(engine)})</h2><pre class="raw">{esc(raw)}</pre>')
        (out / "index.html").write_text(page_html(f"{name} · {engine}", body), encoding="utf-8")

        cells = "".join(f"<td>{chip(results[x][stem]['status'], '●', STATUS_TH.get(results[x][stem]['status'], ''))}</td>"
                        if stem in results[x] else "<td>-</td>" for x in extractors)
        first = next(results[x][stem] for x in extractors if stem in results[x])
        has_words = engine == "paddle" and (out / "ocr_words.json").exists()
        rows.append(f'<tr><td><a href="{esc(name)}/index.html">{esc(name)}</a></td><td>{esc(first["data"].get("document_type"))}</td>{cells}'
                    f'<td>{esc(", ".join(first["missing_fields"]) or "-")}</td><td>{"มี" if has_words else "-"}</td></tr>')

    summary = "".join(f'<p class="sum"><b>{esc(x)}</b> ' + "".join(
        f"<span>{chip(k, '●', STATUS_TH.get(k, k))} {n}</span>" for k, n in Counter(r["status"] for r in results[x].values()).items()) + "</p>"
        for x in extractors)
    head = "".join(f"<th>{esc(x)}</th>" for x in extractors)
    extra = "<th>กรอบคำ</th>"
    body = (f"<h1>{esc(engine)} · {esc(dataset)}</h1><p class='muted'>{len(stems)} เอกสาร · แต่ละหน้าแสดงทีละขั้นตั้งแต่ภาพนำเข้าถึงค่าที่สกัด</p>{summary}"
            f'<div class="tw"><table><tr><th>เอกสาร</th><th>ชนิด</th>{head}<th>ขาด (extractor แรก)</th>{extra}</tr>{"".join(rows)}</table></div>')
    (out_root / "index.html").write_text(page_html(f"{engine} · {dataset}", body), encoding="utf-8")
    return out_root / "index.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", default=["paddle", "typhoon"], choices=["paddle", "typhoon"])
    ap.add_argument("--dataset", default="finance_doc_scan")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--ocr", nargs="*", default=[], help="ไฟล์ (ส่วนท้ายของชื่อ เช่น 001312) ที่ให้ Paddle อ่านใหม่เพื่อเก็บกรอบคำ (หนัก ทำทีละไฟล์)")
    args = ap.parse_args()

    engine_cache = []

    def paddle():  # โหลด Paddle เมื่อจำเป็นเท่านั้น และโหลดครั้งเดียว
        if not engine_cache:
            from fin_extractor.config import load_config
            from fin_extractor.ocr import create_engine

            engine_cache.append(create_engine("paddle", load_config()))
        return engine_cache[0]

    for e in args.engines:
        print(build(e, args.dataset, args.profile, args.ocr, paddle))


if __name__ == "__main__":
    main()
