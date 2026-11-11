"""รายงานภาพ "ทำไม OXR ไม่แม่นกับภาษาไทย" จากข้อมูลที่วัดไว้แล้ว (ไม่เรียก OXR/GPU)

ข้อมูลที่ใช้ (ทั้งหมดอยู่ใต้ data/ ซึ่งไม่เข้า git):
  result_ocr/oxr/<dataset>/raw_data/*.pdf.oxr.json   ผล OXR ของสแกนจริง 18 ใบ (รอบ PDF ดิบ)
  result_ocr/paddle|typhoon/<dataset>/raw_data/      ข้อความดิบของ Paddle / Typhoon
  benchmarks/*cer*.json                              CER ภาพสังเคราะห์ของ Paddle / Typhoon / OXR
  compare/oxr_why/*.json                             ค่าที่วัดแยก: ความน่าจะเป็นของ token, จำนวน token ใน vocabulary, บรรทัดจริงที่อ่านจากภาพ
  compare/english/text_recall.json                   ค่าสำคัญของเอกสารอังกฤษที่ปรากฏในข้อความของแต่ละ engine
  result_ocr/oxr/<dataset>/visualize/<ไฟล์>_pdf/     ภาพ layout และภาพตัดบล็อก (สร้างด้วย scripts/visualize_oxr.py ก่อน)

ผล: data/reports/oxr_why/index.html (มีภาพเอกสารจริง เก็บในเครื่องเท่านั้น ลบทิ้งได้)
ไฟล์ compare/oxr_why/truth_1312.json มีค่าจริงจากเอกสาร จึงไม่ฝังในโค้ดนี้

    .venv/Scripts/python scripts/explain_oxr.py
"""

from __future__ import annotations

import difflib
import html
import json
import math
import re
import shutil
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fin_extractor.extraction import markdown_to_lines  # noqa: E402
from visualize_oxr import sanitize_table  # noqa: E402

from fin_extractor.paths import DataPaths  # noqa: E402

P = DataPaths()
DATASET = "finance_doc_scan"
REPORT = P.reports("oxr_why")
PREFIX = "BRWAC50DE3689DA_"
TH = re.compile(r"[฀-๿]")
LAT = re.compile(r"[A-Za-z]")
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
e = html.escape

# ---------- ข้อมูล ----------


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def real_rows() -> list[dict]:
    rows = []
    for f in sorted(P.raw_data("oxr", DATASET).glob("*_p1.pdf.oxr.json")):
        stem = f.name.split("_p1")[0][-6:]
        data = load(f)["data"]
        md = data["result"]["outputs"]["markdown"]
        paddle = (P.raw_data("paddle", DATASET) / f"{PREFIX}{stem}_p1.txt").read_text(encoding="utf-8")
        th, la = len(TH.findall(paddle)), len(LAT.findall(paddle))
        windows = [md[i:i + 20] for i in range(0, max(1, len(md) - 20), 5)]
        top = max((windows.count(w) for w in set(windows)), default=0) / max(1, len(windows))
        rows.append({"stem": stem, "share": th / max(1, th + la), "cjk": len(CJK.findall(md)), "chars": len(md),
                     "sec": data["metadata"]["processing_time_ms"] / 1000, "repeat": top})
    return rows


def cer_rows() -> dict[str, list[float]]:
    b = P.benchmarks()
    return {
        "Paddle": [r["cer"] for r in load(b / "cer_synthetic_paddle.json")["engines"]["paddle"]["files"]],
        "Typhoon": [r["cer"] for r in load(b / "cer_synthetic_vlm.json")["engines"]["vlm"]["files"]],
        "OXR": [r["cer"] for r in load(b / "oxr_synthetic_preprocessed.json")["files"]],
    }


# ---------- SVG ----------
W = 760
ML, MR, MT, MB = 64, 24, 20, 52


def svg_open(h: int, label: str) -> str:
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{e(label)}" class="chart">'


def axis_text(x, y, t, anchor="middle", cls="tick") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{e(str(t))}</text>'


def scatter(points, *, h, xmax, ymax, xlab, ylab, xfmt, yfmt, ylog=False, yticks=(), xticks=(), label_when=None, title=""):
    pw, ph = W - ML - MR, h - MT - MB

    def fy(v):
        return (math.log10(1 + v) / math.log10(1 + ymax) if ylog else v / ymax)

    out = [svg_open(h, title)]
    for t in yticks:
        y = MT + ph * (1 - fy(t))
        out += [f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>', axis_text(ML - 8, y + 4, yfmt(t), "end")]
    for t in xticks:
        x = ML + pw * t / xmax
        out.append(axis_text(x, h - MB + 18, xfmt(t)))
    out += [f'<line x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}" class="axis"/>',
            axis_text(ML + pw / 2, h - 10, xlab, cls="axlab"),
            f'<text transform="translate(16 {MT + ph / 2:.0f}) rotate(-90)" text-anchor="middle" class="axlab">{e(ylab)}</text>']
    placed: list[tuple[float, float, float, float]] = []
    for p in points:
        x, y = ML + pw * p["x"] / xmax, MT + ph * (1 - fy(p["y"]))
        out.append(f'<g class="pt"><circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="dot"/><title>{e(p["tip"])}</title></g>')
    for p in points:  # ป้ายวางทีหลังจุดทั้งหมด เลือกตำแหน่งที่ไม่ชนขอบหรือป้ายอื่น
        if not (label_when and label_when(p)):
            continue
        x, y = ML + pw * p["x"] / xmax, MT + ph * (1 - fy(p["y"]))
        tw = 8.2 * len(p["label"])
        for dx, dy, anchor in ((10, 4, "start"), (-10, 4, "end"), (0, -14, "middle"), (0, 24, "middle"), (10, -12, "start"), (-10, -12, "end"), (10, 20, "start")):
            x0 = x + dx - (tw if anchor == "end" else tw / 2 if anchor == "middle" else 0)
            box = (x0, y + dy - 13, x0 + tw, y + dy + 4)
            if box[0] < ML - 4 or box[2] > W - 6 or any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
                continue
            placed.append(box)
            out.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" class="dlabel">{e(p["label"])}</text>')
            break
    out.append("</svg>")
    return "".join(out)


def strip_cer(data: dict[str, list[float]], colors: dict[str, str]) -> str:
    lo, hi = math.log10(0.001), math.log10(100)
    pw, row = W - 150 - MR, 74
    h = MT + row * len(data) + 48
    out = [svg_open(h, "CER ของแต่ละ engine บนภาพสังเคราะห์ 21 ภาพ (แกนล็อก)")]
    for t in (0.001, 0.01, 0.1, 1, 10, 100):
        x = 150 + pw * (math.log10(t) - lo) / (hi - lo)
        out += [f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{h - 40}" class="grid"/>', axis_text(x, h - 22, f"{t:g}")]
    out.append(axis_text(150 + pw / 2, h - 4, "CER (ยิ่งต่ำยิ่งดี, แกนล็อก; ค่า 0 วางที่ 0.001)", cls="axlab"))
    for i, (name, vals) in enumerate(data.items()):
        y = MT + row * i + row / 2
        out.append(f'<text x="{ML - 52}" y="{y + 4:.1f}" class="rowlab">{e(name)}</text>')
        out.append(f'<line x1="150" x2="{150 + pw}" y1="{y}" y2="{y}" class="axis"/>')
        for k, v in enumerate(vals):
            x = 150 + pw * (math.log10(max(v, 0.001)) - lo) / (hi - lo)
            out.append(f'<g class="pt"><circle cx="{x:.1f}" cy="{y + (k % 3 - 1) * 9}" r="6" fill="{colors[name]}" class="ring"/>'
                       f'<title>{e(name)} · ภาพที่ {k + 1} · CER {v:.4f}</title></g>')
        m = median(vals)
        mx = 150 + pw * (math.log10(max(m, 0.001)) - lo) / (hi - lo)
        out += [f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{y - 24}" y2="{y + 24}" class="median"/>',
                f'<text x="{mx + 6:.1f}" y="{y - 26}" class="dlabel">มัธยฐาน {m:.4f}</text>']
    out.append("</svg>")
    return "".join(out)


def token_bars(tokens: list[dict]) -> str:
    n = len(tokens)
    h = 360
    pw, ph = W - ML - MR, h - MT - 96
    bw = pw / n
    out = [svg_open(h, "ความน่าจะเป็นของ token ที่โมเดลเลือก ทีละ token")]
    for t in (0, 0.25, 0.5, 0.75, 1):
        y = MT + ph * (1 - t)
        out += [f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>', axis_text(ML - 8, y + 4, f"{t:.2f}", "end")]
    for i, t in enumerate(tokens):
        tok = t["t"]
        cls = ("cjk" if CJK.search(tok) else "thai" if TH.search(tok) else "markup" if re.fullmatch(r"[<>/​␣()]*|td|\)</|</|>|\(byte\)", tok) else "latin")
        x = ML + bw * i + 3
        hgt = max(ph * t["p"], 2)
        alts = " · ".join(f"{a[0]} {a[1]:.2f}" for a in t["alts"]) or "—"
        out.append(f'<g class="pt"><rect x="{x:.1f}" y="{MT + ph - hgt:.1f}" width="{bw - 6:.1f}" height="{hgt:.1f}" rx="3" class="bar-{cls}"/>'
                   f'<title>token #{i + 1} "{e(tok)}" p={t["p"]:.2f} · ตัวเลือกถัดไป: {e(alts)}</title></g>')
        out.append(f'<text transform="translate({x + bw / 2 - 3:.1f} {MT + ph + 12}) rotate(60)" class="tok">{e(tok)}</text>')
    out += [axis_text(ML + pw / 2, h - 6, "token ที่โมเดลสร้าง เรียงตามลำดับ (ซ้าย → ขวา)", cls="axlab"),
            f'<text transform="translate(16 {MT + ph / 2:.0f}) rotate(-90)" text-anchor="middle" class="axlab">ความน่าจะเป็นของ token ที่เลือก</text>', "</svg>"]
    return "".join(out)


def hbars(items: list[tuple[str, int, str]], colors: list[str]) -> str:
    row, left = 54, 190
    h = MT + row * len(items) + 16
    pw = W - left - 110
    vmax = max(v for _, v, _ in items)
    out = [svg_open(h, "จำนวน token ใน vocabulary แยกตามอักษร")]
    for i, (name, v, tip) in enumerate(items):
        y = MT + row * i + 8
        w = pw * v / vmax
        out += [f'<text x="{left - 10}" y="{y + 20}" text-anchor="end" class="rowlab">{e(name)}</text>',
                f'<g class="pt"><rect x="{left}" y="{y}" width="{w:.1f}" height="30" rx="4" fill="{colors[i]}"/><title>{e(tip)}</title></g>',
                f'<text x="{left + w + 8:.1f}" y="{y + 21}" class="dlabel">{v:,}</text>']
    out.append("</svg>")
    return "".join(out)


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><tr>{head}</tr>{body}</table></div>'


def details(summary: str, inner: str) -> str:
    return f"<details><summary>{e(summary)}</summary>{inner}</details>"


def norm(s: str) -> str:
    return "".join(s.split()).casefold()


def best_line(lines: list[str], truth: str) -> tuple[str, float]:
    best = max(lines, key=lambda l: difflib.SequenceMatcher(None, norm(l), norm(truth)).ratio())
    return best, difflib.SequenceMatcher(None, norm(best), norm(truth)).ratio()


def sim_chip(r: float) -> str:
    return f'<span class="sim">ใกล้เคียง {r * 100:.0f}%</span>'


BADGES = {"m": ("วัดแล้ว", "b-m"), "d": ("ตามที่ผู้พัฒนาระบุ", "b-d"), "i": ("อนุมาน", "b-i"), "u": ("ยังไม่ทดสอบ", "b-u")}


def badge(*keys: str) -> str:
    return "".join(f'<span class="badge {BADGES[k][1]}">{BADGES[k][0]}</span>' for k in keys)


# ---------- สร้างรายงาน ----------

def main() -> None:
    why = P.compare("oxr_why")
    rows = real_rows()
    cers = cer_rows()
    lp = load(why / "logprobs_line1.json")
    vocab = load(why / "vocab.json")
    truth = load(why / "truth_1312.json")
    recall = load(P.compare("english") / "text_recall.json")

    (REPORT / "img").mkdir(parents=True, exist_ok=True)
    vis = P.visualize("oxr", DATASET)
    for src, name in ((vis / "001312_pdf" / "03_layout.png", "layout_1312.png"), (vis / "001315_pdf" / "03_layout.png", "layout_1315.png"),
                      (vis / "001312_pdf" / "blocks" / "13_Table.png", "table_1312.png")):
        shutil.copy(src, REPORT / "img" / name)

    c_oxr, c_pad, c_typ = "var(--s-oxr)", "var(--s-paddle)", "var(--s-typhoon)"

    # กราฟ 1: ภาษา -> อักษรจีนที่หลุด
    pts = [{"x": r["share"] * 100, "y": r["cjk"], "label": r["stem"], "tip":
            f'ไฟล์ {r["stem"]} · ไทย {r["share"] * 100:.0f}% ของข้อความ · OXR ส่งอักษรจีน {r["cjk"]:,} ตัว ผลยาว {r["chars"]:,} ตัวอักษร'} for r in rows]
    chart_lang = scatter(pts, h=380, xmax=100, ymax=10000, ylog=True, xlab="สัดส่วนตัวอักษรไทยในเอกสาร (%, วัดจากข้อความ Paddle)",
                         ylab="อักษรจีนที่ OXR ส่งออกมา (ตัว, แกนล็อก)", xfmt=lambda v: f"{v:g}%", yfmt=lambda v: f"{v:,}",
                         yticks=(0, 10, 100, 1000, 10000), xticks=(0, 20, 40, 60, 80, 100),
                         label_when=lambda p: p["x"] < 15 or p["y"] > 2000, title="ภาษาของเอกสารกับอักษรจีนที่หลุด")
    lang_table = table(["ไฟล์", "ไทย %", "อักษรจีน (ตัว)", "ความยาวผล (ตัวอักษร)", "เวลา (วินาที)"],
                       [[r["stem"], f'{r["share"] * 100:.0f}', f'{r["cjk"]:,}', f'{r["chars"]:,}', f'{r["sec"]:.0f}'] for r in sorted(rows, key=lambda r: r["share"])])

    low = sorted(r["cjk"] for r in rows if r["share"] < 0.2)
    high = sorted(r["cjk"] for r in rows if r["share"] >= 0.6)
    low_txt = ", ".join(str(v) for v in low)
    hi_min, hi_max, hi_med = high[0], high[-1], median(high)
    n_long = sum(r["chars"] > 10000 for r in rows)

    # กราฟ 2: ความยาวผลกับเวลา
    pts2 = [{"x": r["chars"], "y": r["sec"], "label": r["stem"], "tip": f'ไฟล์ {r["stem"]} · ผลยาว {r["chars"]:,} ตัวอักษร · {r["sec"]:.0f} วินาที · ข้อความซ้ำ {r["repeat"] * 100:.0f}%'} for r in rows]
    chart_len = scatter(pts2, h=340, xmax=55000, ymax=340, xlab="ความยาวผลที่ OXR ตอบ (ตัวอักษร)", ylab="เวลาต่อหน้า (วินาที)",
                        xfmt=lambda v: f"{v // 1000}k", yfmt=lambda v: f"{v:g}", yticks=(0, 100, 200, 300), xticks=(0, 10000, 20000, 30000, 40000, 50000),
                        label_when=lambda p: p["x"] > 10000, title="ผลยาวผิดปกติกับเวลาที่ใช้")
    len_table = table(["ไฟล์", "ความยาวผล", "เวลา (วินาที)", "ข้อความซ้ำสูงสุด (%)"],
                      [[r["stem"], f'{r["chars"]:,}', f'{r["sec"]:.0f}', f'{r["repeat"] * 100:.0f}'] for r in sorted(rows, key=lambda r: -r["chars"])])

    # กราฟ 3: CER
    colors = {"OXR": c_oxr, "Paddle": c_pad, "Typhoon": c_typ}
    chart_cer = strip_cer({k: cers[k] for k in ("Paddle", "Typhoon", "OXR")}, colors)
    cer_table = table(["engine", "ภาพ (n)", "CER มัธยฐาน", "CER ต่ำสุด", "CER สูงสุด"],
                      [[k, str(len(v)), f"{median(v):.4f}", f"{min(v):.4f}", f"{max(v):.2f}"] for k, v in cers.items()])

    # กราฟ 4: vocabulary
    chart_vocab = hbars([("ภาษาไทย", vocab["thai"], "token ที่มีอักษรไทย"), ("ภาษาจีน/ญี่ปุ่น/เกาหลี", vocab["cjk"], "token ที่มีอักษร CJK"),
                         ("ตัวอักษรละติน (ล้วน)", vocab["latin_letters_only"], "token ที่เป็นตัวอักษรอังกฤษล้วน (รวมคำอังกฤษ/โค้ด)")],
                        [c_oxr, c_oxr, c_oxr])
    ratio = vocab["cjk"] / vocab["thai"]

    # กราฟ 5: token
    chart_tok = token_bars(lp["tokens"])
    tok_table = table(["#", "token", "p", "ตัวเลือกถัดไป"],
                      [[str(i + 1), e(t["t"]), f'{t["p"]:.2f}', e(" · ".join(f"{a[0]} {a[1]:.2f}" for a in t["alts"]) or "—")] for i, t in enumerate(lp["tokens"])])

    # เทียบบล็อกเดียวกันของแต่ละ engine (001312)
    oxr = load(P.raw_data("oxr", DATASET) / f"{PREFIX}001312_p1.pdf.oxr.json")["data"]["result"]["outputs"]["json"]["pages"][0]["blocks"]
    by_idx = {b["idx"]: b for b in oxr}
    pad_lines = [l for l in (P.raw_data("paddle", DATASET) / f"{PREFIX}001312_p1.txt").read_text(encoding="utf-8").split("\n") if l.strip()]
    typ_lines = markdown_to_lines((P.raw_data("typhoon", DATASET) / f"{PREFIX}001312_p1.md").read_text(encoding="utf-8"))
    cmp_rows = []
    for key in sorted(truth, key=lambda k: int(k) if k.isdigit() else 0):
        if not key.isdigit():
            continue
        i, tr = int(key), truth[key]
        b = by_idx[i]
        src = next((vis / "001312_pdf" / "blocks").glob(f"{i:02d}_*.png"))
        shutil.copy(src, REPORT / "img" / f"crop_{i:02d}.png")
        pl, pr = best_line(pad_lines, tr)
        tl, tr_ = best_line(typ_lines, tr)
        orr = difflib.SequenceMatcher(None, norm(b["content"]), norm(tr)).ratio()
        cmp_rows.append([f'<img src="img/crop_{i:02d}.png" alt="ภาพบล็อก {i}">', e(tr),
                         f"{e(pl)}<br>{sim_chip(pr)}", f"{e(tl)}<br>{sim_chip(tr_)}", f"{e(b['content'])}<br>{sim_chip(orr)}"])
    cmp_table = table(["บล็อกที่ OXR ตัดออกมา", "ข้อความจริง (อ่านจากภาพ)", "Paddle (บรรทัดที่ใกล้ที่สุด)", "Typhoon (บรรทัดที่ใกล้ที่สุด)", "OXR (ผลของบล็อกนั้น)"], cmp_rows)

    # ตารางทั้งก้อน
    tbl_html = sanitize_table(by_idx[13]["content"])

    # เอกสารอังกฤษ
    rec_rows = []
    for stem, per in recall.items():
        rec_rows.append([stem] + [f'{per[k]["found"]}/{per[k]["of"]}' for k in ("Paddle", "Typhoon", "OXR")])
    rec_table = table(["ไฟล์ (อังกฤษเป็นหลัก)", "Paddle", "Typhoon", "OXR"], rec_rows)

    loop_src = load(P.raw_data("oxr", "synthetic") / "preprocessed" / "skew_+6.0.json")["data"]["result"]["outputs"]["markdown"]
    loop_snip = loop_src[:150] + " … " + loop_src[-60:]

    css = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#d9d8d3;--card:#f3f2ee;--s-oxr:#2a78d6;--s-paddle:#eb6834;--s-typhoon:#1baf7a;--ok:#1d6b3a;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--card:#242423;--s-oxr:#3987e5;--s-paddle:#d95926;--s-typhoon:#199e70;--ok:#6fcf97;--bad:#f2867e}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px 16px 64px}
h1{font-size:1.55rem;margin:.6em 0 .2em} h2{font-size:1.2rem;margin-top:2.4rem;border-top:1px solid var(--line);padding-top:1.1rem} h3{font-size:1rem;margin:1.4rem 0 .3rem}
.muted,.note{color:var(--muted);font-size:.9rem} .lead{font-size:1.05rem}
.badge{display:inline-block;font-size:.78rem;border-radius:999px;padding:0 9px;margin:0 6px 0 0;border:1px solid var(--line);vertical-align:middle}
.b-m{background:var(--card)} .b-d{background:var(--card);font-style:italic} .b-i{border-style:dashed} .b-u{border-style:dotted;color:var(--muted)}
.chart{width:100%;height:auto;display:block;margin:.4rem 0} .grid{stroke:var(--line);stroke-width:1} .axis{stroke:var(--muted);stroke-width:1.2}
.tick,.tok{fill:var(--muted);font-size:12px;font-family:ui-monospace,Consolas,monospace} .axlab,.rowlab,.dlabel{fill:var(--fg);font-size:13px}
.rowlab{font-weight:600} .dot{fill:var(--s-oxr);stroke:var(--bg);stroke-width:2} .ring{stroke:var(--bg);stroke-width:2}
.pt:hover circle,.pt:hover rect{stroke:var(--fg);stroke-width:2.5} .median{stroke:var(--fg);stroke-width:2}
.bar-thai{fill:var(--s-oxr)} .bar-cjk{fill:var(--s-paddle)} .bar-latin{fill:var(--s-typhoon)} .bar-markup{fill:var(--muted);opacity:.55}
.legend span{display:inline-block;margin-right:14px;font-size:.9rem} .sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.tw{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.92rem} th,td{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
td img{max-width:260px;height:auto;border:1px solid var(--line)} .sim{display:inline-block;margin-top:4px;font-size:.8rem;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 6px}
img.full{max-width:100%;height:auto;border:1px solid var(--line)} .two{display:grid;grid-template-columns:1fr 1fr;gap:12px} @media(max-width:720px){.two{grid-template-columns:1fr}}
pre{white-space:pre-wrap;word-break:break-word;background:var(--card);padding:8px;font-size:.85rem}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:stretch;margin:.6rem 0} .flow>div{flex:1 1 140px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:.9rem}
.flow .ok{border-left:5px solid var(--ok)} .flow .bad{border-left:5px solid var(--bad)} .flow .mid{border-left:5px dashed var(--muted)}
.callout{background:var(--card);border-left:4px solid var(--s-oxr);padding:8px 12px;margin:.8rem 0} details{margin:.4rem 0} summary{cursor:pointer;color:var(--muted);font-size:.9rem}
.rendered table td,.rendered table th{border:1px solid var(--line);padding:2px 6px;font-size:.85rem}
"""
    html_doc = f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ทำไม OXR ไม่แม่นกับภาษาไทย</title><style>{css}</style></head><body><main>
<h1>ทำไม OXR ไม่แม่นกับภาษาไทย</h1>
<p class="muted">รวมข้อมูลที่วัดไว้ในโปรเจกต์นี้ · เอกสารจริง เก็บในเครื่องเท่านั้น · ทุกข้อสรุปติดป้ายว่ามีหลักฐานระดับไหน:
{badge("m")} = วัดเองในโปรเจกต์ · {badge("d")} = ผู้พัฒนาระบุเอง · {badge("i")} = ผมอนุมานเอง · {badge("u")} = ยังไม่ได้ทดสอบ</p>

<h2>สรุปหนึ่งหน้า</h2>
<div class="flow">
<div class="ok"><b>1 layout</b><br>แบ่งกรอบ/ชนิดบล็อกได้ดี {badge("m")}</div>
<div class="mid"><b>2 ลำดับการอ่าน</b><br>เรียงสมเหตุสมผลในใบที่ดูแล้ว {badge("m")}</div>
<div class="bad"><b>3 อ่านตัวอักษร</b><br>อังกฤษ/ตัวเลขถูก <b>ไทยผิด</b> {badge("m")}</div>
<div class="mid"><b>4 รวมผล</b><br>แยกป้ายกับค่าคนละบล็อก ทำให้ rules จับคู่ยาก {badge("m")}</div></div>
<p class="lead">ต้นเหตุคือขั้นที่ 3: โมเดลที่อ่านตัวอักษรได้รับการฝึกจากจีน+อังกฤษเป็นหลัก {badge("d")} จึง "ไม่รู้จัก" รูปร่างตัวไทย พอไม่รู้จักมันไม่ตอบว่าไม่รู้ แต่ <b>เดา</b> ทีละ token จากภาษาที่คุ้นที่สุด ผลคืออักษรจีน ข้อความที่จำมา และการวนซ้ำ ส่วนขั้นอื่นไม่ใช่สาเหตุหลัก (ดูหัวข้อ 9)</p>

<h2>1 OXR ถูกออกแบบมาเพื่ออะไร {badge("d")}</h2>
<ul><li>เป็นโมเดลอ่านเอกสาร 1.3B ที่ฝึกจากศูนย์ ผู้พัฒนาระบุว่า <i>"ฝึกจากจีนและอังกฤษเป็นหลัก ภาษาอื่นไม่รับประกัน ไม่เน้นหลายภาษาในรุ่นนี้"</i> และ <b>ไม่มีผลประเมินแยกรายภาษา</b></li>
<li>คะแนนนำที่เขาโชว์ (olmOCR-bench, OmniDocBench, DocParsingBench) เป็นเอกสารดิจิทัลจีน/อังกฤษ ภาษาไทยไม่ปรากฏทั้งใน README, FAQ และเอกสาร benchmark</li>
<li>โมเดลไม่มีตัวเลือกภาษา prompt ตายตัวเป็นภาษาอังกฤษ (<code>Recognize the content of the image.</code>) {badge("m")}</li></ul>
<p class="note">ดังนั้น "ไม่แม่นกับไทย" ไม่ใช่ข้อบกพร่องที่ผู้พัฒนาไม่รู้ แต่เป็นขอบเขตที่เขาประกาศไว้เอง</p>

<h2>2 ขั้น layout ทำงานดี (ปัญหาไม่ได้อยู่ตรงนี้) {badge("m")}</h2>
<div class="two"><figure><img class="full" src="img/layout_1312.png" alt="layout 001312"><figcaption class="note">001312: โลโก้/ลายเซ็น/QR = Picture (ไม่ OCR), ที่อยู่ทีละบรรทัด = Text, ตาราง = Table</figcaption></figure>
<figure><img class="full" src="img/layout_1315.png" alt="layout 001315"><figcaption class="note">001315 (สแกนตะแคง): ได้บล็อกใหญ่บล็อกเดียว ไม่แบ่งแถว แต่ข้อความอังกฤษข้างในยังอ่านถูก</figcaption></figure></div>
<p>กรอบตรงตำแหน่งจริงในทั้งสองใบ จุดที่ผิดมีเล็กน้อย เช่น หัวข้อ "ออกให้กับ" ถูกตัดเป็น Title ชิ้นเล็ก</p>

<h2>3 ขั้นอ่านตัวอักษร: ภาพเดียวกัน แต่ละ engine อ่านต่างกัน {badge("m")}</h2>
<p>ภาพที่ OXR ตัดออกมาเป็นบล็อกของ 001312 เทียบกับข้อความจริง และสิ่งที่ Paddle/Typhoon ให้ในบรรทัดที่ใกล้ที่สุด ("ใกล้เคียง" = อัตราส่วนความเหมือนของข้อความ เมื่อตัดช่องว่าง 100% = เหมือนเป๊ะ)</p>
{cmp_table}
<p class="note">ข้อความจริงคือบรรทัดที่ผมอ่านจากภาพเอง ไม่ใช่เฉลยทางการ บรรทัดที่ใกล้ที่สุดของ Paddle/Typhoon เลือกอัตโนมัติจากความเหมือน อาจไม่ใช่บรรทัดเดียวกับบล็อกเสมอไป</p>
<h3>ตารางทั้งก้อน: ภาพที่ส่ง vs ผลที่ได้</h3>
<div class="two"><div><img class="full" src="img/table_1312.png" alt="ตารางรายการสินค้า"></div><div class="rendered">{tbl_html}</div></div>
<p class="note">OXR ส่งทั้งตารางเป็นก้อนเดียวและตอบเป็น HTML เลขในคอลัมน์ถูกส่วนใหญ่ แต่หัวคอลัมน์และชื่อสินค้าไทยเป็นคำอังกฤษที่ไม่มีในภาพ</p>

<h2>4 ทำไมถึงเป็นอักษรจีน (1): ศัพท์ที่โมเดลมี {badge("m")}</h2>
{chart_vocab}
<p>OXR สร้างข้อความโดยเลือก token จากคลังศัพท์ {vocab["total_tokens"]:,} ตัว ในนั้นมี token ภาษาจีน/ญี่ปุ่น/เกาหลี {vocab["cjk"]:,} ตัว แต่ภาษาไทยแค่ {vocab["thai"]:,} ตัว (<b>น้อยกว่าราว {ratio:.0f} เท่า</b>) ตัวไทยมีสระและวรรณยุกต์ที่ประกอบกันหลายชั้น จึงถูกหั่นเป็นเศษเล็ก ๆ เมื่อโมเดลไม่แน่ใจ ตัวเลือกที่มีมากกว่าจึงเข้ามาแทน</p>
{details("ตารางข้อมูล", table(["กลุ่ม", "จำนวน token"], [["ไทย", f'{vocab["thai"]:,}'], ["CJK", f'{vocab["cjk"]:,}'], ["ละติน (ล้วน)", f'{vocab["latin_letters_only"]:,}']]))}
<p class="note">{e(vocab["_note"])}</p>

<h2>5 ทำไมถึงเป็นอักษรจีน (2): โมเดลเดาทีละ token {badge("m")}</h2>
<p>ผมส่ง <b>บรรทัดไทยสะอาดบรรทัดเดียว</b> ("บริษัท ตัวอย่างเทคโนโลยี จำกัด") เข้าโมเดลตรง ๆ แล้วดูความน่าจะเป็นของ token ที่มันเลือก แท่งยิ่งเตี้ย = ยิ่งไม่มั่นใจ ชี้ที่แท่งเพื่อดูตัวเลือกอันดับถัดไป</p>
<p class="legend"><span><i class="sw" style="background:var(--s-oxr)"></i>ตัวอักษรไทย</span><span><i class="sw" style="background:var(--s-paddle)"></i>อักษรจีน</span><span><i class="sw" style="background:var(--s-typhoon)"></i>ละติน/คำอังกฤษ</span><span><i class="sw" style="background:var(--muted);opacity:.55"></i>แท็ก HTML/ตัวคั่น</span></p>
{chart_tok}
<ol><li><b>ที่ตัวอักษรไทย</b> (แท่งสีน้ำเงินช่วงแรก): token ที่เลือกได้ความน่าจะเป็นแค่ 0.01–0.09 เป็นส่วนใหญ่ และตัวเลือกถัดไปกระจายข้ามภาษา มีทั้งไทย อังกฤษ และจีน (<code>处于</code>, <code>你想</code>, <code>面积</code>) = features ของตัวไทยชี้ไม่ชัดว่าเป็นตัวไหน</li>
<li><b>แล้วโมเดลหลุดไปพิมพ์แท็กตาราง</b> (<code>&lt;/td&gt;</code>) ทั้งที่ภาพเป็นบรรทัดข้อความ เพราะนั่นเป็นรูปแบบที่คุ้นจากข้อมูลฝึก</li>
<li><b>แล้ววนซ้ำ:</b> <code>面epsilon</code> ความน่าจะเป็นไต่ 0.04 → 0.20 → 0.58 → 0.99 ยิ่งซ้ำยิ่งมั่นใจ เพราะประวัติข้อความที่ตัวเองสร้างกลายเป็นสิ่งที่ดึงให้ซ้ำต่อ</li></ol>
{details("ตารางข้อมูล", tok_table)}
<div class="callout"><b>ที่ตัวเลข 0.01–0.09 วัดจากบรรทัดเดียว</b> (greedy ครั้งเดียว) {badge("m")} ส่วนที่ว่า "สาเหตุคือข้อมูลฝึกขาดภาษาไทย" เป็นการ {badge("i")}: สอดคล้องกับ FAQ ของผู้พัฒนาและรูปแบบผลทั้งหมด แต่ข้อมูลฝึกจริงไม่เปิดเผย</div>
<p class="note">ข้อความที่โมเดลตอบจริง: <code>{e(lp["model_output"])}</code></p>

<h2>6 ภาษาของเอกสารกำหนดผล {badge("m")}</h2>
{chart_lang}
<p>แต่ละจุดคือเอกสารจริงหนึ่งใบ ใบที่ไทยน้อย (ต่ำกว่า 20%) ได้อักษรจีน {low_txt} ตัว ส่วนใบที่ไทยมาก (ตั้งแต่ 60%) ได้ตั้งแต่ {hi_min:,} ถึง {hi_max:,} ตัว (มัธยฐาน {hi_med:,.0f})
<b>ค่าสูงเป็นพันเกิดกับใบที่ผลวนซ้ำ</b> (หัวข้อ 7) ไม่ใช่แค่สัดส่วนไทย จึงอย่าอ่านเป็นเส้นแนวโน้มเรียบ ๆ ข้อมูลบอกได้แค่ว่า ใบอังกฤษแทบไม่หลุด ใบไทยหลุดทุกใบ</p>
{details("ตารางข้อมูลของทุกใบ", lang_table)}
<h3>ใบอังกฤษ: อ่านได้ดีเท่า Paddle {badge("m")}</h3>
<p>ค่าสำคัญของเอกสารที่ปรากฏในข้อความที่แต่ละ engine อ่านได้ (ตรวจว่ามีในข้อความ ไม่ตรวจตำแหน่ง):</p>
{rec_table}
<p class="note">แสดงว่าโมเดลไม่ได้พังทั้งตัว ภาษาอังกฤษและตัวเลขอ่านได้ แม้แต่ใบที่สแกนตะแคง</p>

<h2>7 แต่งข้อความและวนซ้ำ {badge("m")}</h2>
<p>เมื่อไม่รู้จักตัวอักษร โมเดลไม่ตอบว่าไม่รู้ ตัวอย่างข้อความที่ได้จริงจากภาพสังเคราะห์ที่เอียง 6° (ส่วนต้น … ส่วนท้าย):</p>
<pre>{e(loop_snip)}</pre>
<p>ผู้พัฒนารู้เรื่องนี้ ในโค้ดมีกลไก "ตอบยาวจนชนเพดาน → ลองใหม่ด้วย repetition_penalty 1.1" แต่เรายังเจอผลยาวเกิน 10,000 ตัวอักษร {n_long} จาก {len(rows)} ใบ (ใบปกติ 800–8,000) และเวลาเพิ่มตามความยาว (ชี้จุดดู):</p>
{chart_len}
{details("ตารางข้อมูล", len_table)}
<p class="note">ผลวนซ้ำไม่ขึ้นเป็นข้อผิดพลาด (<code>code 0</code>) ต้องมีตัวตรวจความยาว/ข้อความซ้ำเอง · ผลยาวที่สุดในชุดนี้คือไฟล์ {max(rows, key=lambda r: r["chars"])["stem"]} ({max(rows, key=lambda r: r["chars"])["chars"]:,} ตัวอักษร)</p>

<h2>8 เทียบกับ Paddle และ Typhoon {badge("m")}</h2>
<p class="legend"><span><i class="sw" style="background:var(--s-paddle)"></i>Paddle</span><span><i class="sw" style="background:var(--s-typhoon)"></i>Typhoon</span><span><i class="sw" style="background:var(--s-oxr)"></i>OXR</span></p>
{chart_cer}
<p class="note">ภาพสังเคราะห์ 21 ภาพ (ใบกำกับสมมติเนื้อหาเดียวกัน) ผ่าน preprocess ของโปรเจกต์ · CER = สัดส่วนตัวอักษรที่ผิด 1.0 = ผิดเท่าความยาวข้อความจริง เกิน 1 ได้เมื่อผลยาวกว่าของจริง · สีเขียวน้ำเงินต่ำกว่า 3:1 บนพื้นสว่าง จึงมีป้ายชื่อแถวและตารางข้อมูลกำกับ</p>
{details("ตารางข้อมูล", cer_table)}
<h3>ทำไม Paddle ต่างออกไป {badge("i")}</h3>
<table><tr><th></th><th>PaddleOCR (รุ่นไทย)</th><th>OXR</th></tr>
<tr><td>หน่วยที่อ่าน</td><td>ทีละบรรทัด (detector หากรอบ แล้ว recognizer อ่านแต่ละกรอบ)</td><td>ทั้งบล็อก (ย่อหน้า/ทั้งตาราง) ในครั้งเดียว</td></tr>
<tr><td>วิธีถอดตัวอักษร</td><td>recognizer เฉพาะการอ่านตัวอักษร ผลผูกกับสิ่งที่เห็นในกรอบ</td><td>โมเดลภาษาสร้างข้อความต่อเนื่อง ต่อเติมได้</td></tr>
<tr><td>ภาษาไทย</td><td>มีโมเดลเฉพาะ (<code>th_PP-OCRv5</code>)</td><td>ไม่อยู่ในข้อมูลฝึกหลัก</td></tr>
<tr><td>เมื่อไม่รู้จัก</td><td>ผิดบางตัว</td><td>แต่งข้อความ/วนซ้ำได้</td></tr></table>
<p class="note">แถวสถาปัตยกรรมเป็นความรู้ทั่วไปของวิธีทั้งสองแบบ ผมไม่ได้ทดสอบข้อนี้แยกในโปรเจกต์ ส่วนผลลัพธ์ในกราฟข้างบนวัดแล้ว</p>

<h2>9 สาเหตุที่เราตัดออก และที่ยังไม่ได้ทดสอบ</h2>
{table(["ข้อสงสัย", "ผล", "หลักฐาน"], [
    ["preprocess ไม่ดีพอ", "ไม่ใช่ตัวหลัก", "ภาพตั้งตรงสะอาดยังได้ CER 0.69 (ดีสุดในชุด ~0.61) ภาพเอียง 4° preprocess ช่วยลด 0.78→0.62 แต่ยังไม่ถึงใช้งานได้ (มีแค่ 2 คู่เทียบ) " + badge("m")],
    ["layout แบ่งบล็อกผิด", "ไม่ใช่ตัวหลัก", "กรอบตรงตำแหน่งจริง (หัวข้อ 2) " + badge("m")],
    ["ค่าที่ลดเพื่อให้พอ VRAM", "ไม่ใช่ตัวหลัก", "free API ที่ใช้ค่าเต็มของผู้พัฒนาก็พังแบบเดียวกัน " + badge("m")],
    ["บรรทัดไทยเล็ก/ไม่คม", "ไม่ใช่", "บรรทัดสะอาดและขยาย 2 เท่าก็ยังเป็นขยะ " + badge("m")],
    ["ตัวเลือกภาษา/prompt", "ไม่มี", "API มีแต่ file, api_key, output_formats, feature_config (keep_header_footer), extra, element_formats " + badge("m")],
    ["ความละเอียด (OXR เรนเดอร์ 150 dpi, เราส่ง ~300 dpi)", "ไม่ทราบ", badge("u")],
    ["ปรับ sampling (temperature, repetition_penalty)", "น่าจะลดการวนซ้ำได้ ไม่น่าแก้ภาษาไทย", badge("u")],
    ["fine-tune ด้วยข้อมูลไทย", "ทางเดียวที่ตรงต้นเหตุ แต่ข้อมูล/GPU/โค้ดเทรนไม่พร้อม", badge("i")],
])}

<h2>10 ข้อจำกัดของหลักฐานในรายงานนี้</h2>
<ul><li>เอกสารจริงมี 18 ใบ (ไทยเป็นส่วนใหญ่) ไม่มีเฉลย เทียบกับค่าที่ผมอ่านจากภาพเองหรือเทียบข้าม engine</li>
<li>ข้อความ OXR ในกราฟภาษา/ความยาวมาจากรอบที่ส่ง PDF ดิบ (ไม่ผ่าน preprocess) ส่วน CER ของภาพสังเคราะห์ผ่าน free API มีความแกว่งระหว่างรอบ</li>
<li>กราฟ token มาจากบรรทัดเดียวและครั้งเดียว (greedy) ค่าบันทึกจากผลที่พิมพ์ตอนวัด</li>
<li>จำนวน token ใน vocabulary นับเมื่อ 2026-11-11 ไฟล์โมเดลถูกลบแล้ว นับซ้ำไม่ได้โดยไม่โหลดใหม่</li></ul>
</main></body></html>"""
    (REPORT / "index.html").write_text(html_doc, encoding="utf-8")
    print("->", REPORT / "index.html")


if __name__ == "__main__":
    main()
