"""เทียบผลสกัดข้ามวิธี (luma/flat/bin) จาก manifest โดย **ปฏิเสธแถวที่เทียบกันไม่แฟร์**

ตารางเทียบรอบก่อนปนกัน (ผลเก่าของบางวิธี, bin ของ 1307 ใช้ ``--drop-empty-rows``) คำสั่งนี้จึงติดธงแถวที่
- ``ผังต่างกัน``: ลายนิ้วมือผัง (id + bbox ที่เก็บ) ใน ``.sections.json`` ของแต่ละวิธีไม่ตรงกัน
  (เทียบเฉพาะวิธีที่ใช้ผังชุดเดียวกัน: ``regions-luma`` ใช้ผัง region ซึ่งต่างจากผัง section โดยตั้งใจ จึงเทียบกับวิธีอื่นได้ตรง ๆ)
- ``ผลเก่ากว่าผัง``: ``.sections.json`` เก่ากว่าไฟล์ผังปัจจุบัน
- ``drop-empty-rows``: ภาพวิธีนั้นถูกลบแถวตารางที่ช่องค่าว่าง (ไม่ใช่ภาพเต็มของส่วน)
- ``ไม่มีผล``: วิธีนั้นไม่มีผลของไฟล์นี้
ยอดรวมนับเฉพาะแถวที่ไม่มีธง (วัดแค่ความ "ครบ" ของฟิลด์ ไม่ใช่ความถูก)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import DataPaths, doc_dir
from .read import REGIONS_VARIANT, images_dir, layout_file


def _manifest_rows(paths: DataPaths, engine: str, dataset: str, variant: str, extractor: str) -> dict[str, dict]:
    p = paths.final(engine, dataset, extractor, variant) / "manifest.json"
    if not p.exists():
        return {}
    return {Path(f["file"]).stem: f for f in json.loads(p.read_text(encoding="utf-8")).get("files", [])}


def _raw_info(paths: DataPaths, engine: str, dataset: str, variant: str, stem: str) -> dict | None:
    raw_dir = paths.raw_data(engine, dataset, variant)
    sections = doc_dir(raw_dir, stem, nest=True) / f"{stem}_p1.sections.json"
    if not sections.exists():
        sections = raw_dir / f"{stem}_p1.sections.json"
    return {"layout": json.loads(sections.read_text(encoding="utf-8")).get("layout"), "mtime": sections.stat().st_mtime} if sections.exists() else None


def _drops_rows(paths: DataPaths, dataset: str, variant: str, stem: str) -> bool:
    index = images_dir(paths, dataset, stem, variant) / "index.json"
    return index.exists() and any("rows" in s for s in json.loads(index.read_text(encoding="utf-8"))["sections"])


def compare_variants(paths: DataPaths, dataset: str, engine: str, variants: tuple[str, ...], extractor: str = "rules") -> dict:
    """{"rows": [{file, flags, per_variant: {v: {status, missing, checks_failed}}}], "totals": {v: {...}}, "comparable": n}."""
    manifests = {v: _manifest_rows(paths, engine, dataset, v, extractor) for v in variants}
    stems = sorted({s for m in manifests.values() for s in m})
    rows = []
    for stem in stems:
        flags, per = [], {}
        raws = {v: _raw_info(paths, engine, dataset, v, stem) for v in variants}
        for v in variants:
            lf = layout_file(paths, dataset, stem, v)
            m = manifests[v].get(stem)
            if m is None or raws[v] is None:
                flags.append(f"ไม่มีผล:{v}")
                continue
            per[v] = {"status": m["status"], "missing": len(m["missing_fields"]), "checks_failed": m["checks_failed"]}
            if v != REGIONS_VARIANT and lf.exists() and raws[v]["mtime"] < lf.stat().st_mtime:  # ผัง region เขียนใหม่ทุกรอบ (ผลเหมือนเดิม) ใช้ลายนิ้วมือแทนเวลา
                flags.append(f"ผลเก่ากว่าผัง:{v}")
            if _drops_rows(paths, dataset, v, stem):
                flags.append(f"drop-empty-rows:{v}")
        prints = {json.dumps(r["layout"]) for v, r in raws.items() if r and v != REGIONS_VARIANT}
        if len(prints) > 1:
            flags.append("ผังต่างกัน")
        rows.append({"file": stem, "flags": flags, "per_variant": per})
    ok = [r for r in rows if not r["flags"]]
    totals = {v: {"missing": sum(r["per_variant"][v]["missing"] for r in ok), "checks_failed": sum(r["per_variant"][v]["checks_failed"] for r in ok),
                  "incomplete": sum(r["per_variant"][v]["status"] == "incomplete" for r in ok)} for v in variants}
    return {"rows": rows, "totals": totals, "comparable": len(ok), "variants": list(variants)}


def to_markdown(result: dict) -> str:
    vs = result["variants"]
    out = ["| ไฟล์ | " + " | ".join(f"{v} (ขาด/เช็คไม่ผ่าน/สถานะ)" for v in vs) + " | ธง |", "|---|" + "---|" * len(vs) + "---|"]
    for r in result["rows"]:
        cells = [f"{p['missing']}/{p['checks_failed']}/{p['status'][:6]}" if (p := r["per_variant"].get(v)) else "-" for v in vs]
        out.append(f"| {r['file'][-4:]} | " + " | ".join(cells) + f" | {', '.join(r['flags']) or ''} |")
    out.append("")
    out.append(f"รวมเฉพาะแถวที่เทียบกันได้ (ไม่มีธง) {result['comparable']}/{len(result['rows'])} ไฟล์:")
    for v in vs:
        t = result["totals"][v]
        out.append(f"- {v}: ฟิลด์ที่ขาด {t['missing']} · เช็คไม่ผ่าน {t['checks_failed']} · incomplete {t['incomplete']}")
    out.append("วัดความครบ ไม่ใช่ความถูก (ไม่มีเฉลย)")
    return "\n".join(out)
