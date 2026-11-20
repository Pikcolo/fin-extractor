"""ให้คะแนนผลสกัดเทียบ ground truth ที่คนกรอก (scripts/eval/groundtruth_app.py) — วัดว่า "ถูก" ไม่ใช่แค่ "ครบ"

    .venv/Scripts/python scripts/eval/score_groundtruth.py data/01_input/finance_doc_scan
    .venv/Scripts/python scripts/eval/score_groundtruth.py data/01_input/finance_doc_scan --runs openthai/luma/rules openthai/regions-luma/rules
    .venv/Scripts/python scripts/eval/score_groundtruth.py data/01_input/finance_doc_scan --by-field     # แยกตามช่อง (นับอย่างเดียว ไม่พิมพ์ค่า)

ไม่ใส่ --runs = ทุกผลที่มีใต้ 05_extract/<dataset>/ นับเฉพาะไฟล์ที่สถานะ done (--include-drafts เพื่อรวมร่าง)
พิมพ์แค่จำนวนนับ ไม่พิมพ์ค่าของเอกสาร ผลเขียนที่ data/etc/groundtruth/<dataset>/score.md ด้วย
accuracy = ถูก / ช่องที่เอกสารมีค่า (ถูก + ผิด + ไม่ได้ค่า) · spurious = ระบบใส่ค่าในช่องที่เอกสารไม่มี
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor import groundtruth as G  # noqa: E402
from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.doctypes import load_profiles  # noqa: E402
from fin_extractor.paths import DataPaths, dataset_of  # noqa: E402


def all_runs(paths: DataPaths, dataset: str) -> list[str]:
    base = paths.final("x", dataset, "x").parents[2]  # 05_extract/<dataset>/
    return sorted(f"{m.parent.parent.parent.name}/{m.parent.parent.name}/{m.parent.name}" for m in base.glob("*/*/*/manifest.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--runs", nargs="*", help="<engine>/<วิธี>/<extractor> (ไม่ใส่ = ทุกชุดที่มี)")
    ap.add_argument("--include-drafts", action="store_true")
    ap.add_argument("--by-field", action="store_true", help="ตารางแยกตามช่อง (จำนวนนับ)")
    args = ap.parse_args()

    cfg = load_config()
    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    profiles = load_profiles(cfg.get("document_types") or {})
    gts = {}
    for p in sorted(G.gt_dir(paths, dataset).glob("*.json")):
        gt = json.loads(p.read_text(encoding="utf-8"))
        if gt.get("status") == "done" or args.include_drafts:
            gts[p.stem] = gt
    if not gts:
        sys.exit(f"ยังไม่มี ground truth ที่ตรวจเสร็จใน {G.gt_dir(paths, dataset)} — กรอกด้วย scripts/eval/groundtruth_app.py")
    runs = args.runs or all_runs(paths, dataset)

    lines = [f"ground truth {len(gts)} ไฟล์ ({'รวมร่าง' if args.include_drafts else 'เฉพาะ done'}) · เทียบแบบตรงทุกตัวหลัง normalize ไม่มีเกณฑ์ความคล้าย", "",
             "| ผลสกัด | ไฟล์ที่มีผล | accuracy | ถูก | ผิด | ไม่ได้ค่า | ใส่เกิน (spurious) |", "|---|---|---|---|---|---|---|"]
    per_field: dict[str, dict[str, dict[str, int]]] = {}
    for run in runs:
        outs = []
        for stem, gt in gts.items():
            res = G.load_result(paths, dataset, run, stem)
            types = G.profile_types(profiles, ((res or {}).get("data") or {}).get("document_profile"))
            one = G.score(gt, res, types)
            outs.append(one)
            for k, o in one.items():
                key = re.sub(r"\[\d+\]", "[]", k)
                per_field.setdefault(key, {}).setdefault(run, dict.fromkeys(G.OUTCOMES, 0))[o] += 1
        have = sum(1 for stem in gts if G.load_result(paths, dataset, run, stem))
        s = G.summarize(outs)
        acc = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "—"
        lines.append(f"| {run} | {have}/{len(gts)} | {acc} | {s['correct']} | {s['wrong']} | {s['missed']} | {s['spurious']} |")
    if args.by_field:
        lines += ["", "แยกตามช่อง: ถูก/ช่องที่เอกสารมีค่า (ผิด, ไม่ได้ค่า)", "",
                  "| ช่อง | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
        for key in sorted(per_field):
            cells = []
            for run in runs:
                n = per_field[key].get(run, dict.fromkeys(G.OUTCOMES, 0))
                present = n["correct"] + n["wrong"] + n["missed"]
                cells.append(f"{n['correct']}/{present} ({n['wrong']}, {n['missed']})" if present else "—")
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    out = G.gt_dir(paths, dataset) / "score.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
