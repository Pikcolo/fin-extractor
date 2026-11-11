"""รัน pipeline เต็มโดยใช้ OXR เป็นตัวอ่านข้อความ (เทียบกับ paddle / typhoon ที่ใช้ภาพและขั้นสกัดเดียวกัน)

ภาพมาจาก preprocess/<dataset>/<profile>/final/ (ชุดเดียวกับที่ engine อื่นใช้) ส่ง JPEG ไปที่ OXR **ในเครื่อง**
(container `oxr` ที่ 127.0.0.1:8000) ผลเป็น Markdown + ตาราง HTML (เหมือน Typhoon) จึงเข้าทางเดียวกัน: markdown_to_lines → rules → validation
ไม่มี LLM ตัวสกัด และไม่มี finish_reason จาก OXR ผลวนซ้ำ/ถูกตัดจึงไม่ถูกตั้งธง ``vlm_complete``

ห้ามชี้ ``--api`` ไปที่ free API กับสแกนจริง (ข้อมูลออกนอกเครื่อง) สคริปต์จึงปฏิเสธโฮสต์ที่ไม่ใช่ localhost

    .venv/Scripts/python scripts/extract_oxr.py data/raw/finance_doc_scan [--redo]
    # ผล: result_ocr/oxr/<dataset>/raw_data/<ไฟล์>_p1.md (+ .oxr.json = ผลเต็มของ OXR: บล็อก/bbox/ลำดับ)
    #     result_ocr/oxr/<dataset>/final/rules/<ไฟล์>_p1.json
    # รันต่อจากที่ค้างได้ (ข้ามหน้าที่มีผลแล้ว) ใส่ --redo เพื่อทำใหม่ทั้งหมด
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fin_extractor.config import load_config  # noqa: E402
from fin_extractor.io import iter_input_files  # noqa: E402
from fin_extractor.paths import DEFAULT_PROFILE, DataPaths, dataset_of  # noqa: E402
from fin_extractor.pipeline import FinanceOCRPipeline  # noqa: E402
from fin_extractor.runs import Run  # noqa: E402
from fin_extractor.stages import prepared_pages, save_raw  # noqa: E402


@dataclass
class OXRResult:
    engine: str
    markdown: str
    truncated: bool = False


class OXRReader:
    name = "oxr"

    def __init__(self, api: str, timeout: int = 900):
        self.api, self.timeout = api, timeout
        self.last_payload: dict | None = None  # ผลเต็มของคำขอล่าสุด (เก็บลง raw_data)

    def recognize(self, img) -> OXRResult:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        r = httpx.post(
            f"{self.api}/v1/parse/sync",
            files=[("file", ("page.jpg", buf.tobytes(), "image/jpeg")), ("output_formats", (None, "markdown")),
                   ("output_formats", (None, "json")), ("api_key", (None, "dummy"))],
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0 or not body.get("data"):
            raise RuntimeError(f"OXR error: {body.get('code')} {body.get('message')}")
        self.last_payload = body
        return OXRResult("oxr", body["data"]["result"]["outputs"]["markdown"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--redo", action="store_true", help="ทำใหม่ทุกหน้า (ค่าตั้งต้น: ข้ามหน้าที่มีผลสุดท้ายแล้ว)")
    args = ap.parse_args()
    if urlparse(args.api).hostname not in ("127.0.0.1", "localhost"):
        sys.exit("ปฏิเสธ: --api ต้องเป็น localhost (ไม่ส่งสแกนจริงออกนอกเครื่อง)")

    cfg = load_config()
    paths, dataset = DataPaths(cfg), dataset_of(args.input)
    raw_dir, final_dir = paths.raw_data("oxr", dataset), paths.final("oxr", dataset, "rules")
    reader = OXRReader(args.api)
    pipe = FinanceOCRPipeline(cfg, mode="vlm", use_llm=False, with_reader=False)  # ขั้นสกัด/ตรวจเหมือนทางของ Typhoon แบบ rules
    pipe.reader = reader
    settings = {"mode": "oxr", "engine": pipe.engine_name, "api": args.api, "profile": args.profile}
    with Run("extract", args.input, cfg, final_dir, settings) as run:
        for f in iter_input_files(args.input):
            for page_no, img, _ in prepared_pages(f, cfg, paths.preprocess(dataset, args.profile)):
                stem = f"{f.stem}_p{page_no}"
                out = final_dir / f"{stem}.json"
                if out.exists() and not args.redo:
                    old = json.loads(out.read_text(encoding="utf-8"))  # manifest ของรอบต่อยังนับหน้าที่ทำไว้แล้วด้วย
                    run.add(file=f.name, page=page_no, status=old["status"], missing_fields=old["missing_fields"],
                            output=out.relative_to(run.dir).as_posix(), skipped=True)
                    print(f"ข้าม {stem} (มีผลแล้ว)", flush=True)
                    continue
                t0 = time.perf_counter()
                r = pipe.process_page(str(f), page_no, img)
                seconds = round(time.perf_counter() - t0, 2)
                save_raw(raw_dir, stem, r.raw_text or "", "md", {"engine": "oxr", "profile": args.profile, "seconds": seconds,
                                                                   "ocr_confidence": None, "truncated": False})
                (raw_dir / f"{stem}.oxr.json").write_text(json.dumps(reader.last_payload, ensure_ascii=False), encoding="utf-8")
                r.raw_text = None
                out.write_text(r.model_dump_json(indent=2), encoding="utf-8")
                passed = sum(c.passed is True for c in r.checks)
                failed = sum(c.passed is False for c in r.checks)
                run.add(file=f.name, page=page_no, status=r.status, missing_fields=r.missing_fields,
                        checks_passed=passed, checks_failed=failed, seconds=seconds, output=out.relative_to(run.dir).as_posix())
                print(f"{f.name} p{page_no}: {r.status} checks={passed}✓ {failed}✗ "
                      f"missing={','.join(r.missing_fields) or '-'} ({seconds:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
