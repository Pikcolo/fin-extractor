# fin_extractor

OCR + Information Extraction สำหรับเอกสารการเงินไทย (ใบกำกับภาษี / ใบเสร็จ / invoice) ที่ **ไม่มี pattern ตายตัว**
ผลลัพธ์เป็น JSON มาตรฐานตาม [`schemas/invoice.schema.json`](schemas/invoice.schema.json) (schema 4.0: ยอดเงินเป็น `Decimal` ระดับ + เหตุผลรายฟิลด์ `review` เรียงตามที่ควรตรวจก่อน ฟิลด์เฉพาะชนิดเอกสาร `document_fields` ตามโปรไฟล์ และ `remark` สำหรับข้อมูลที่ไม่มีช่อง)
อธิบายขั้นตอนการทำงานแบบละเอียดอยู่ใน [`docs/`](docs/README.md)

```
ไฟล์ (jpg/png/pdf)
  └─ io/loader          โหลด + render PDF เป็นภาพทีละหน้า
      └─ preprocessing  upscale → orientation → deskew   (คงสี; denoise/CLAHE/binarize ปิดเป็นค่าตั้งต้น)
          └─ sections/  flow หลัก: แบ่งเป็น region (sections/regions/) → อ่านทีละส่วน → .md + chunks.json → rules   (ดู "flow หลัก")
              ├─ readers/   ตัวอ่านทุก engine ทางเดียวกัน: openthai (ค่าตั้งต้น) · openthai21 · openthai-text · typhoon · paddle → ReadResult
              └─ extraction/  rules (regex+keyword)  และ/หรือ  LLM → InvoiceData
                  └─ validation/  checksum เลขภาษี, Σitems, ส่วนลด, ก่อนVAT+VAT=รวม, VAT 7% → status + review
```

## เริ่มใช้งาน

ต้องมี Python 3.10 ขึ้นไป รายละเอียดทุกขั้น (GPU/CPU, Ollama, ปัญหาที่พบบ่อย) อยู่ใน [docs/00-setup-and-usage.md](docs/00-setup-and-usage.md)

```powershell
git clone https://github.com/Pikcolo/fin-extractor.git
cd fin-extractor
python -m venv .venv
.venv\Scripts\activate

# paddlepaddle ต้องลงก่อน เลือกอย่างใดอย่างหนึ่ง
pip install paddlepaddle-gpu==3.4.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://pypi.org/simple   # มี GPU
pip install paddlepaddle                                                                                                                  # ไม่มี GPU

pip install -e ".[all]"
pytest                                    # ต้องผ่านทั้งหมด (ไม่ต้องใช้ GPU / Ollama / ข้อมูล)
```

`.[all]` = paddle + vlm + pytest ยังไม่รวมตัวอ่านค่าตั้งต้น OpenThai (torch + `.[openthai]` + โมเดล) และ `onnxruntime` ของชั้นตรวจความครบ ลงตามตาราง "ส่วนเสริม" ใน [docs/00-setup-and-usage.md](docs/00-setup-and-usage.md)

### รันครั้งแรก

เอกสารจริงไม่อยู่ใน git (ทุกอย่างใต้ `data/` ถูก `.gitignore`) วางไฟล์ของคุณที่ `data/01_input/` หรือสร้างชุดสังเคราะห์:

```powershell
python scripts/data/make_synthetic_samples.py                  # ใบกำกับภาษีสมมติ 21 ใบ → data/01_input/synthetic/
python -X utf8 -m fin_extractor sections data/01_input/synthetic --engine paddle   # Paddle ทีละท่อน + rules → data/05_extract/synthetic/paddle/regions-luma/rules/
```

### คำสั่งหลัก

```powershell
python -m fin_extractor preprocess data/01_input --steps           # ดูผล preprocess ทีละขั้น
python -m fin_extractor sections data/01_input/finance_doc_scan    # flow หลัก (ดูหัวข้อถัดไป) --engine openthai|openthai21|openthai-text|typhoon|paddle
python -m fin_extractor extract data/01_input/finance_doc_scan --engine typhoon --only 1306   # สกัดซ้ำจากข้อความรายส่วนที่เก็บไว้ (ไม่อ่านภาพ ไม่ใช้ GPU)
python -m fin_extractor compare data/01_input/finance_doc_scan     # เทียบผลสกัด luma/flat/bin ติดธงแถวที่เทียบไม่แฟร์
python -m fin_extractor schema                                # export JSON Schema (หลังแก้ schema.py)
python -X utf8 scripts/demo/ocr_demo_app.py                  # หน้าเว็บลอง OCR 1 ไฟล์ ดูผลทุกขั้นจนได้ JSON → http://127.0.0.1:8766
pytest
```

**ห้ามอ่านทั้งหน้า:** ทุก engine อ่านทีละ section ผ่านคำสั่ง `sections` เท่านั้น (ไม่มีคำสั่ง `ocr` แล้ว) และ `extract` สกัดจากข้อความรายส่วนที่เก็บไว้เท่านั้น
`--engine typhoon` ต้อง `ollama pull scb10x/typhoon-ocr1.5-3b` · LLM สกัด (`extract --llm on`) ต้อง `ollama pull qwen2.5:3b` (หรือใช้ Typhoon API ตั้งค่าใน `.env`)

### flow หลัก: แบ่งเป็น region แล้วอ่านด้วย OpenThai (สแกนจริง)

อินพุต → preprocess → **แบ่งเป็น region (กรอบชิดข้อความ ตารางตามเส้น ระบาย QR/บาร์โค้ด)** → OCR ทีละท่อน → ชั้นตรวจความครบ (OpenThai 2.1 Text บน CPU ข้ามเองถ้าไม่มี onnxruntime/โมเดล ปิดด้วย `--no-coverage-check`) → **ข้อความ + ที่มาของแต่ละท่อน** → สกัดเป็น schema (ข้อมูลสำคัญที่ไม่มีช่องไปอยู่ใน `remark`)
ไม่ใช้ OCR คัดส่วน ท่อนที่ไม่มีข้อความ (มีแต่เส้นตาราง) ไม่ส่ง OCR ใช้ GPU เฉพาะขั้น 4 ([docs/03-regions.md](docs/03-regions.md))

```powershell
python -X utf8 -m fin_extractor sections data/01_input/finance_doc_scan            # รันครบทุกขั้น ขั้นที่ผลยังใช้ได้ถูกข้าม
python -X utf8 -m fin_extractor sections data/01_input/finance_doc_scan --from ocr --only 1306   # เริ่มกลางทาง/เฉพาะบางไฟล์ (GPU: รันทีละชุด)
python -X utf8 scripts/demo/ocr_demo_app.py                                          # หน้าเว็บลอง 1 ไฟล์ ดูทุกขั้น + ตรวจ/แก้รายฟิลด์
```

| ขั้น | ผลที่ดูได้ (ใต้ `data/`) | ใช้อะไรดู debug |
|---|---|---|
| 2 preprocess | `02_preprocess/<ชุด>/<profile>/final/` ภาพทั้งหน้าตรงแล้ว | ภาพ + `manifest.json` (`orientation_error`) |
| 3 layout | `03_sections/<ชุด>/<profile>/regions/<ไฟล์>/` `01_normalize` … `06_pieces.png` · `crops/` · `sections.json` · `manifest.json` | `06_pieces.png` = กรอบที่ส่ง OCR จริง · `manifest.json → stages.mask.codes` (ข้อความใน QR) · `skipped_regions` (ท่อนที่ไม่ส่ง) |
| 4 ocr | `04_ocr/<ชุด>/openthai/regions-luma/<ไฟล์>/` `.md` `.sections.json` `.chunks.json` | `chunks.json` = ท่อนข้อความ ↔ ภาพ ↔ บรรทัดใน `.md` |
| 5 extract | `05_extract/<ชุด>/openthai/regions-luma/rules/<ไฟล์>_p1.json` | `review` (ระดับ + เหตุผลรายฟิลด์), `remark[].source_chunk` ชี้กลับไปที่ท่อน |

- **ค่าตั้งต้นคือผัง regions ตั้งแต่ 2026-11-24** (ผู้ใช้เลือกหลังดูภาพที่ส่ง OCR) ผลวัดเดิม (OpenThai 18 ใบ rules ล้วน ไม่มีเฉลย) ฟิลด์ที่ขาดรวม regions 20 เทียบ sections 14 → ควรวัดซ้ำกับเฉลย ([docs/03-regions.md](docs/03-regions.md))
- **ผังแบบเดิม** (แถบเต็มหน้า ภาพ `luma|flat|bin|color`) ยังใช้ได้ด้วย `--layout sections` ผลที่ `<engine>/luma/` ฯลฯ ไม่ทับกัน ([docs/03-sections.md](docs/03-sections.md)) รันทีละขั้นด้วย `--to layout|images|ocr` (สคริปต์เดิม `scripts/sections/` ลบแล้ว)
- เทียบผลสกัดข้ามวิธี: `python -m fin_extractor compare <input> --variants luma regions-luma [--out ไฟล์.md]` ติดธงแถวที่เทียบไม่แฟร์และไม่นับในยอดรวม
- ข้อมูลภาษีจริงไม่ถูก commit และผลวัดทั้งหมดไม่ใช่คะแนนความถูกต้อง (มีเฉลยแล้วใช้ `scripts/eval/score_groundtruth.py`)
- ข้อควรระวัง: ตรวจ `orientation_error` ใน `manifest.json` ของ preprocess ก่อนเชื่อผล (ตัวตรวจทิศอาจล้มเงียบ)

### ผลลัพธ์อยู่ที่ไหน

ข้อมูลทั้งหมดอยู่ใต้ `data/` (ไม่ commit) ชื่อโฟลเดอร์เรียงตามขั้น ทุกขั้นใช้ `<ชุด>` และ `<ไฟล์>` เดียวกัน ตามไฟล์เดียวข้ามขั้นได้ด้วยชื่อเดียว:

```
data/
├── 01_input/<ชุด>/                              ไฟล์อินพุต
├── 02_preprocess/<ชุด>/<profile>/{final,steps}  ภาพสุดท้ายที่ทุกขั้นถัดไปใช้ร่วมกัน · ภาพทีละขั้น
├── 03_sections/<ชุด>/<profile>/                 แบ่งเป็นส่วน: regions/<ไฟล์>/ (ค่าตั้งต้น: ภาพแต่ละขั้น + crops/) · layout/ + images/<luma|flat|bin>/ (ผังแบบเดิม)
├── 04_ocr/<ชุด>/<engine>/<วิธี>/<ไฟล์>/          ข้อความจาก OCR (วิธี = regions-luma ค่าตั้งต้น · luma|flat|bin|color ผังแบบเดิม)
├── 05_extract/<ชุด>/<engine>/<วิธี>/<extractor>/ ผลตาม schema (JSON) extractor = rules | llm | hybrid
└── etc/compare/ · etc/groundtruth/ · etc/demo/  เทียบรายไฟล์ · ค่าที่คนกรอก · ผลของหน้าเว็บเดโม
```

ถ้ารันหนึ่งครั้งมีไฟล์อินพุตมากกว่า 2 ไฟล์ ผลของแต่ละไฟล์จะอยู่ในโฟลเดอร์ย่อยชื่อไฟล์ (เช่น `05_extract/<ชุด>/openthai/regions-luma/rules/<ไฟล์>/<ไฟล์>_p1.json`) ไม่เกิน 2 ไฟล์ก็วางตรง ๆ (ตั้งที่ `NEST_OVER` ใน `src/fin_extractor/paths.py`)

เช่น `sections data/01_input/finance_doc_scan --engine typhoon` → ข้อความรายส่วนที่ `data/04_ocr/finance_doc_scan/typhoon/regions-luma/` และ JSON ที่ `data/05_extract/finance_doc_scan/typhoon/regions-luma/rules/`
`extract ... --engine <ตัวเดิม>` สกัดซ้ำจากข้อความนั้นโดยไม่อ่านภาพ/ไม่ใช้ GPU ดู [docs/00-setup-and-usage.md](docs/00-setup-and-usage.md#โครงเก็บผลลัพธ์)

## โครงสร้างโฟลเดอร์

```
fin_extractor/
├── configs/default.yaml        ค่าตั้งต้นทั้งหมด (preprocess, engine, endpoint VLM/LLM, coverage_check, document_types)
├── configs/regions/            ค่าของผัง regions: pipeline.yaml (ทุกเอกสารใช้ค่าชุดเดียวกัน ไม่มีตั้งค่ารายไฟล์)
├── assets/fonts/               ฟอนต์ Sarabun (SIL OFL) ใช้วาดชุดสังเคราะห์และหน้าเว็บเดโม
├── data/                       01_input … 05_extract + etc/ (ไม่ commit ยกเว้น .gitkeep)
├── docs/                       เอกสารเรียงตาม flow: 00 ติดตั้ง · 01–05 ทีละขั้น · 06 หลักการ/roadmap · draft-* แผนที่ยังไม่ทำ
├── models/                     โมเดลที่ library โหลดเอง (paddlex/, OpenThai2.0-OCR-Markdown/) ไม่ commit
├── schemas/                    JSON Schema (Data Contract) — generate จาก schema.py
├── scripts/                    สคริปต์เสริมนอก pipeline แยกตาม part (เลขเดียวกับขั้นใน docs/)
│   ├── data/                   make_synthetic_samples (ชุดสังเคราะห์ที่รู้เฉลย)
│   ├── demo/                   ocr_demo_app + ocr_demo_worker + ocr_demo.html (หน้าเว็บลองทีละขั้น) · review.py (ตรวจ/แก้รายฟิลด์)
│   └── eval/                   compare_file (เทียบผลสกัด) · groundtruth_app · score_groundtruth · summarize_extract
├── src/fin_extractor/
│   ├── schema.py               ★ Data Contract (pydantic)
│   ├── config.py
│   ├── pipeline.py             FinanceOCRPipeline: ข้อความดิบ → สกัด (rules/LLM/hybrid) → validation → สถานะ (ไม่อ่านภาพ ลำดับขั้นอยู่ที่ cli.py + sections/flow.py)
│   ├── cli.py                  python -m fin_extractor ... (preprocess · sections · extract · regions · compare · schema)
│   ├── paths.py                ที่เดียวที่กำหนดว่าแต่ละขั้นเขียนข้อมูลที่ไหน (data/01_input … 05_extract)
│   ├── stages.py               preprocess ที่เก็บไว้ใช้ร่วมกัน + ข้อความดิบ
│   ├── doctypes.py             โปรไฟล์ชนิดเอกสาร (ฟิลด์/required/checks ต่อชนิด)
│   ├── runs.py                 manifest ของการรัน
│   ├── groundtruth.py          ground truth ที่คนกรอก + ให้คะแนน (data/etc/groundtruth/)
│   ├── io/loader.py
│   ├── preprocessing/          หนึ่งไฟล์ต่อหนึ่งขั้น (แบบ learning/preprocess_finance) ลำดับตาม pipeline.py
│   │   ├── clean.py            ลบเส้นสกปรกจากสแกน (ก่อนหมุน) · denoise
│   │   ├── enhance.py          gray, upscale, CLAHE
│   │   ├── orient.py           ทิศทั้งหน้า 0/90/180/270 (Paddle doc-orientation)
│   │   ├── deskew.py           แก้เอียง (projection profile)
│   │   ├── crop.py             ตัดขอบ
│   │   ├── binarize.py         Otsu / adaptive / Sauvola
│   │   └── pipeline.py         รวมขั้นตอนตาม config: clean(streaks) → upscale → orient → deskew → crop → denoise → CLAHE → binarize
│   ├── sections/               ★ flow หลัก: layout.py (ผัง) · images.py (luma/flat/bin) · crops.py · build.py (ขั้น 3a–3b) · read.py (ขั้น 4 OCR รายส่วน + chunks.json) · coverage.py (ชั้นตรวจความครบ) · graphics.py (ระบาย QR/บาร์โค้ด, trim) · flow.py (กติกาข้ามขั้น) · compare.py (คำสั่ง compare) · regions/ (ผัง regions ค่าตั้งต้น: region/ตาราง/ช่อง/key-value)
│   ├── readers/                ตัวอ่านขั้น 4 ทุก engine: __init__.py (make_reader, ENGINES) · base.py (ReadResult) · openthai.py · openthai_text.py · typhoon.py · paddle.py (+ lines.py)
│   ├── llm_client.py           เรียกโมเดลแบบแชต (Ollama/OpenAI-compatible) ใช้ทั้ง Typhoon และ LLM ตัวสกัด
│   ├── extraction/             patterns.py (regex), rule_extractor.py, profile_extractor.py, core_labels.py, item_table.py, llm_extractor.py, prompts.py, hybrid.py
│   └── validation/             tax_id.py, checks.py, scoring.py, evidence.py
└── tests/
```

## หมายเหตุการออกแบบ

- **Preprocess ชุดเดียว**: แก้เรขาคณิต (ตะแคง/เอียง) แต่คงสี ไม่ binarize ไม่ลด noise — ชุดสังเคราะห์วัดว่าภาพสีอ่านได้ครบที่สุดและเร็วที่สุด ส่วนสแกนจริง 18 ใบ (ไม่มีเฉลย) ภาพเทาไม่ต่างจากสี แต่ binarize (Otsu/Sauvola/adaptive) อ่านเลขภาษีได้น้อยลงและ NL-means ช้าลง 4 เท่าโดยไม่ช่วย (ดู [docs/02-preprocessing.md](docs/02-preprocessing.md#ผลวัดบนสแกนจริง))
- **ต่อคำภาษาไทย**: `OCRLine.text` เว้นวรรคตามระยะห่างจริงในภาพ (ไม่งั้นได้ `บ ร ิ ษั ท`) และแก้ `ํา` → `ำ`
- **Ollama ใช้ native API** เพราะ endpoint แบบ OpenAI ตั้ง `num_ctx` ไม่ได้ (default 4096 ทำให้ VLM ตอบไม่จบ)
- **LLM + rules + cross-check**: `extract --llm on|auto` รันทั้งสองแบบ เลือกผลที่ตัวเลขสอดคล้องกันมากกว่า แล้วเติม field ที่ขาดจากอีกฝั่ง · `--llm hybrid` = rules นำ LLM เติมเฉพาะช่องที่ยังไม่เสร็จ
- **status**: `auto_approved` (field สำคัญครบ, ไม่มี check ไหน fail และผ่านอย่างน้อย 1 ข้อ) / `needs_review` / `failed` (หา total ไม่เจอ)
- **review** (schema 4.0): ระดับรายฟิลด์ (`conflict` / `missing` / `unverified` / `verified` / `not_applicable`) + เหตุผลเป็นคำพูด เรียงตามที่ควรตรวจก่อน ค่าอยู่ใน `data` ที่เดียว ดู [docs/05-validation-output.md](docs/05-validation-output.md)
