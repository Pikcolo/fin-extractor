# ติดตั้งและวิธีใช้

## ติดตั้ง (เครื่องใหม่ตั้งแต่ clone)

ต้องมี Python 3.10 ขึ้นไป (ทดสอบกับ 3.13) PaddleOCR เป็น OCR engine ตัวเดียว ต้องลง `paddlepaddle` (ตัว framework) **ก่อน** แล้วค่อยลงโปรเจกต์

```powershell
git clone https://github.com/Pikcolo/fin-extractor.git
cd fin-extractor
python -m venv .venv
.venv\Scripts\activate                 # Windows (Linux/macOS: source .venv/bin/activate)

# 1) paddlepaddle เลือกอย่างใดอย่างหนึ่ง
pip install paddlepaddle-gpu==3.4.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://pypi.org/simple   # มี GPU
pip install paddlepaddle                                                                                                                  # ไม่มี GPU

# 2) โปรเจกต์ (paddleocr + vlm + pytest)
pip install -e ".[all]"

# 3) ตรวจ
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"   # ใช้ GPU ต้องได้ True
pytest                                                                     # ต้องผ่านทั้งหมด ไม่ต้องใช้ GPU/Ollama/ข้อมูลในโฟลเดอร์ data
```

- `paddlepaddle-gpu` ไม่อยู่บน PyPI ต้องดึงจาก index ของ Paddle และต้องมี `--extra-index-url`
  ไว้ดึงแพ็กเกจ `nvidia-*` (cuBLAS, cuDNN) จาก PyPI ไม่งั้นล้มที่ `nvidia-cudnn-cu13`
  เลือก `cuXXX` ให้ตรงกับ driver (ดู `nvidia-smi` → `CUDA Version`)
- โปรเจกต์ใช้ `opencv-contrib-python` ไม่ใช่ `headless` เพราะ `paddlex` (ที่ PaddleOCR ใช้)
  บังคับรุ่น contrib — สองแพ็กเกจเขียนลงโฟลเดอร์ `cv2` เดียวกัน ห้ามลงคู่กัน
- รันครั้งแรกจะโหลดโมเดลเก็บไว้ที่ `~/.paddlex/official_models` (ช้ากว่าปกติ) ตั้ง
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` เพื่อข้ามการเช็กเซิร์ฟเวอร์โมเดลทุกครั้งที่เริ่ม
  (PowerShell: `$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"`)
- Windows: ใส่ `-X utf8` (`python -X utf8 -m fin_extractor ...`) ถ้าคอนโซลแสดงภาษาไทยเพี้ยนหรือ error `charmap`

### ส่วนเสริม (ตามที่จะใช้)

| สิ่งที่ต้องมี | ใช้กับ | ติดตั้ง |
|---|---|---|
| Ollama + Typhoon OCR | `--mode vlm` | `ollama pull scb10x/typhoon-ocr1.5-3b` |
| Ollama + LLM | การสกัดด้วย LLM (`--mode vlm`, `--llm on`) | `ollama pull qwen2.5:3b` (เปลี่ยนรุ่นที่ `llm_extractor.model`) |
| ฟอนต์ Tahoma (Windows) | `scripts/make_synthetic_samples.py` สร้างชุดสังเคราะห์ | มากับ Windows (`C:/Windows/Fonts/tahoma.ttf`) บน OS อื่นต้องแก้ path ในสคริปต์ |

**ใช้ Typhoon API บน cloud แทน Ollama:** คัดลอก `.env.example` เป็น `.env` ใส่ `TYPHOON_API_KEY` แล้วตั้ง `vlm.provider: openai` และ
`vlm.base_url: https://api.opentyphoon.ai/v1` (ส่งเอกสารการเงินออกนอกเครื่อง ตรวจนโยบายข้อมูลก่อน) โหมด `ocr` ไม่ต้องใช้ทั้ง Ollama และ key

## ข้อมูลตัวอย่าง

โฟลเดอร์ `data/raw` และ `data/output` อยู่ใน git เพียงแค่ `.gitkeep` เนื้อหาข้างในถูก `.gitignore` ไว้ (เอกสารจริงอาจมีข้อมูลภาษีและข้อมูลส่วนบุคคล)
หลัง clone จึงไม่มีเอกสารให้รัน มีสองทาง

1. **สแกนจริง:** รับชุดสแกนจากเจ้าของโปรเจกต์นอก git แล้ววางที่ `data/raw/finance_doc_scan/` (ไฟล์ `.pdf`, `.jpg`, `.png`, `.tif`, `.bmp`, `.webp`)
2. **ชุดสังเคราะห์** (ไม่ต้องรอข้อมูลจริง): `python scripts/make_synthetic_samples.py` ได้ใบกำกับภาษีสมมติ 21 ใบพร้อมเฉลยที่ `data/raw/synthetic/`

## รันครั้งแรก

```powershell
python scripts/make_synthetic_samples.py                      # ถ้ายังไม่มีเอกสาร
python -m fin_extractor extract data/raw/synthetic            # โหมด ocr (Paddle + rules) ไม่ต้องใช้ Ollama
```

ได้ผลที่ `data/output/extract/synthetic__ocr-rules/` (ดูโครงข้างล่าง) และบนหน้าจอจะแสดง status กับผล check ของแต่ละใบ
ตัวโหลดไฟล์กรองเฉพาะนามสกุลภาพ/PDF จึงไม่หยิบ `truth.json` ในโฟลเดอร์เดียวกัน

โหมด `vlm` (Typhoon + LLM) ต้องเปิด Ollama ก่อน และช้ากว่ามาก:

```powershell
python -m fin_extractor extract data/raw/finance_doc_scan --mode vlm
```

## โครงเก็บผลลัพธ์ (ทุกคำสั่งเหมือนกัน)

```
data/output/<คำสั่ง>/<ชื่ออินพุต>__<แท็ก>/
├── manifest.json     คำสั่งที่รัน (argv), เวลา, schema_version, ค่า preprocessing, โมเดลที่ใช้, รายการไฟล์พร้อมสถานะ/เวลา
└── results/ | text/ | images/ | debug/ | steps/    ผลของคำสั่งนั้น
```

| คำสั่ง | โฟลเดอร์ผลตั้งต้น | ข้างใน |
|---|---|---|
| `extract` | `data/output/extract/<อินพุต>__<mode>-<llm\|rules>/` เช่น `finance_doc_scan__ocr-rules` | `results/<ไฟล์>_p<หน้า>.json` (ตาม `schemas/invoice.schema.json`), `debug/` เมื่อใส่ `--debug` |
| `ocr` | `data/output/ocr/<อินพุต>__<engine>/` เช่น `BITKUB-001__paddle` | `text/<ไฟล์>_p<หน้า>.txt` (vlm เป็น `.md`) |
| `preprocess` | `data/output/preprocess/<อินพุต>/` | `images/<ไฟล์>_p<หน้า>.png`, `steps/` เมื่อใส่ `--steps` |
| script วัดผล | `data/output/benchmarks/` | ไฟล์ผลของแต่ละ script (ดูตารางข้างล่าง) |

- **อินพุต** คือชื่อโฟลเดอร์ หรือชื่อไฟล์ (ไม่รวมนามสกุล) **แท็ก** บอกวิธีที่ใช้: `ocr-rules` (Paddle + rules), `ocr-llm`, `vlm-llm`, `vlm-rules`
- รันซ้ำด้วยอินพุตและวิธีเดิมจะเขียนทับโฟลเดอร์เดิม ต้องการเก็บหลายรอบให้ใช้ `--out <โฟลเดอร์>` (โครงข้างในยังเหมือนเดิม)
- `manifest.json` เขียนตอนจบ ถ้ารันล้มกลางทางจะมี `"status": "failed"` พร้อมไฟล์ที่ทำเสร็จแล้ว

## คำสั่ง

| คำสั่ง | ขั้นที่ทำ | ผลลัพธ์ |
|---|---|---|
| `python -m fin_extractor preprocess <input> [--steps]` | 1 → 2 | `data/output/preprocess/` |
| `python -m fin_extractor ocr <input> --engine paddle\|vlm` | 1 → 3 | `data/output/ocr/` |
| `python -m fin_extractor extract <input> --mode ocr\|vlm` | 1 → 7 | `data/output/extract/` |
| `python -m fin_extractor schema` | — | `schemas/invoice.schema.json` (generate จาก `schema.py` ห้ามแก้ไฟล์ตรง ๆ) |
| `pytest` | รันเทสต์ | |

ตัวเลือกของ `extract`:

| ตัวเลือก | ความหมาย |
|---|---|
| `--mode ocr\|vlm` | `ocr` = Paddle + rules (ตั้งต้น), `vlm` = Typhoon OCR + LLM + rules |
| `--engine` | engine สำหรับ `--mode ocr` (ตั้งต้น `paddle`) |
| `--llm auto\|on\|off` | `auto` = ใช้ LLM เฉพาะโหมด vlm |
| `--debug` | บันทึกภาพทุกขั้นของ preprocess ลง `debug/` |
| `--keep-raw` | เก็บข้อความดิบใน JSON |
| `--out` | โฟลเดอร์ผลทั้งโฟลเดอร์ (แทนชื่อตั้งต้นข้างบน) |
| `--config` | YAML ที่ override `configs/default.yaml` (วางก่อนชื่อคำสั่ง) |
| `-v` | แสดง log (วางก่อนชื่อคำสั่ง เช่น `python -m fin_extractor -v extract ...`) |

## Script วัดผล

ไม่ใช่ส่วนของ pipeline ใช้วัดและเทียบ ผลทั้งหมดอยู่ใต้ `data/output/benchmarks/`

| Script | ทำอะไร | ต้องมีมาก่อน | ผลลัพธ์ |
|---|---|---|---|
| `make_synthetic_samples.py` | สร้างใบกำกับภาษีสังเคราะห์ 21 ใบ พร้อมเฉลย | — | `data/raw/synthetic/` (ภาพ + `truth.json`) |
| `benchmark_preprocessing.py` | วัดว่าแต่ละชุด preprocessing ทำให้ Paddle อ่านค่าสำคัญได้ครบกี่ค่าจาก 6 | `make_synthetic_samples.py` | `preprocessing_synthetic.json` |
| `benchmark_cer.py [--engines paddle vlm]` | วัด CER บนชุดสังเคราะห์ (ข้อความอ้างอิงดึงจาก `render_page` ไม่ต้องมีเฉลยเพิ่ม) `vlm` ต้องมี Ollama + Typhoon | `make_synthetic_samples.py` | `cer_synthetic_<engines>.json` |
| `benchmark_real_scans.py` | เทียบ preprocessing 13 ชุดบนสแกนจริง ไม่ใช้เฉลย (~10 นาที) | สแกนใน `data/raw/finance_doc_scan/` | `real_scans.json` |
| `organize_preprocess_results.py` | จัดผลข้างบนเป็นโฟลเดอร์ต่อชุด | `benchmark_real_scans.py` | `preprocess_experiments/` |
| `compare_modes.py` | เทียบผล `extract` สองโหมดรายใบ | `extract data/raw/finance_doc_scan` ทั้ง `--mode ocr` และ `--mode vlm` (อ่าน `results/` ของ `finance_doc_scan__ocr-rules` และ `finance_doc_scan__vlm-llm`) | `compare_modes.md` |
| `benchmark.py prepare --per-cell 2` | โหลดชุด Invoice Extraction Benchmark จาก Hugging Face (ต้องต่อเน็ต) พร้อมเฉลย; `--thai <โฟลเดอร์>` เพิ่มเอกสารไทยที่มีเฉลย `.json` คู่กัน | — | `data/raw/benchmark/` |
| `benchmark.py run paddle\|paddle+llm\|vlm` | รันทีละ engine กับชุดข้างบน (`paddle+llm`, `vlm` ต้องมี Ollama) | `prepare` | `ieb/<engine>.jsonl` |
| `benchmark.py report` | สรุปความถูกต้องรายฟิลด์และเวลาของทุก engine ที่ `run` แล้ว | `run` | `ieb/report.md` |

ตัวอย่างแผนทดลองเทียบสองโหมดบนสแกนจริง:

```powershell
python -m fin_extractor extract data/raw/finance_doc_scan                # → extract/finance_doc_scan__ocr-rules
python -m fin_extractor extract data/raw/finance_doc_scan --mode vlm     # → extract/finance_doc_scan__vlm-llm
python scripts/compare_modes.py                                          # → benchmarks/compare_modes.md
```

## โครงสร้างโปรเจกต์

```
fin_extractor/
├── configs/default.yaml        ค่าตั้งต้นทั้งหมด
├── data/                       raw/ (เอกสารต้นฉบับ) · output/ (ผล) ไม่ถูก commit ยกเว้น .gitkeep
├── docs/                       เอกสารนี้
├── schemas/invoice.schema.json Data Contract (generate จาก schema.py)
├── scripts/                    benchmark (มีเฉลย) · benchmark_preprocessing + benchmark_cer + make_synthetic_samples (สังเคราะห์)
│                               · benchmark_real_scans + organize_preprocess_results + compare_modes (สแกนจริง ไม่ใช้เฉลย)
├── src/fin_extractor/
│   ├── schema.py               Data Contract (pydantic)
│   ├── config.py
│   ├── pipeline.py             ตัวควบคุมลำดับ
│   ├── cli.py
│   ├── runs.py                 โครงเก็บผล + manifest
│   ├── io/                     ขั้นที่ 1
│   ├── preprocessing/          ขั้นที่ 2
│   ├── ocr/                    ขั้นที่ 3A
│   ├── vlm/                    ขั้นที่ 3B + client สำหรับ LLM
│   ├── extraction/             ขั้นที่ 4
│   └── validation/             ขั้นที่ 5–6 + evidence รายฟิลด์
└── tests/
```

## แก้ไฟล์แล้วต้องรันอะไร

| แก้อะไร | รันอะไร |
|---|---|
| โค้ดทั่วไป | `pytest` |
| `schema.py` | `pytest` + `python -m fin_extractor schema` (เพิ่ม `SCHEMA_VERSION` ถ้าโครงเปลี่ยน) |
| extraction / validation / prompt | `pytest` + ลอง `extract` กับเอกสารจริงอย่างน้อยหนึ่งใบ (prompt/LLM ต้องลองโหมด vlm ด้วย เทสต์ไม่ครอบพฤติกรรมของ LLM) |
| `configs/default.yaml` | ลอง `extract` ถ้าแก้ preprocessing หรือ `document_types` |
| `make_synthetic_samples.py` | ระวัง: ผลวัดใน [02](02-preprocessing.md) อ้างภาพชุดปัจจุบัน สร้างใหม่แล้วตัวเลขเดิมอาจเทียบไม่ได้ |
| docs | ไม่ต้องรัน แต่ตัวเลขผลวัดต้องตรงกับผลจริงล่าสุดใน `data/output/benchmarks/` |

## ข้อจำกัดที่ยังไม่ได้ทดสอบ

- **PaddleOCR** รันจริงแล้วบนเอกสาร 2 ใบ (RTX 4050, CUDA 13), ชุดสังเคราะห์ 21 ใบ และสแกนจริง 18 ใบ (`--mode ocr` ราว 2–5 วินาที/ใบ ไม่นับใบแรกที่โหลดโมเดล) ผลเทียบ engine เก่า (รวม Tesseract/EasyOCR ที่เลิกใช้แล้ว)
  อยู่ใน [03-text-reading.md](03-text-reading.md#ผลเทียบสาม-engine-บนเครื่องนี้) — ยังน้อยเกินกว่าจะสรุป
- **Prompt ของ Typhoon OCR** (`vlm/prompts.py`) อิงรูปแบบใน model card ควรเทียบกับ model card ของรุ่นที่ใช้อีกครั้ง
- **Typhoon API บน cloud**: มี config แล้วแต่ยังไม่ได้เรียกจริง
- **rules ถูกปรับให้เข้ากับ `BITKUB-001.jpg`** ชุดสแกนจริง 18 ใบ (`data/raw/finance_doc_scan`) ที่รันแล้วไม่มีใบไหนได้ `auto_approved`
  ในโหมด ocr (11 `needs_review`, 7 `failed`) ส่วนโหมด vlm ได้ `auto_approved` 1 ใบ แต่เป็นการอนุมัติที่ผิด (ดู [05](05-validation-output.md#ขั้นที่-6-สถานะเอกสาร)) เพราะมีเอกสารหลายชนิดนอกใบกำกับภาษี (หนังสือรับรอง 50 ทวิ, ใบสำคัญรับเงินลายมือ, ใบเสร็จศุลกากร)
  และ rules บางส่วนพลาดแม้ OCR อ่านครบ (เช่น 001310) รายละเอียดผล preprocess ดู [02-preprocessing.md](02-preprocessing.md#ผลวัดบนสแกนจริง)
- **LLM ขนาด 3B** ยังสลับ field ยอดเงินได้ ใช้รุ่นใหญ่ขึ้นจะเสถียรกว่า
- **วันที่:** ถ้าช่องวันที่ถูกบังหรือว่าง ระบบจะหยิบวันที่แรกในเอกสาร ซึ่งอาจไม่ใช่วันที่ออกเอกสาร
