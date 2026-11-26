# ติดตั้งและวิธีใช้

## ติดตั้ง (เครื่องใหม่ตั้งแต่ clone)

ต้องมี Python 3.10 ขึ้นไป (ทดสอบกับ 3.13) ต้องลง `paddlepaddle` (ตัว framework) **ก่อน** แล้วค่อยลงโปรเจกต์ เพราะขั้น preprocess ใช้ตัวจำแนกทิศของ Paddle (`preprocessing/orient.py`) และ `--engine paddle` ใช้ PaddleOCR ส่วนตัวอ่านค่าตั้งต้น OpenThai ลงเพิ่มตามตาราง "ส่วนเสริม" (`.[all]` ไม่รวม)

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
- รันครั้งแรกจะโหลดโมเดลของ Paddle (ช้ากว่าปกติ) เก็บไว้ที่ `models/paddlex/official_models/` **ในโปรเจกต์** (ไม่ commit) ไม่ใช่ `~/.paddlex` โค้ดตั้ง `PADDLE_PDX_CACHE_HOME` ให้เอง (ถ้าตั้งไว้เองก่อนแล้วจะไม่ทับ) ย้ายเครื่องแล้วคัดลอกโฟลเดอร์ `models/` ไปด้วยก็ไม่ต้องโหลดใหม่ ตั้ง
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` เพื่อข้ามการเช็กเซิร์ฟเวอร์โมเดลทุกครั้งที่เริ่ม
  (PowerShell: `$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"`)
- Windows: ใส่ `-X utf8` (`python -X utf8 -m fin_extractor ...`) ถ้าคอนโซลแสดงภาษาไทยเพี้ยนหรือ error `charmap`

### ส่วนเสริม (ตามที่จะใช้)

| สิ่งที่ต้องมี | ใช้กับ | ติดตั้ง |
|---|---|---|
| Ollama + Typhoon OCR | `sections --engine typhoon` | `ollama pull scb10x/typhoon-ocr1.5-3b` |
| Ollama + LLM | การสกัดด้วย LLM (`extract --llm on\|hybrid`) | `ollama pull qwen2.5:3b` (เปลี่ยนรุ่นที่ `llm_extractor.model`) |
| OpenThai 2.0 OCR (iApp, 0.63B) | `sections --engine openthai` (ค่าตั้งต้น) | `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128` แล้ว `pip install -e ".[openthai]"` และ `hf download iapp/OpenThai2.0-OCR-Markdown --local-dir models/OpenThai2.0-OCR-Markdown` (index cu128 มี torch สูงสุด 2.11; ถ้าลง torch จาก PyPI เฉย ๆ บน Windows จะได้รุ่น CPU) ทดสอบบน RTX 4050 6GB: VRAM สูงสุด ~2.4GB, 1 หน้า ~20 วินาที (รวมโหลดโมเดล) |
| OpenThai 2.1 OCR Markdown | `sections --engine openthai21` (ทดลอง ผลวัดดู [04-openthai-ocr.md](04-openthai-ocr.md) หัวข้อ 6) | ตัวอ่านและ torch ชุดเดียวกับ 2.0 แค่ดาวน์โหลด `hf download iapp/OpenThai2.1-OCR-Markdown-20261118 --local-dir models/OpenThai2.1-OCR-Markdown-20261118` (2.6 GB) VRAM ~1.4 GB |
| OpenThai 2.1 OCR Text (CPU) | ชั้นตรวจความครบของ `sections` (ค่าตั้งต้น `coverage_check.engine` ถ้าไม่มี onnxruntime หรือโมเดลจะข้ามพร้อมข้อความ ไม่ทำให้ flow ล้ม) และ `sections --engine openthai-text` | `pip install onnxruntime` (ไม่อยู่ใน extra ใด ต้องลงเอง) และ `hf download iapp/OpenThai2.1-OCR-Text-20261118 --local-dir models/OpenThai2.1-OCR-Text-20261118` (9 MB) ไม่ใช้ GPU |
| ฟอนต์ | `scripts/data/make_synthetic_samples.py`, หน้าเว็บเดโม `scripts/demo/ocr_demo_app.py` | **มากับโปรเจกต์แล้ว** ที่ `assets/fonts/` (Sarabun, SIL OFL) ไม่ใช้ฟอนต์ของเครื่อง ไม่ต้องติดตั้ง |

**อะไรอยู่ในโปรเจกต์ อะไรอยู่นอก:**

| ของ | ที่อยู่ | หมายเหตุ |
|---|---|---|
| ฟอนต์ | `assets/fonts/` (commit) | ไม่ใช้ `C:/Windows/Fonts` |
| โมเดล Paddle (detection, recognition, orientation) | `models/paddlex/` (ไม่ commit) | ตั้งผ่าน `PADDLE_PDX_CACHE_HOME` อัตโนมัติ |
| โมเดล OpenThai 2.0 OCR | `models/OpenThai2.0-OCR-Markdown/` (ไม่ commit) | ดาวน์โหลดด้วย `hf download` ตามตารางส่วนเสริม |
| ไลบรารี Python | `.venv/` | |
| ข้อมูล ผลลัพธ์ cache | `data/` | |
| โมเดลของ Ollama (Typhoon, qwen) | **นอกโปรเจกต์** (ที่เก็บของ Ollama เอง เช่น `~/.ollama`) | Ollama เป็นบริการแยก ย้ายได้ด้วยตัวแปรแวดล้อม `OLLAMA_MODELS` ฝั่งเซิร์ฟเวอร์ของ Ollama โปรเจกต์ตั้งให้ไม่ได้ |

**ใช้ Typhoon API บน cloud แทน Ollama:** คัดลอก `.env.example` เป็น `.env` ใส่ `TYPHOON_API_KEY` แล้วตั้ง `typhoon.provider: openai` และ
`typhoon.base_url: https://api.opentyphoon.ai/v1` (ส่งเอกสารการเงินออกนอกเครื่อง ตรวจนโยบายข้อมูลก่อน) engine อื่น (`openthai*`, `paddle`) กับ `--llm off` ไม่ต้องใช้ทั้ง Ollama และ key

## ข้อมูลตัวอย่าง

ทุกโฟลเดอร์ใต้ `data/` อยู่ใน git เพียงแค่ `.gitkeep` เนื้อหาข้างในถูก `.gitignore` ไว้ (เอกสารจริงอาจมีข้อมูลภาษีและข้อมูลส่วนบุคคล)
หลัง clone จึงไม่มีเอกสารให้รัน มีสองทาง

1. **สแกนจริง:** รับชุดสแกนจากเจ้าของโปรเจกต์นอก git แล้ววางที่ `data/01_input/finance_doc_scan/` (ไฟล์ `.pdf`, `.jpg`, `.png`, `.tif`, `.bmp`, `.webp`)
2. **ชุดสังเคราะห์** (ไม่ต้องรอข้อมูลจริง): `python scripts/data/make_synthetic_samples.py` ได้ใบกำกับภาษีสมมติ 21 ใบพร้อมเฉลยที่ `data/01_input/synthetic/` (วาดด้วยฟอนต์ Sarabun ใน `assets/fonts/` ชุดที่สร้างไว้ก่อน 2026-11-11 ใช้ Tahoma ภาพจึงต่างจากชุดใหม่เล็กน้อย ตัวเลข CER บนชุดสังเคราะห์ในเอกสาร (บันทึกเก่า) มาจากชุด Tahoma เดิม)

## รันครั้งแรก

```powershell
python scripts/data/make_synthetic_samples.py                      # ถ้ายังไม่มีเอกสาร
python -m fin_extractor sections data/01_input/synthetic --engine paddle   # Paddle ทีละท่อน + rules ไม่ต้องใช้ Ollama
```

ได้ผลที่ `data/05_extract/synthetic/paddle/regions-luma/rules/` (ข้อความรายส่วนที่ `04_ocr/synthetic/paddle/regions-luma/` ดูโครงข้างล่าง) และบนหน้าจอจะแสดง status กับผล check ของแต่ละใบ
ตัวโหลดไฟล์กรองเฉพาะนามสกุลภาพ/PDF จึงไม่หยิบ `truth.json` ในโฟลเดอร์เดียวกัน

**ห้ามอ่านทั้งหน้า:** ทุก engine (`openthai` ตั้งต้น · `openthai21` · `openthai-text` · `typhoon` · `paddle`) อ่านทีละท่อนผ่าน `sections` เท่านั้น (ไม่มีคำสั่ง `ocr` แล้ว)
Typhoon ต้องเปิด Ollama ก่อน:

```powershell
python -m fin_extractor sections data/01_input/finance_doc_scan --engine typhoon
```

## โครงเก็บผลลัพธ์

ข้อมูลทุกอย่างอยู่ใต้ `data/` และไม่ถูก commit (เก็บเฉพาะ `.gitkeep`) ตำแหน่งทั้งหมดกำหนดที่เดียวใน `src/fin_extractor/paths.py`
ชื่อไฟล์ผลของทุกขั้นเป็น `<ไฟล์>_p<หน้า>` เหมือนกัน จึงดูไฟล์เดียวข้ามขั้น/ข้าม engine ได้ และลบ engine ที่ไม่ใช้ได้ด้วยการลบโฟลเดอร์เดียว

```
data/
├── 01_input/<dataset>/                         ไฟล์อินพุต (dataset = ชื่อโฟลเดอร์ เช่น finance_doc_scan, synthetic)
├── 02_preprocess/<dataset>/<profile>/          ผล preprocess หนึ่งชุดตั้งค่า (profile ตั้งต้นชื่อ default)
│   ├── final/<ไฟล์>_p<หน้า>.png                ภาพสุดท้ายที่ส่งให้ OCR/VLM ทุกตัว ใช้ร่วมกัน (ไม่ preprocess ซ้ำ)
│   ├── steps/                                  ภาพทีละขั้น (เมื่อใส่ --steps)
│   └── settings.json · manifest.json
├── 03_sections/<dataset>/<profile>/            แบ่งเป็นส่วนจาก final/
│   ├── regions/<ไฟล์>/                         ค่าตั้งต้น (docs/03-regions): 01_normalize … 06_pieces.png · crops/ (ภาพที่ส่ง OCR + index.json) · sections.json · manifest.json
│   ├── layout/<ไฟล์>.json · <ไฟล์>_layout.png  ผังแบบเดิม (--layout sections, docs/03-sections): ผังแบ่ง + ภาพกรอบสี
│   └── images/<luma|flat|bin>/<ไฟล์>/sNN.png   ผังแบบเดิม: ภาพรายส่วนก่อน OCR + index.json
├── 04_ocr/<dataset>/<engine>/                  engine = openthai | openthai21 | openthai-text | typhoon | paddle
│   ├── regions-luma/<ไฟล์>/                    อ่านทีละท่อน (ค่าตั้งต้น): .md .sections.json (รวมกรอบบล็อกของ OpenThai) .chunks.json (ที่มาของแต่ละท่อน)
│   └── <luma|flat|bin|color>/<ไฟล์>/           อ่านทีละ section ของผังแบบเดิม
├── 05_extract/<dataset>/<engine>/<วิธี>/<extractor>/<ไฟล์>_p<หน้า>.json   ผลตาม schema (extractor = rules | llm | hybrid) + manifest.json
└── etc/                                        งานที่ข้าม engine/ข้ามไฟล์
    ├── compare/<เลขไฟล์>/                      เทียบรายไฟล์ + ค่าอ้างอิงที่คนอ่านจากภาพ
    ├── groundtruth/<dataset>/<ไฟล์>.json       ค่าที่ถูกตามเอกสาร คนกรอกในเครื่อง (scripts/eval/groundtruth_app.py)
    └── demo/<งาน>/                             ผลของหน้าเว็บเดโม
```

**โฟลเดอร์ย่อยต่อไฟล์:** ถ้ารันหนึ่งครั้งมีไฟล์อินพุต **มากกว่า 2 ไฟล์** ผลของแต่ละไฟล์ (`02_preprocess/…/final/` และ `steps/`, `04_ocr/…/<วิธี>/`, `05_extract/…/<extractor>/`) จะอยู่ในโฟลเดอร์ย่อยชื่อไฟล์ เช่น `04_ocr/finance_doc_scan/openthai/regions-luma/BRWAC50DE3689DA_001312/BRWAC50DE3689DA_001312_p1.md` ถ้าไม่เกิน 2 ไฟล์ (เช่นลองไฟล์เดียว) จะวางตรง ๆ ใต้โฟลเดอร์นั้น ค่า 2 อยู่ที่ `NEST_OVER` ใน `src/fin_extractor/paths.py` ตัวอ่าน (`extract`, สคริปต์เทียบ) หาได้ทั้งสองแบบ ผลที่ปนกันจึงอ่านได้ และ `manifest.json` ใช้พาธ `<ไฟล์>/<ไฟล์>_p1.json` ในช่อง `output`

| คำสั่ง | เขียนที่ | ข้างใน |
|---|---|---|
| `preprocess` | `data/02_preprocess/<dataset>/<profile>/` | `final/` ภาพสุดท้าย, `steps/` เมื่อใส่ `--steps`, `settings.json` (ค่าที่ทำให้ได้ภาพนี้), `manifest.json` |
| `sections` ขั้น ocr | `data/04_ocr/<dataset>/<engine>/<วิธี>/<ไฟล์>/` | `.sections.json` (ผลรายส่วน + `blocks` กรอบบล็อกของ OpenThai) · `.md` (ต่อทุกส่วน + ข้อความเสริมจากชั้นตรวจความครบ) · `.chunks.json` (ท่อนข้อความ ↔ ภาพ section) |
| `extract` (หรือ `sections` ขั้น extract) | `data/05_extract/<dataset>/<engine>/<วิธี>/<extractor>/` (อ่านข้อความจาก `data/04_ocr/<dataset>/<engine>/<วิธี>/`) | `<ไฟล์>_p<หน้า>.json` ตาม `schemas/invoice.schema.json` + `manifest.json` |
| `sections` ขั้น layout (หรือคำสั่ง `regions`) | `data/03_sections/<dataset>/<profile>/regions/<ไฟล์>/` | ภาพแต่ละขั้น · `crops/` ภาพที่ส่ง OCR · `sections.json` (ผัง) · `manifest.json` (QR ที่ถอดได้ ท่อนที่ไม่ส่ง) |
| `sections --layout sections` (รันทีละขั้นด้วย `--to layout\|images\|ocr`) | `data/03_sections/<dataset>/<profile>/` | ผังแบบเดิม: `layout/<ไฟล์>.json` + `_layout.png` และ `images/<luma\|flat\|bin>/<ไฟล์>/` ภาพย่อยแยกโฟลเดอร์ตามวิธี (ขั้นอ่านเขียนที่ `04_ocr/<dataset>/<engine>/<วิธี>/` ขั้นสกัดที่ `05_extract/…`) |

- **engine** ในชื่อโฟลเดอร์ = ค่าของ `--engine` (`openthai` `openthai21` `openthai-text` `typhoon` `paddle`) · **วิธี** = `regions-luma` (ค่าตั้งต้น) หรือ `--variant` ของผังแบบเดิม (`luma` `flat` `bin` `color`) · **extractor**: `llm` เมื่อใช้ LLM สกัด, `hybrid` เมื่อ `--llm hybrid`, ไม่งั้น `rules`
- **OCR ครั้งเดียว สกัดได้หลายแบบ:** ขั้น ocr เก็บข้อความรายส่วนไว้ที่ `04_ocr/<dataset>/<engine>/<วิธี>/` แล้ว `extract --engine <ตัวเดิม>` สกัดซ้ำจากข้อความนั้น (เปลี่ยน rules ↔ LLM, แก้ขั้นสกัดแล้วลองใหม่) **โดยไม่อ่านภาพและไม่ใช้ GPU**
- **ภาพ preprocess ใช้ร่วมกันทุก engine:** ถ้าตั้งค่า preprocess เปลี่ยนจากที่เก็บไว้ ระบบจะไม่ปนภาพคนละตั้งค่า ให้ใช้ `--profile <ชื่อใหม่>` หรือ `--force` (ล้างโฟลเดอร์ profile นั้นแล้วทำใหม่)
- รันซ้ำด้วยอินพุตและวิธีเดิมจะเขียนทับ `extract --out <โฟลเดอร์>` อ่านข้อความจาก `<โฟลเดอร์>/04_ocr/<วิธี>/` และเขียนผลที่ `<โฟลเดอร์>/05_extract/<วิธี>/<extractor>/` แทน (`sections` ไม่มี `--out`)
- `manifest.json` เขียนตอนจบ ถ้ารันล้มกลางทางจะมี `"status": "failed"` พร้อมไฟล์ที่ทำเสร็จแล้ว

## คำสั่ง

| คำสั่ง | ขั้นที่ทำ | ผลลัพธ์ |
|---|---|---|
| `python -m fin_extractor preprocess <input> [--profile P] [--steps] [--force] [--orientation 90/180/270]` | 1 → 2 | `data/02_preprocess/<dataset>/<profile>/` |
| `python -m fin_extractor extract <input> [--engine openthai\|openthai21\|openthai-text\|typhoon\|paddle] [--variant regions-luma\|luma\|flat\|bin\|color] [--llm off\|on\|auto\|hybrid] [--only 1309 …]` | 5 (สกัดซ้ำจากข้อความรายส่วนที่มี ไม่อ่านภาพ) | `data/05_extract/<dataset>/<engine>/<วิธี>/<extractor>/` |
| `python -m fin_extractor sections <input> [--variant luma\|flat\|bin\|color] [--engine openthai\|openthai21\|openthai-text\|typhoon\|paddle] [--from ขั้น] [--to ขั้น] [--only 1309 …] [--redo] [--no-coverage-check] [--layout regions\|sections] [--table-mode whole\|cells\|bands\|rows] [--max-piece-frac F] [--fill-aware] [--dotted-rules] [--no-mask-codes] [--trim-sections all\|text-lr] [--drop-empty-rows] [--profile P]` | flow หลัก: preprocess → layout → images → ocr (+ ชั้นตรวจความครบ) → extract (rules อย่างเดียว) (ชื่อขั้นใน `--from`/`--to`: `preprocess` `layout` `images` `ocr` `extract`) ผังค่าตั้งต้น `regions` (ไม่มีขั้น images ผลที่ `<engine>/regions-luma/`) · `--no-coverage-check` ปิดชั้นตรวจความครบ · `--layout sections` = ผังแบบเดิม ตัวเลือก `--variant` `--table-mode` `--max-piece-frac` `--fill-aware` `--dotted-rules` `--no-mask-codes` `--trim-sections` `--drop-empty-rows` ใช้กับแบบเดิมเท่านั้น (เขียนทีละ variant ต่อการรัน) | `data/02_preprocess` … `data/05_extract` ตามขั้น |
| `python -m fin_extractor regions <input> [--only 1307 …] [--no-debug-images] [--profile P]` | ขั้น 3 (ผัง regions ค่าตั้งต้นของ `sections`) รันขั้นนี้อย่างเดียว: region (แถบตามเส้น/ช่องว่าง) + ตาราง → ช่อง + key/value + ระบาย QR จากภาพ final (CPU ไม่ OCR ไม่แตะผังเดิม) ตั้งค่าที่ `configs/regions/` ดู [03-regions.md](03-regions.md) | `data/03_sections/<dataset>/<profile>/regions/` |
| `python -m fin_extractor compare <input> [--engine openthai] [--variants luma flat bin] [--out ไฟล์.md]` | — | ตารางเทียบผลสกัดข้ามวิธี พิมพ์ที่หน้าจอ (ใส่ `--out` เขียน .md ด้วย) แถวที่เทียบไม่แฟร์ถูกติดธงและไม่นับในยอดรวม |
| `python -m fin_extractor schema` | — | `schemas/invoice.schema.json` (generate จาก `schema.py` ห้ามแก้ไฟล์ตรง ๆ) |
| `pytest` | รันเทสต์ | |

ตัวเลือกของ `sections` ที่เกี่ยวกับสิ่งที่ส่งเข้า OCR:

| ตัวเลือก | ความหมาย |
|---|---|
| (ค่าตั้งต้น) ชั้นตรวจความครบ · `--no-coverage-check` ปิด | หลังขั้น ocr ให้ OpenThai 2.1 Text (CPU, `coverage_check` ใน config) อ่านภาพย่อยซ้ำ ภาพที่ตัวอ่านหลักอ่านไม่ครบ: แยกซ้าย/ขวาให้ตัวอ่านหลักอ่านใหม่ หรือแนบข้อความเสริมใน `.md` ไม่มี onnxruntime/โมเดลก็ข้าม ([04-openthai-ocr.md](04-openthai-ocr.md)) |
| (ผังแบบเดิม ค่าตั้งต้น) ระบาย QR/บาร์โค้ด · `--no-mask-codes` ปิด | หา QR/บาร์โค้ดแล้วระบายขาวก่อนแบ่ง ไม่ส่งเข้า OCR ข้อความในรหัสเก็บใน `masked_codes` ของผัง ([03-sections.md](03-sections.md)) |
| `--trim-sections all\|text-lr` | หดกรอบส่วนให้เหลือแค่ข้อความ วัดกับ OpenThai แล้วไม่ดีขึ้นทั้งสองแบบ จึงไม่เปิดเป็นค่าตั้งต้น ([03-sections.md](03-sections.md)) |

**หน้าเว็บลอง OCR ทีละขั้น** (`scripts/demo/ocr_demo_app.py`): อัปโหลดเอกสาร 1 ไฟล์ แล้วดูผลทุกขั้นของ flow หลัก (preprocess → ผังแบ่งส่วนพร้อมภาพกลางทาง → ภาพรายส่วน → OCR ทีละส่วน → บรรทัด → ฟิลด์ → check/สถานะ → JSON) คลิกรหัสส่วนเพื่อตามส่วนนั้นข้ามทุกขั้น

```bash
python -X utf8 scripts/demo/ocr_demo_app.py          # เปิด http://127.0.0.1:8766 (เฉพาะเครื่องนี้)
```

- ตั้งค่าเป็นค่าตั้งต้นของ `sections` ให้เอง (`FLOW_DEFAULTS` ในสคริปต์: ผัง regions) เลือกได้แค่ตัวอ่านและวิธีสกัด ผังแบบเดิมยังดูได้โดยเปลี่ยน `FLOW_DEFAULTS["layout"]` เป็น `"sections"`
- ตัวเลือก **สกัดฟิลด์**: rules อย่างเดียว (ค่าตั้งต้น) · hybrid · winner เหมือน `extract --llm off|hybrid|on` (ต้องเปิด Ollama และมีโมเดลตาม `llm_extractor.model`) ขั้นสกัดแสดงค่าที่ต่างจาก rules อย่างเดียวพร้อมที่มา ค่าที่ด่านของ hybrid ปัดตก และแจ้งเมื่อเรียก LLM ไม่ได้ (ถอยไปใช้ rules) ค่าจาก LLM มีป้าย "LLM" ทั้งในขั้นสกัดและหน้าตรวจ · **ทั้ง 3 แบบ เทียบกัน** = สกัด rules / hybrid / winner จากข้อความชุดเดียวกัน แสดงสรุปต่อแบบ (สถานะ ฟิลด์ขาด check ค่าที่ไม่พบในข้อความ ค่าที่ด่านปัดตก เวลา) และค่ารายฟิลด์ข้างกัน ฟิลด์ที่ได้ค่าต่างกันขึ้นก่อน ขั้นตรวจ/แก้และ JSON ใช้ rules ก่อน กด "ใช้แบบนี้" เพื่อสลับโดยไม่สกัดใหม่
- หัวหน้าเว็บแสดง branch/commit ที่เซิร์ฟเวอร์เปิดมา หน้าเว็บอ่านไฟล์ครั้งเดียวตอนเริ่ม **สลับ git branch แล้วต้องรีสตาร์ตเซิร์ฟเวอร์**
- ขั้นตรวจ/แก้: ช่องชื่อ ที่อยู่ รายละเอียดรายการ และฟิลด์ข้อความของโปรไฟล์ใช้กล่องหลายบรรทัด
- ขั้น **OCR ทีละส่วน** วาดกรอบบล็อกที่ OpenThai บอกเองว่าอ่านจากตรงไหน (กรอบ + ชนิด เช่น Text/Table มากับคำตอบของโมเดล `ReadResult.blocks`) ตรงที่มีข้อความแต่ไม่มีกรอบ = ไม่ได้อ่าน · flow หลักเก็บกรอบไว้ใน `.sections.json` (`blocks`) ด้วย ตัวอ่านอื่นไม่มีกรอบ
- ขั้น **ตรวจ/แก้รายฟิลด์** (`scripts/demo/review.py`): เรียงฟิลด์ตามที่ควรตรวจก่อน (ขัดแย้ง → หาไม่เจอ → ไม่มีตัวยืนยัน → ยืนยันแล้ว) พร้อมเหตุผลเป็นคำพูดจากหลักฐานที่ระบบมี (ไม่ใช้ decision engine) แก้ค่า/รายการสินค้าแล้วตรวจใหม่ทันทีผ่าน `FinanceOCRPipeline.assess` บันทึกได้เป็น `corrected.json` (ผลหลังแก้) + `edits.json` (ค่าเดิม → ค่าที่แก้) ในโฟลเดอร์งาน
- ขั้นที่โหลดโมเดล (preprocess ที่ใช้ตัวตรวจทิศของ Paddle และ OCR) รันใน `scripts/demo/ocr_demo_worker.py` แยก process รันได้ทีละงาน ใช้หน้าแรกของ PDF
- ผลแต่ละงานอยู่ที่ `data/etc/demo/<งาน>/` (ไม่ commit) ใช้ฟังก์ชันเดียวกับ pipeline จริง ไม่ได้เขียนตรรกะใหม่

ตัวเลือกของ `extract`:

| ตัวเลือก | ความหมาย |
|---|---|
| `--engine` | ตัวอ่านที่สร้างข้อความรายส่วน (`openthai` ตั้งต้น · `openthai21` · `openthai-text` · `typhoon` · `paddle`) อ่านจาก `04_ocr/<dataset>/<engine>/<วิธี>/` |
| `--variant regions-luma\|luma\|flat\|bin\|color` | ภาพรายส่วนแบบที่ใช้ตอนอ่าน (`regions-luma` ตั้งต้น = flow หลัก · ที่เหลือ = ผัง section แบบเดิม) ผลเขียนที่ `05_extract/…/<วิธี>/<extractor>/` |
| `--llm off\|on\|auto\|hybrid` | `off` = rules อย่างเดียว (ตั้งต้น) · `on`/`auto` = rules และ LLM สกัดแยก เลือกชุดที่ check ดีกว่า · `hybrid` = rules นำ LLM เติมเฉพาะช่องที่ยังไม่เสร็จ (ผลลง `…/hybrid/`) |
| `--profile` | ชื่อชุดตั้งค่า preprocess (ใช้หาไฟล์ ไม่แตะภาพ) |
| `--only` | สกัดเฉพาะไฟล์ที่ชื่อลงท้ายด้วยเลขเหล่านี้ เช่น `--only 1309 1310` (ไม่ใส่ = ทุกไฟล์) |
| `--keep-raw` | เก็บข้อความดิบใน JSON ผลสุดท้ายด้วย (ตั้งต้นเก็บเฉพาะใน `04_ocr/…`) |
| `--out` | โฟลเดอร์เอง: อ่านข้อความจาก `<out>/04_ocr/<วิธี>/` เขียนผลที่ `<out>/05_extract/<วิธี>/<extractor>/` |
| `--config` | YAML ที่ override `configs/default.yaml` (วางก่อนชื่อคำสั่ง) |
| `-v` | แสดง log (วางก่อนชื่อคำสั่ง เช่น `python -m fin_extractor -v extract ...`) |

## Script วัดผล

ไม่ใช่ส่วนของ pipeline ใช้สร้างข้อมูลทดสอบและเทียบผลสกัด พาธในตารางนับจาก `scripts/` (โฟลเดอร์ย่อย: `data` · `demo` (หน้าเว็บเดโม ดูหัวข้อก่อนหน้า) · `eval` (05))

สคริปต์วัดผลแบบอ่านทั้งหน้า (`eval/benchmark.py`, `eval/compare_modes.py`, `ocr/benchmark_cer.py`, `ocr/visualize_bbox.py`, `ocr/oxr/*`, `preprocess/benchmark_preprocessing.py`, `preprocess/benchmark_real_scans.py`, `preprocess/organize_preprocess_results.py`, `preprocess/make_preprocess_notebooks.py`) และ `sections/make_section_layouts.py` · `make_clean_sections.py` · `run_sections.py` ลบแล้ว (อยู่ในประวัติ git ถึง commit f6cc6eb) ผลวัดที่สร้างไว้ในเอกสารเป็นบันทึกของรอบนั้น

| Script | ทำอะไร | ต้องมีมาก่อน | ผลลัพธ์ |
|---|---|---|---|
| `data/make_synthetic_samples.py` | สร้างใบกำกับภาษีสังเคราะห์ 21 ใบ พร้อมเฉลย | — | `data/01_input/synthetic/` (ภาพ + `truth.json`) |
| `eval/compare_file.py <เลขไฟล์>` | เทียบผลสกัดของไฟล์เดียวรายฟิลด์ ข้าม engine/extractor (ไม่ใช้ GPU) มีคอลัมน์อ้างอิงถ้าใส่ `reference.json` | ผล `extract` ของแต่ละ engine | `data/etc/compare/<ไฟล์>/` |
| `eval/groundtruth_app.py <input>` | หน้าเว็บในเครื่อง (`127.0.0.1` เท่านั้น) กรอก ground truth: ภาพเอกสารคู่ฟอร์มฟิลด์ เติมค่าจากผลสกัด (`--prefill`, ค่าตั้งต้น `openthai/regions-luma/rules`) คนยืนยันหรือแก้ ช่องที่ต่างจากระบบเป็นสีเหลือง | ภาพ final + ผล `extract` | `data/etc/groundtruth/<dataset>/<ไฟล์>.json` |
| `eval/summarize_extract.py <โฟลเดอร์ผล>… [--detail]` | สรุปผล `extract` หลายชุด (เช่น rules / llm / hybrid) เทียบกันโดยไม่ต้องมีเฉลย: สถานะ, ฟิลด์ขาด, check, ค่าที่ไม่พบในข้อความ, conflict, ฟิลด์จาก LLM, ค่าที่ด่านปัดตก `--detail` = รายใบ | ผล `extract` | พิมพ์ตาราง Markdown |
| `eval/score_groundtruth.py <input> [--runs …] [--by-field]` | ให้คะแนนผลสกัดทุกชุดเทียบ ground truth ที่ตรวจเสร็จ: accuracy = ถูก / ช่องที่เอกสารมีค่า, ผิด, ไม่ได้ค่า, ใส่เกิน พิมพ์แค่จำนวนนับ ไม่พิมพ์ค่า | ground truth + ผล `extract` | `data/etc/groundtruth/<dataset>/score.md` |

ตัวอย่างแผนทดลองเทียบ engine บนสแกนจริง (ทุกตัวอ่านทีละท่อนจากผังและภาพชุดเดียวกัน):

```powershell
python -m fin_extractor sections data/01_input/finance_doc_scan                      # OpenThai → 05_extract/finance_doc_scan/openthai/regions-luma/rules/
python -m fin_extractor sections data/01_input/finance_doc_scan --engine typhoon     # → 05_extract/finance_doc_scan/typhoon/regions-luma/rules/ (ข้ามขั้นที่ผลยังใช้ได้)
python -m fin_extractor extract data/01_input/finance_doc_scan --engine typhoon --llm on   # LLM จากข้อความ Typhoon เดิม ใช้แค่ Ollama → …/typhoon/regions-luma/llm/
python scripts/eval/compare_file.py 001312                                          # เทียบรายฟิลด์ข้าม engine
```

## โครงสร้างโปรเจกต์

โครงโฟลเดอร์และหน้าที่ของแต่ละไฟล์อยู่ใน [README หลัก](../README.md) หัวข้อ "โครงสร้างโฟลเดอร์" (ที่เดียว ไม่ซ้ำที่นี่)

## แก้ไฟล์แล้วต้องรันอะไร

| แก้อะไร | รันอะไร |
|---|---|
| โค้ดทั่วไป | `pytest` |
| `schema.py` | `pytest` + `python -m fin_extractor schema` (เพิ่ม `SCHEMA_VERSION` ถ้าโครงเปลี่ยน) |
| extraction / validation / prompt | `pytest` + ลอง `extract` กับเอกสารจริงอย่างน้อยหนึ่งใบ (prompt/LLM ต้องลอง `extract --llm on\|hybrid` ด้วย เทสต์ไม่ครอบพฤติกรรมของ LLM) |
| `configs/default.yaml` | ลอง `extract` ถ้าแก้ preprocessing หรือ `document_types` |
| ขั้นสกัด / `document_types` | `python -m fin_extractor extract <input> --engine <ตัวที่อ่าน>` (ไม่อ่านภาพ) สกัดและตรวจซ้ำจากข้อความดิบใน `04_ocr/` ไม่เรียก OCR/GPU (วินาทีเดียว) เทียบก่อน/หลังบนข้อความเดิมได้ |
| `make_synthetic_samples.py` | ระวัง: ผลวัดใน [02](02-preprocessing.md) อ้างภาพชุดปัจจุบัน สร้างใหม่แล้วตัวเลขเดิมอาจเทียบไม่ได้ |
| docs | ไม่ต้องรัน แต่ตัวเลขผลวัดต้องตรงกับผลจริงล่าสุดใน `data/05_extract/` (ผลวัดเก่าจากสคริปต์ที่ลบแล้วให้ระบุว่าเป็นบันทึกของรอบนั้น) |

## ข้อจำกัดที่ยังไม่ได้ทดสอบ

- **PaddleOCR** รันจริงแล้วบนเอกสาร 2 ใบ (RTX 4050, CUDA 13), ชุดสังเคราะห์ 21 ใบ และสแกนจริง 18 ใบ (ตอนอ่านทั้งหน้าแบบเดิม ราว 2–5 วินาที/ใบ ไม่นับใบแรกที่โหลดโมเดล) ผลเทียบ engine ที่เคยลองดู
  อยู่ใน [04-ocr.md](04-ocr.md) — ยังน้อยเกินกว่าจะสรุป
- **Prompt ของ Typhoon OCR** (`TYPHOON_OCR_PROMPT` ใน `readers/typhoon.py`) อิงรูปแบบใน model card ควรเทียบกับ model card ของรุ่นที่ใช้อีกครั้ง
- **Typhoon API บน cloud**: มี config แล้วแต่ยังไม่ได้เรียกจริง
- **rules ถูกปรับให้เข้ากับ `BITKUB-001.jpg`** ชุดสแกนจริง 18 ใบ (`data/01_input/finance_doc_scan`) มีเอกสารหลายชนิดนอกใบกำกับภาษี (หนังสือรับรอง 50 ทวิ, ใบสำคัญรับเงินลายมือ, ใบเสร็จศุลกากร)
  ผล OpenThai ผัง regions + rules ล่าสุด (2026-11-26 ไม่มีเฉลย): 11 `check` · 5 `ready` · 2 `incomplete` — `ready` ไม่ได้แปลว่าค่าถูก (ผ่านได้ด้วย checksum เลขภาษีอย่างเดียว ควรตรวจมือก่อนเชื่อ)
  ตัวเลขและเหตุผลอยู่ใน [05-extraction.md](05-extraction.md) เกณฑ์สถานะใน [05-validation-output.md](05-validation-output.md) และผล preprocess ใน [02-preprocessing.md](02-preprocessing.md#ผลวัดบนสแกนจริง)
- **LLM ขนาด 3B** ยังสลับ field ยอดเงินได้ ใช้รุ่นใหญ่ขึ้นจะเสถียรกว่า
- **วันที่:** ถ้าช่องวันที่ถูกบังหรือว่าง ระบบจะหยิบวันที่แรกในเอกสาร ซึ่งอาจไม่ใช่วันที่ออกเอกสาร
