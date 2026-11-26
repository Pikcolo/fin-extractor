"""ผัง regions — ขั้น 3 ค่าตั้งต้นของ flow (``sections --layout regions``) ผังแบบเดิมอยู่ที่ sections/layout.py (``--layout sections``)

รวมเทคนิคจาก learning/preprocess_finance: normalize ตามขอบเขตเนื้อหา, ระบาย QR/บาร์โค้ด (ถอดรหัสเก็บไว้), ลบลายน้ำสีรายไฟล์,
template ของฟอร์มตายตัว (50 ทวิ) ที่ดูดขอบเข้าหาเส้นจริง, แถบตามเส้นนอนของเอกสารทั่วไป, ตาราง → ช่อง → คู่ key/value และตัด crop ให้เหลือแต่ข้อความ
ลำดับงานอยู่ที่ run.py ค่าตั้งอยู่ที่ configs/regions/
"""
