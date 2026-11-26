"""ส่งคำถามให้ decision model แล้วแปลงคำตอบเป็น ``Decision`` (schema.py)

ลำดับตัวเลือกมีผลกับคำตอบ (P0 spike: สลับ 3 ลำดับ ชนิดเอกสารพลิก 6/18 ใบ) และแพ็กเกจ openthai-systemone 0.1.0 ไม่มีโหมด order-invariant
ที่ model card พูดถึง คำถาม choice จึงถามทุกลำดับใน ``orders`` (เดิม · กลับด้าน · หมุนครึ่ง กำหนดตายตัว ไม่สุ่ม) แล้วเฉลี่ยความน่าจะเป็น
และบันทึกว่าแต่ละลำดับเลือกเหมือนกันไหม (``agreed``) คำถาม yes/no ไม่มีลำดับให้สลับ ถามครั้งเดียว
backend ที่ล้ม (โหลดโมเดลไม่ได้ ตอบผิดรูป) = คืนรายการว่าง ไม่ทำให้ pipeline ล้ม
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from ..config import resolve_path
from ..schema import Decision

log = logging.getLogger(__name__)


class Backend(Protocol):
    name: str

    def choice(self, state: dict, instructions: str, options: dict[str, str | None]) -> tuple[dict[str, float], float]:
        """(ความน่าจะเป็นของแต่ละตัวเลือก รวมเป็น 1, ความน่าจะเป็นของ abstain)."""

    def nouls(self, state: dict, questions: dict[str, str]) -> dict[str, float]:
        """p(yes) ของหลายคำถามบน state เดียวกัน (โมเดลอ่าน state ครั้งเดียวแล้วตอบทุกคำถามในรอบเดียว)."""


class SystemOneBackend:
    """แพ็กเกจ ``openthai-systemone`` โหลด weights ในเครื่อง (ไม่ส่งข้อมูลออกนอกเครื่อง) โหลดครั้งแรกที่ถูกเรียก."""

    def __init__(self, model: str, device: str = "cpu"):
        self.model, self.device = model, device
        self.name = f"systemone:{resolve_path(model).name}"
        self._client = None

    def _load(self):
        if self._client is None:
            from openthai_systemone import SystemOneClient

            self._client = SystemOneClient(str(resolve_path(self.model)), device=self.device)
        return self._client

    def choice(self, state, instructions, options):
        from openthai_systemone import Choice

        a = self._load().system_one(state, {"q": Choice(instructions=instructions, criteria=options)}).answers["q"]
        return dict(a.probabilities), float(a.abstain)

    def nouls(self, state, questions):
        from openthai_systemone import Noul

        r = self._load().system_one(state, {k: Noul(instructions=t) for k, t in questions.items()})
        return {k: float(r.answers[k].noul) for k in questions}


def orders(n: int) -> list[list[int]]:
    """ลำดับตัวเลือกที่ถาม: เดิม · กลับด้าน · หมุนครึ่ง (ตัดลำดับที่ซ้ำ เช่น n ≤ 2)."""
    base = list(range(n))
    out = [base, base[::-1], base[n // 2:] + base[: n // 2]]
    return [o for i, o in enumerate(out) if o not in out[:i]]


class Decider:
    """ถามคำถามที่ ``questions.build`` สร้าง คืน ``list[Decision]`` · ``backend`` ใส่เองได้ (tests ใช้ตัวปลอม ไม่โหลดโมเดล)."""

    def __init__(self, cfg: dict[str, Any], backend: Backend | None = None):
        self.cfg = cfg
        self.tasks = list(cfg.get("tasks") or [])
        self.backend = backend or SystemOneBackend(cfg.get("model", "models/OpenThai-SystemOne"), cfg.get("device", "cpu"))

    def ask(self, questions: list[dict]) -> list[Decision]:
        """คำถาม yes/no ที่ใช้ state เดียวกันถามรวมในการเรียกครั้งเดียว choice ถามแยกทีละลำดับตัวเลือก คืนตามลำดับคำถามเดิม."""
        out: dict[int, Decision] = {}
        try:
            groups: dict[str, list[int]] = {}
            for i, q in enumerate(questions):
                if q["type"] == "choice":
                    out[i] = self._choice(q)
                else:
                    groups.setdefault(json.dumps(q["state"], ensure_ascii=False, sort_keys=True), []).append(i)
            for idx in groups.values():
                ps = self.backend.nouls(questions[idx[0]]["state"], {str(i): questions[i]["instructions"] for i in idx})
                for i in idx:
                    out[i] = self._noul(questions[i], ps[str(i)])
        except Exception as e:  # noqa: BLE001 — shadow mode: โมเดลใช้ไม่ได้ต้องไม่ทำให้ผลสกัดหาย
            log.warning("decision model failed, no decisions recorded: %s", e)
            return []
        return [out[i] for i in range(len(questions))]

    def _choice(self, q: dict) -> Decision:
        names = list(q["options"])
        runs = []
        for o in orders(len(names)):
            probs, abstain = self.backend.choice(q["state"], q["instructions"], {names[i]: q["options"][names[i]] for i in o})
            runs.append((probs, abstain))
        avg = {k: sum(p.get(k, 0.0) for p, _a in runs) / len(runs) for k in names}
        picks = {max(p, key=p.get) for p, _a in runs}
        best = max(avg, key=avg.get)
        return Decision(task=q["task"], key=q["key"], type="choice", answer=best,
                        probabilities={k: round(v, 4) for k, v in avg.items()},
                        abstain=round(max(a for _p, a in runs), 4), orders=len(runs), agreed=len(picks) == 1,
                        model=self.backend.name, state=q["state"], instructions=q["instructions"])

    def _noul(self, q: dict, p: float) -> Decision:
        # คำตอบ = ตัวเลือกที่ความน่าจะเป็นสูงกว่า (แบบเดียวกับ choice) ไม่ใช่เกณฑ์ที่ตั้ง การใช้ p ตัดสินอะไรต้องรอผลวัด P3
        return Decision(task=q["task"], key=q["key"], type="noul", answer="yes" if p >= 1 - p else "no",
                        probabilities={"yes": round(p, 4), "no": round(1 - p, 4)}, orders=1, agreed=True,
                        model=self.backend.name, state=q["state"], instructions=q["instructions"])
