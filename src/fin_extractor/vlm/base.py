"""ผลลัพธ์ร่วมของตัวอ่านภาพทุกตัวใน ``vlm/`` (Typhoon, OpenThai, OXR)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLMResult:
    engine: str
    markdown: str
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"
