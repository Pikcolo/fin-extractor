"""Client กลางสำหรับเรียก VLM/LLM — รองรับ 2 provider.

- ``ollama`` : Ollama native API (``/api/chat``) — ต้องใช้ตัวนี้เพราะ endpoint แบบ OpenAI ของ Ollama
               ไม่ให้ตั้ง ``num_ctx`` (default 4096 token ไม่พอ: ภาพเอกสาร 1 หน้ากิน ~2-4k token)
- ``openai`` : endpoint แบบ OpenAI-compatible (Typhoon API, vLLM, ฯลฯ)
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import get_secret


@dataclass
class ChatResponse:
    content: str
    finish_reason: str | None  # "stop" | "length" | ...


def encode_image(img: np.ndarray, max_side: int | None = None) -> str:
    """ย่อภาพ (ถ้าใหญ่เกิน) แล้ว encode เป็น base64 PNG — VLM คิด token ตามขนาดภาพ."""
    if max_side and max(img.shape[:2]) > max_side:
        scale = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("encode ภาพไม่ได้")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class ChatClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.provider = cfg.get("provider", "openai")
        self.model = cfg["model"]
        if self.provider == "openai":
            from openai import OpenAI

            # vLLM ไม่ตรวจ key แต่ library บังคับให้มีค่า
            api_key = get_secret(cfg.get("api_key_env")) or "not-needed"
            self._openai = OpenAI(base_url=cfg["base_url"], api_key=api_key, timeout=cfg.get("timeout", 300))
        elif self.provider == "ollama":
            import httpx

            self._http = httpx.Client(base_url=cfg["base_url"], timeout=cfg.get("timeout", 300))
        else:
            raise ValueError(f"unknown provider: {self.provider}")

    def chat(
        self,
        messages: list[dict[str, str]],
        images: list[np.ndarray] | None = None,
        json_mode: bool = False,
    ) -> ChatResponse:
        """``messages`` เป็นข้อความล้วน ({role, content}); ``images`` จะแนบไปกับ message สุดท้าย."""
        images_b64 = [encode_image(im, self.cfg.get("max_image_side")) for im in images or []]
        if self.provider == "ollama":
            return self._chat_ollama(messages, images_b64, json_mode)
        return self._chat_openai(messages, images_b64, json_mode)

    def _chat_ollama(self, messages, images_b64, json_mode) -> ChatResponse:
        msgs = [dict(m) for m in messages]
        if images_b64:
            msgs[-1]["images"] = images_b64
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {
                "temperature": self.cfg.get("temperature", 0.0),
                "num_ctx": self.cfg.get("num_ctx", 16384),
                "num_predict": self.cfg.get("max_tokens", 4096),
                **({"top_p": self.cfg["top_p"]} if "top_p" in self.cfg else {}),
            },
        }
        if json_mode:
            body["format"] = "json"
        resp = self._http.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        return ChatResponse(data["message"]["content"], data.get("done_reason"))

    def _chat_openai(self, messages, images_b64, json_mode) -> ChatResponse:
        msgs: list[dict[str, Any]] = [dict(m) for m in messages]
        if images_b64:
            msgs[-1]["content"] = [
                {"type": "text", "text": msgs[-1]["content"]},
                *({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
                  for b in images_b64),
            ]
        kwargs: dict[str, Any] = {}
        if "top_p" in self.cfg:
            kwargs["top_p"] = self.cfg["top_p"]
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._openai.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.cfg.get("temperature", 0.0),
            max_tokens=self.cfg.get("max_tokens", 4096),
            **kwargs,
        )
        choice = resp.choices[0]
        return ChatResponse(choice.message.content or "", choice.finish_reason)
