"""Optional local OCR helpers for image fallback.

The harness does not require OCR dependencies to run. When Pillow and
pytesseract are installed, image blocks can be converted into text before being
sent to a text-only backend; otherwise callers get an empty string and use their
normal degradation message.
"""

from __future__ import annotations

import base64
from io import BytesIO


def extract_text_from_block(block: dict) -> str:
    source = block.get("source") or {}
    if source.get("type") != "base64" or not source.get("data"):
        return ""
    try:
        from PIL import Image
        import pytesseract

        data = base64.b64decode(source["data"])
        image = Image.open(BytesIO(data))
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""
