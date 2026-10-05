"""
struct_agent/vision/encoding.py
══════════════════════════════
Unified multimodal payload serializer for architectural & structural drawings.
Converts images (PNG, JPG, WebP) and PDFs into standardized Base64 Data URLs
compatible with Gemini, Ollama (gemma4:31b, Qwen2-VL), and OpenAI.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List


def encode_image_file_to_data_url(file_path: str) -> str:
    """Read a raster image file and return an RFC 2397 Data URL."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Drawing file not found: {file_path}")

    ext = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(ext, "image/png")

    with open(path, "rb") as f:
        data = f.read()

    b64_str = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def render_pdf_to_data_urls(file_path: str, max_pages: int = 3, scale: float = 2.0) -> List[str]:
    """
    Render PDF drawing sheets to crisp PNG data URLs.
    Uses pypdfium2 if available, with a graceful fallback.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    urls: List[str] = []

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        num_pages = min(len(pdf), max_pages)

        for page_idx in range(num_pages):
            page = pdf[page_idx]
            # scale=2.0 renders at ~144 DPI for high-resolution text & dimension reading
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            urls.append(f"data:image/png;base64,{b64_str}")

        return urls
    except ImportError:
        # Fallback: if pypdfium2 is missing, encode raw PDF bytes as application/pdf (Gemini native)
        with open(path, "rb") as f:
            data = f.read()
        b64_str = base64.b64encode(data).decode("utf-8")
        return [f"data:application/pdf;base64,{b64_str}"]


def encode_attachments(file_paths: List[str]) -> List[str]:
    """
    Convert a list of file paths (images or PDFs) into a flat list of Data URLs.
    """
    data_urls: List[str] = []
    for fp in file_paths:
        if not fp or not os.path.exists(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".pdf":
            data_urls.extend(render_pdf_to_data_urls(fp))
        else:
            data_urls.append(encode_image_file_to_data_url(fp))
    return data_urls


def encode_file_to_data_urls(file_path: str) -> List[str]:
    """Convert a single file path (image or PDF) into a list of Data URLs."""
    return encode_attachments([file_path])


def build_multimodal_message(
    user_prompt: str, attachment_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Construct a normalized multimodal user message dictionary conforming to OpenAI/Ollama/Gemini schemas.
    """
    if not attachment_paths:
        return {"role": "user", "content": user_prompt}

    data_urls = encode_attachments(attachment_paths)
    if not data_urls:
        return {"role": "user", "content": user_prompt}

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_prompt}
    ]
    for url in data_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })

    return {"role": "user", "content": content}
