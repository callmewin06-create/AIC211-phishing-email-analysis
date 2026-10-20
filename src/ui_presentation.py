"""Pure presentation helpers for the Streamlit security-matrix view.

This module deliberately formats only already-computed layer results.  It must
not perform security checks, train models, access feedback storage, or call
external services.
"""

from __future__ import annotations

from html import escape
from typing import Any


SECURITY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Nhóm 1: Kiểm duyệt Kỹ thuật & Hạ tầng mạng (IP, SPF/DKIM/DMARC, MTA Hops)",
        ("1. Network IP Layer", "3. Authentication Protocols", "4. MTA Routing Layer"),
    ),
    (
        "Nhóm 2: Kiểm duyệt Đối tượng & Tên miền (Domain Reputation, Header Metadata, HTML, URL, Attachment Metadata)",
        (
            "2. Domain Reputation",
            "5. Header Metadata Layer",
            "8. HTML & Visual Layer",
            "9. URL & Link Intelligence",
            "10. Attachment Sandboxing",
            "10. Attachment Metadata Risk",
        ),
    ),
    (
        "Nhóm 3: Kiểm duyệt Lõi trí tuệ AI & Ngữ nghĩa (Body Text, Leetspeak, ML estimate, optional LLM review)",
        (
            "6. Body Text Content",
            "7. Adversarial Defense",
            "11. Statistical & ML Layer",
            "12. Deep Semantic AI Layer",
        ),
    ),
)


def group_security_layers(layers: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return the historical three visual groups while preserving row order."""
    return [
        (title, [layer for layer in layers if any(name in str(layer.get("layer", "")) for name in names)])
        for title, names in SECURITY_GROUPS
    ]


def security_status_style(status: object) -> dict[str, str]:
    """Return the legacy row and pill colours for a computed layer status."""
    normalized = str(status or "").lower()
    if "n/a" in normalized:
        return {
            "row_bg": "rgba(139, 148, 158, 0.04)",
            "row_text": "#8b949e",
            "pill_bg": "rgba(139, 148, 158, 0.1)",
            "pill_text": "#8b949e",
            "pill_border": "rgba(139, 148, 158, 0.25)",
        }
    if (
        "cảnh báo" in normalized
        or "risk" in normalized
        or "indicator detected" in normalized
        or "🚨" in str(status)
    ):
        return {
            "row_bg": "rgba(248, 81, 73, 0.06)",
            "row_text": "#c9d1d9",
            "pill_bg": "rgba(248, 81, 73, 0.15)",
            "pill_text": "#ff7b72",
            "pill_border": "rgba(248, 81, 73, 0.4)",
        }
    if any(token in normalized for token in ("unknown", "not supplied", "⚠️")):
        return {
            "row_bg": "rgba(210, 153, 34, 0.05)",
            "row_text": "#c9d1d9",
            "pill_bg": "rgba(210, 153, 34, 0.15)",
            "pill_text": "#d29922",
            "pill_border": "rgba(210, 153, 34, 0.4)",
        }
    if any(token in normalized for token in ("checked", "no indicator", "model", "estimate", "pass")):
        model_status = "model" in normalized or "estimate" in normalized
        return {
            "row_bg": "rgba(88, 166, 255, 0.04)" if model_status else "rgba(46, 160, 67, 0.04)",
            "row_text": "#c9d1d9",
            "pill_bg": "rgba(88, 166, 255, 0.15)" if model_status else "rgba(46, 160, 67, 0.15)",
            "pill_text": "#58a6ff" if model_status else "#56d364",
            "pill_border": "rgba(88, 166, 255, 0.4)" if model_status else "rgba(46, 160, 67, 0.4)",
        }
    if any(
        token in normalized
        for token in (
            "an toàn",
            "kiểm duyệt",
            "đạt tiêu chuẩn",
            "đã xác thực",
            "hợp lệ",
            "sạch sẽ",
            "bình thường",
            "thấu hiểu ngữ nghĩa",
            "xanh",
        )
    ) or "🟢" in str(status):
        return {
            "row_bg": "rgba(46, 160, 67, 0.04)",
            "row_text": "#c9d1d9",
            "pill_bg": "rgba(46, 160, 67, 0.15)",
            "pill_text": "#56d364",
            "pill_border": "rgba(46, 160, 67, 0.4)",
        }
    if any(
        token in normalized
        for token in ("⚠️", "nghi vấn", "đáng ngờ", "bất thường", "thiếu xác thực", "vàng", "chờ", "giải mã")
    ):
        return {
            "row_bg": "rgba(210, 153, 34, 0.05)",
            "row_text": "#c9d1d9",
            "pill_bg": "rgba(210, 153, 34, 0.15)",
            "pill_text": "#d29922",
            "pill_border": "rgba(210, 153, 34, 0.4)",
        }
    return {
        "row_bg": "rgba(110, 118, 129, 0.02)",
        "row_text": "#c9d1d9",
        "pill_bg": "rgba(110, 118, 129, 0.1)",
        "pill_text": "#c9d1d9",
        "pill_border": "rgba(110, 118, 129, 0.2)",
    }


def render_security_layer_group(title: str, layers: list[dict[str, Any]]) -> str:
    """Render one escaped, legacy-styled HTML table from layer result data."""
    if not layers:
        return ""

    rows: list[str] = []
    for layer in layers:
        layer_name = str(layer.get("layer", ""))
        status = str(layer.get("status", ""))
        detail = str(layer.get("detail", ""))
        style = security_status_style(status)
        rows.append(
            f'''<tr style="background-color: {style['row_bg']}; color: {style['row_text']};">
<td style="font-weight: 500;">{escape(layer_name)}</td>
<td><span class="status-pill" style="background-color: {style['pill_bg']}; color: {style['pill_text']}; border-color: {style['pill_border']};">{escape(status)}</span></td>
<td>{escape(detail)}</td>
</tr>'''
        )

    return f'''<div style="margin-top: 15px; margin-bottom: 25px;">
<h5 style="color: #58a6ff; font-weight: bold; margin-bottom: 10px;">{escape(title)}</h5>
<table class="security-table">
<thead><tr>
<th style="width: 30%;">Tầng Bảo Mật (Layer)</th>
<th style="width: 25%;">Trạng Thái Kiểm Duyệt</th>
<th style="width: 45%;">Chi Tiết Kết Quả Phân Tích</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>'''


def render_security_matrix(layers: list[dict[str, Any]]) -> str:
    """Render all non-empty legacy visual groups as one safe HTML fragment."""
    return "".join(
        render_security_layer_group(title, group_layers)
        for title, group_layers in group_security_layers(layers)
        if group_layers
    )
