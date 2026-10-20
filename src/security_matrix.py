"""Evidence-oriented 12-layer email security matrix.

Layers 1-10 are independent technical/behavioural evidence.  Layer 11 is the
ML estimate and layer 12 is the optional Local-LLM estimate.  Keeping those
roles separate prevents the circular claim that "ML + 12 layers" contains ML
and LLM a second time during an ablation study.
"""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, asdict
from email import policy
from email.parser import Parser
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MATRIX_DEFINITIONS = (
    (1, "transport_ip", "Transport/IP evidence", "technical"),
    (2, "sender_domain", "Sender/domain reputation", "technical"),
    (3, "authentication", "SPF/DKIM/DMARC evidence", "technical"),
    (4, "routing", "Message routing anomaly", "technical"),
    (5, "header_consistency", "Header identity consistency", "technical"),
    (6, "social_engineering", "Social-engineering language", "behavioural"),
    (7, "obfuscation", "Obfuscation/adversarial content", "behavioural"),
    (8, "html_deception", "HTML/visual deception", "technical"),
    (9, "url_intelligence", "URL/link intelligence", "technical"),
    (10, "attachment", "Attachment risk", "technical"),
    (11, "ml_detector", "Machine-learning detector", "model"),
    (12, "llm_review", "Local-LLM semantic review", "model"),
)

# Pre-registered engineering weights.  They are fixed before evaluating the
# external datasets.  Unknown layers contribute neither risk nor reassurance.
TECHNICAL_WEIGHTS = {
    "transport_ip": 0.12,
    "sender_domain": 0.06,
    "authentication": 0.13,
    "routing": 0.05,
    "header_consistency": 0.10,
    "social_engineering": 0.14,
    "obfuscation": 0.07,
    "html_deception": 0.08,
    "url_intelligence": 0.17,
    "attachment": 0.08,
}

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
RISKY_TLDS = {"xyz", "top", "click", "shop", "site", "info", "tk", "gq", "ml", "cf"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "buff.ly"}
DANGEROUS_EXTENSIONS = {
    # ``.com`` executables are intentionally omitted because email/domain text
    # makes that suffix unusably ambiguous without MIME attachment metadata.
    "exe", "vbs", "js", "jse", "scr", "bat", "cmd", "ps1", "iso", "img", "lnk", "hta", "docm", "xlsm",
}


@dataclass(frozen=True)
class LayerEvidence:
    layer: int
    key: str
    name: str
    category: str
    state: str
    risk: float | None
    observed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _bounded(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _evidence(number: int, risk: float | None, observed: bool, detail: str) -> LayerEvidence:
    _, key, name, category = MATRIX_DEFINITIONS[number - 1]
    if not observed:
        state = "UNKNOWN"
        risk = None
    elif number >= 11:
        state = "MODEL"
    elif risk and risk > 0:
        state = "RISK"
    else:
        state = "NO_INDICATOR"
    return LayerEvidence(number, key, name, category, state, None if risk is None else _bounded(risk), observed, detail)


def _header_value(message, name: str) -> str:
    return str(message.get(name, "") or "")


def _domain_from_address(value: str) -> str:
    match = re.search(r"@([a-z0-9.-]+)", value, re.I)
    return match.group(1).casefold().rstrip(".") if match else ""


@lru_cache(maxsize=1)
def _load_ip_blacklist() -> set[str]:
    path = ROOT / "data" / "ip_blacklist.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("blacklisted_ips", payload.get("ips", []))
        return {str(item) for item in payload}
    except (OSError, ValueError, TypeError):
        return set()


def extract_technical_evidence(raw_text: str, *, trusted_gateway_auth: bool = False) -> list[LayerEvidence]:
    """Extract layers 1-10 without DNS or live-URL side effects."""

    raw = str(raw_text or "")
    header_like = bool(re.search(r"^(?:from|to|subject|received|authentication-results):", raw, re.I | re.M))
    try:
        message = Parser(policy=policy.default).parsestr(raw) if header_like else None
    except Exception:
        message = None

    # Behavioural and HTML evidence operates on decoded MIME bodies. Raw
    # RFC822 commonly contains long Base64 transfer blocks; that is normal
    # transport encoding and must not be treated as adversarial obfuscation.
    analysis_content = raw
    if message is not None:
        decoded_parts = []
        parts = message.walk() if message.is_multipart() else (message,)
        for part in parts:
            if part.get_content_type() not in {"text/plain", "text/html"}:
                continue
            if part.get_content_disposition() == "attachment":
                continue
            try:
                value = part.get_content()
            except Exception:
                value = ""
            if isinstance(value, str) and value:
                decoded_parts.append(value)
        if decoded_parts:
            analysis_content = "\n".join(decoded_parts)
    analysis_lowered = analysis_content.casefold()

    # 1. Transport IP: a blacklist hit is risk; absence of headers is UNKNOWN.
    received = re.findall(r"^received:\s*(.+(?:\n[ \t].+)*)", raw, re.I | re.M)
    ips = re.findall(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", "\n".join(received))
    public_ips: list[str] = []
    for item in ips:
        try:
            parsed = ipaddress.ip_address(item)
            if parsed.is_global:
                public_ips.append(item)
        except ValueError:
            continue
    blacklist_hits = sorted(set(public_ips) & _load_ip_blacklist())
    layers = [
        _evidence(
            1,
            1.0 if blacklist_hits else 0.0,
            bool(received),
            f"Blacklisted transport IP: {', '.join(blacklist_hits)}" if blacklist_hits else (
                "No listed transport IP observed." if received else "No trusted transport header was supplied."
            ),
        )
    ]

    # 2. A common/risky TLD is only a risk indicator, never proof of identity.
    from_value = _header_value(message, "From") if message else ""
    sender_domain = _domain_from_address(from_value)
    tld = sender_domain.rsplit(".", 1)[-1] if "." in sender_domain else ""
    domain_risk = 0.65 if tld in RISKY_TLDS else 0.0
    layers.append(
        _evidence(
            2,
            domain_risk,
            bool(sender_domain),
            f"Risky sender TLD .{tld}." if domain_risk else (
                f"Sender domain observed: {sender_domain}; this does not prove safety."
                if sender_domain else "Sender domain unavailable."
            ),
        )
    )

    # 3. Only supplied gateway results are interpreted. DNS-key publication is
    # not cryptographic DKIM verification and is deliberately excluded here.
    auth_values: list[str] = []
    if message is not None:
        auth_values.extend(str(value) for value in message.get_all("Authentication-Results", []))
        auth_values.extend(str(value) for value in message.get_all("Received-SPF", []))
    auth_lines = " ".join(auth_values)
    if not auth_lines:
        # Fallback for malformed/pasted headers. Include continuation lines so
        # folded SPF/DKIM/DMARC results are not silently counted as zero.
        auth_lines = " ".join(
            re.findall(
                r"^(?:authentication-results|received-spf):\s*(.+(?:\n[ \t].+)*)",
                raw,
                re.I | re.M,
            )
        )
    auth_observed = bool(auth_lines)
    # An uploaded RFC822 message is attacker-controlled.  Authentication
    # headers become evidence only when a verified mail gateway integration
    # explicitly attests to their provenance; otherwise a forged ``pass`` must
    # not lower the technical risk score.
    if not trusted_gateway_auth:
        auth_observed = False
        auth_lines = ""
    failures = re.findall(r"\b(?:spf|dkim|dmarc)\s*=\s*(?:fail|softfail|temperror|permerror)\b", auth_lines, re.I)
    passes = re.findall(r"\b(?:spf|dkim|dmarc)\s*=\s*pass\b", auth_lines, re.I)
    layers.append(
        _evidence(
            3,
            min(1.0, 0.45 * len(failures)) if auth_observed else None,
            auth_observed,
            f"Gateway-reported authentication failures: {len(failures)}."
            if failures else (
                f"Gateway-reported passes: {len(passes)}; authenticated origin is not proof of benign intent."
                if auth_observed else "No trusted gateway authentication result supplied."
            ),
        )
    )

    # 4. Routing depth is a weak anomaly only.
    hops = len(received)
    route_risk = min(1.0, max(0, hops - 8) / 8.0) if received else None
    layers.append(_evidence(4, route_risk, bool(received), f"Observed {hops} Received hop(s)." if received else "Routing path unavailable."))

    # 5. From/Reply-To mismatch is evidence, not a verdict.
    reply_to = _header_value(message, "Reply-To") if message else ""
    reply_domain = _domain_from_address(reply_to)
    mismatch = bool(sender_domain and reply_domain and sender_domain != reply_domain)
    layers.append(
        _evidence(
            5,
            0.8 if mismatch else 0.0,
            bool(sender_domain),
            f"From domain {sender_domain} differs from Reply-To {reply_domain}."
            if mismatch else (
                "No From/Reply-To domain mismatch observed." if sender_domain else "Header identity fields unavailable."
            ),
        )
    )

    # 6. Behavioural persuasion/credential/payment requests.
    urgency = len(re.findall(r"\b(?:urgent|immediately|within 24 hours|suspend|action required|khẩn cấp|ngay lập tức|bị khóa|xác minh)\b", analysis_lowered))
    credentials = len(re.findall(r"\b(?:password|credential|login|sign[ -]?in|otp|account verification|mật khẩu|đăng nhập|tài khoản)\b", analysis_lowered))
    payment = len(re.findall(r"\b(?:wire transfer|gift card|bank account|crypto|chuyển khoản|thẻ quà|thẻ cào)\b", analysis_lowered))
    social_score = min(1.0, 0.18 * urgency + 0.25 * credentials + 0.35 * payment)
    layers.append(_evidence(6, social_score, True, f"Urgency={urgency}, credential={credentials}, payment={payment} signals."))

    # 7. Obfuscation indicators.
    bidi_control = bool(re.search(r"[\u202a-\u202e]", analysis_content))
    zero_width_count = len(re.findall(r"[\u200b-\u200f]", analysis_content))
    without_zero_width = re.sub(r"[\u200b-\u200f]", "", analysis_lowered)
    zero_width_evasion = bool(
        zero_width_count
        and any(
            term in without_zero_width and term not in analysis_lowered
            for term in ("password", "login", "credential", "verify", "wire transfer")
        )
    )
    separated_word = bool(re.search(r"\b\w(?:[.\-_]\w){3,}\b", analysis_content))
    # Every alternative below requires an actual substitution. Plain words
    # such as login/password/verify belong to semantic evidence, not evasion.
    leetspeak = bool(
        re.search(
            r"\b(?:p@ssword|passw0rd|p@ssw0rd|l0gin|log1n|l0g1n|v3rify|ver1fy|v3r1fy)\b",
            analysis_content,
            re.I,
        )
    )
    obfuscation_hits = sum((bidi_control, zero_width_evasion, separated_word, leetspeak))
    layers.append(
        _evidence(
            7,
            min(1.0, obfuscation_hits * 0.35),
            True,
            (
                f"Obfuscation signals: bidi-control={bidi_control}, zero-width-count={zero_width_count}, "
                f"zero-width-evasion={zero_width_evasion}, "
                f"separated-word={separated_word}, substituted-leetspeak={leetspeak}. "
                "MIME/Base64 transfer encoding is not treated as evasion."
            ),
        )
    )

    # 8. Visual deception in HTML.
    html_observed = bool(re.search(r"<\s*(?:html|body|a|div|form|img)\b", analysis_content, re.I))
    hidden_style_hits = sum(
        bool(re.search(pattern, analysis_lowered, re.I))
        for pattern in (r"display\s*:\s*none", r"font-size\s*:\s*0")
    )
    script_hits = int(bool(re.search(r"<\s*script\b", analysis_lowered, re.I)))
    password_field_hits = int(bool(re.search(r"type\s*=\s*['\"]?password", analysis_lowered, re.I)))
    form_hits = int(bool(re.search(r"<\s*form\b", analysis_lowered, re.I)))
    hidden_sensitive = bool(
        hidden_style_hits
        and re.search(r"\b(?:password|credential|login|verify account|otp|wire transfer)\b", analysis_lowered)
    )
    # Responsive email templates routinely contain display:none/font-size:0.
    # Hidden styling is therefore context only, not a risk signal by itself.
    form_sensitive = bool(form_hits and credentials)
    html_risk = min(
        1.0,
        0.55 * script_hits
        + 0.65 * password_field_hits
        + (0.25 if form_sensitive else 0.0)
        + (0.2 if hidden_sensitive else 0.0),
    )
    layers.append(
        _evidence(
            8,
            html_risk if html_observed else None,
            html_observed,
            (
                f"HTML script={script_hits}, password-field={password_field_hits}, form={form_hits}; "
                f"hidden-style patterns={hidden_style_hits}; hidden sensitive context={hidden_sensitive}. "
                "A form or responsive CSS is not malicious by itself."
                if html_observed
                else "No HTML body supplied."
            ),
        )
    )

    # 9. Offline URL lexical intelligence. Reachability is intentionally not a
    # risk feature because online/offline status does not establish intent.
    urls = URL_RE.findall(analysis_content)
    url_risks = []
    for url in urls:
        candidate = url if re.match(r"https?://", url, re.I) else f"http://{url}"
        try:
            host = (urlparse(candidate).hostname or "").casefold()
        except ValueError:
            continue
        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        score = 0.0
        if host in SHORTENERS:
            score += 0.35
        if tld in RISKY_TLDS:
            score += 0.35
        try:
            if ipaddress.ip_address(host):
                score += 0.55
        except ValueError:
            pass
        if host.count("-") >= 3 or host.count(".") >= 4:
            score += 0.2
        url_risks.append(_bounded(score))
    max_url_risk = max(url_risks, default=0.0)
    layers.append(_evidence(9, max_url_risk, True, f"Found {len(urls)} URL(s); maximum offline lexical risk={max_url_risk:.2f}."))

    # 10. MIME attachment metadata only. This is not a detonation sandbox and
    # no attachment is opened or executed by this project.
    attachment_names = []
    mime_observed = bool(message and (message.get("MIME-Version") or message.get("Content-Type")))
    if message is not None:
        for part in message.walk():
            filename = part.get_filename()
            if filename or part.get_content_disposition() == "attachment":
                attachment_names.append(str(filename or "unnamed_attachment"))
    attachment_extensions = {
        name.rsplit(".", 1)[-1].casefold()
        for name in attachment_names
        if "." in name
    }
    dangerous = sorted(attachment_extensions & DANGEROUS_EXTENSIONS)
    layers.append(
        _evidence(
            10,
            min(1.0, 0.55 + 0.15 * len(dangerous)) if dangerous else (0.0 if mime_observed else None),
            mime_observed,
            (
                f"Dangerous attachment metadata: {', '.join(dangerous)}. No file was executed."
                if dangerous
                else (
                    f"Observed {len(attachment_names)} MIME attachment(s); no dangerous filename extension. "
                    "Metadata check only—no sandbox execution."
                    if mime_observed
                    else "MIME attachment metadata unavailable; no sandbox execution was performed."
                )
            ),
        )
    )
    return layers


def technical_feature_vector(raw_text: str, *, trusted_gateway_auth: bool = False) -> dict[str, float]:
    evidence = extract_technical_evidence(raw_text, trusted_gateway_auth=trusted_gateway_auth)
    result: dict[str, float] = {}
    for item in evidence:
        result[f"{item.key}_risk"] = float(item.risk or 0.0)
        result[f"{item.key}_observed"] = float(item.observed)
    return result


def technical_matrix_score(raw_text: str, *, trusted_gateway_auth: bool = False) -> float:
    """Return a fixed heuristic score over observed layers 1-10."""

    evidence = extract_technical_evidence(raw_text, trusted_gateway_auth=trusted_gateway_auth)
    weighted_risk = 0.0
    observed_weight = 0.0
    for item in evidence:
        weight = TECHNICAL_WEIGHTS[item.key]
        if item.observed:
            observed_weight += weight
            weighted_risk += weight * float(item.risk or 0.0)
    return _bounded(weighted_risk / observed_weight) if observed_weight else 0.0


def url_risk_score(raw_text: str) -> float:
    """Return the layer-9 URL/link intelligence score by itself.

    This is used for ablation so the paper can answer whether URL evidence
    adds measurable value before bundling it into the full 12-layer matrix.
    """

    for item in extract_technical_evidence(raw_text):
        if item.key == "url_intelligence":
            return float(item.risk or 0.0)
    return 0.0


def build_12_layer_matrix(
    raw_text: str,
    ml_probability: float | None = None,
    llm_probability: float | None = None,
    *,
    trusted_gateway_auth: bool = False,
) -> list[dict]:
    layers = extract_technical_evidence(raw_text, trusted_gateway_auth=trusted_gateway_auth)
    layers.append(
        _evidence(
            11,
            ml_probability,
            ml_probability is not None,
            (
                f"ML probability={float(ml_probability):.2%}; this is a model estimate, "
                "not sender authentication."
            )
            if ml_probability is not None else "ML result unavailable.",
        )
    )
    layers.append(
        _evidence(
            12,
            llm_probability,
            llm_probability is not None,
            (
                f"LLM semantic estimate={float(llm_probability):.2%}; independent review is still required."
            )
            if llm_probability is not None else "Local-LLM result unavailable; no value is imputed.",
        )
    )
    return [item.to_dict() for item in layers]


def matrix_feature_names() -> list[str]:
    return [
        name
        for _, key, _, category in MATRIX_DEFINITIONS[:10]
        for name in (f"{key}_risk", f"{key}_observed")
    ]


def batch_technical_features(texts) -> np.ndarray:
    names = matrix_feature_names()
    rows = [technical_feature_vector(str(text or "")) for text in texts]
    return np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
