import re
import html
import quopri
import unicodedata
import socket
import ipaddress
import os
import json
from functools import lru_cache
from email import policy
from email.parser import BytesParser, Parser
from email.utils import getaddresses
from typing import Any
from urllib.parse import urlparse

# DNS is optional.  Installing packages while importing this module makes an
# analysis request mutate the Python environment and is unsafe in production.
dns: Any
try:
    import dns
    import dns.resolver
except ImportError:
    dns = None

MAX_EMAIL_CHARS = 15_000
MAX_RAW_EMAIL_CHARS = 5_000_000
MAX_EML_BYTES = 5_000_000
MAX_MIME_PARTS = 256
MAX_DECODED_BODY_CHARS = 1_000_000
MAX_EXTRACTED_URLS = 200

# Deterministic language routing for Vietnamese and English.  This is used to
# select presentation language and add an ML feature, never as a security
# verdict.  The previous set-based check made a single English word in a
# Vietnamese message look bilingual and had no explainable confidence signal.
VIETNAMESE_LANGUAGE_MARKERS = frozenset({
    "và", "của", "được", "không", "tài", "khoản", "mật", "khẩu", "đăng",
    "nhập", "xác", "minh", "khẩn", "cấp", "thông", "báo", "vui", "lòng",
    "ngân", "hàng", "chuyển", "tiền", "học", "sinh", "trường", "xin",
    "chào", "cảm", "ơn", "quý", "bạn", "người", "gửi", "nhận", "liên",
    "kết", "nhấp", "truy", "cập", "ngay", "số", "điện", "thoại", "dịch",
    "vụ", "hệ", "thống", "đơn", "thanh", "toán", "hóa", "đơn", "lừa",
    "đảo", "bảo", "hiểm",
})
VIETNAMESE_UNACCENTED_PHRASES = (
    "xin chao", "vui long", "xac minh", "tai khoan", "mat khau", "dang nhap",
    "ngan hang", "chuyen khoan", "cam on", "quy khach", "so dien thoai",
    "ma xac thuc", "bao mat tai khoan", "thanh toan ngay", "chung toi",
    "thong bao", "lich lam viec", "ke hoach", "noi bo", "tuan nay", "cap nhat",
    "ho so", "thong tin giao dich", "su dung dich vu", "truc tuyen", "de nghi",
)
ENGLISH_LANGUAGE_MARKERS = frozenset({
    "the", "and", "your", "account", "password", "verify", "security",
    "please", "meeting", "payment", "invoice", "update", "click", "login",
    "from", "with", "this", "that", "for", "you", "are", "hello", "dear",
    "attached", "attachment", "review", "schedule", "project", "team", "thanks",
    "thank", "access", "confirm", "document", "message", "email", "service",
    "immediately", "urgent", "bank", "transfer", "credential", "reset", "notice",
})
# Do not treat generic Latin accents (for example Spanish ``á`` or ``é``) as
# Vietnamese evidence.  The former implementation did exactly that and
# incorrectly routed most SpaPhish (Spanish) messages to ``vi``.  These are
# Vietnamese-only tone forms and letters.  Plain ``ă`` and ``â`` are excluded
# even though they occur in Vietnamese: Romanian also uses them, and they
# caused Spanish emails that happened to mention Romanian locations to be
# falsely routed as Vietnamese.  Vocabulary and multi-word phrases below are
# the complementary channel for ordinary and unaccented Vietnamese messages.
VIETNAMESE_DIAGNOSTIC_CHARS = frozenset(
    "\u0111\u01a1\u01b0\u0110\u01a0\u01af"
    "\u1ea3\u1ea1\u1ea7\u1ea5\u1ea9\u1eab\u1ead"
    "\u1eb1\u1eaf\u1eb3\u1eb5\u1eb7\u1ebb\u1ebd\u1eb9"
    "\u1ec1\u1ebf\u1ec3\u1ec5\u1ec7\u1ec9\u1ecb"
    "\u1ecf\u1ecd\u1ed3\u1ed1\u1ed5\u1ed7\u1ed9"
    "\u1edd\u1edb\u1edf\u1ee1\u1ee3\u1ee7\u1ee5"
    "\u1eeb\u1ee9\u1eed\u1eef\u1ef1\u1ef7\u1ef9\u1ef5"
)
LANGUAGE_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_email_language_details(text: object) -> dict[str, Any]:
    """Explainably route an email as ``vi``, ``en``, ``mixed`` or ``unknown``.

    Vietnamese can be recognized from Vietnamese-specific letters even when a
    phishing campaign uses vocabulary not present in a short keyword list.  A
    second lexical channel keeps common unaccented Vietnamese messages usable.
    The returned score is deliberately descriptive rather than a probability.
    """
    normalized = decode_and_normalize_text(str(text or ""))
    normalized = unicodedata.normalize("NFC", normalized).casefold()
    tokens = LANGUAGE_TOKEN_RE.findall(normalized)
    if not tokens:
        return {
            "language": "unknown", "confidence": 0.0, "vietnamese_score": 0.0,
            "english_score": 0.0, "vietnamese_marker_hits": [],
            "vietnamese_unaccented_phrase_hits": [], "english_marker_hits": [],
            "diacritic_tokens": 0,
            "distinctive_diacritic_tokens": 0,
            "language_model": {
                "state": "NOT_APPLICABLE",
                "top_language": None,
                "top_probability": None,
                "predictions": [],
                "detail": "no tokenized text",
            },
        }

    vietnamese_hits = sorted({token for token in tokens if token in VIETNAMESE_LANGUAGE_MARKERS})
    unaccented_phrase_hits = [
        phrase for phrase in VIETNAMESE_UNACCENTED_PHRASES
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)
    ]
    english_hits = sorted({token for token in tokens if token in ENGLISH_LANGUAGE_MARKERS})
    distinctive_diacritic_tokens = sum(
        bool(set(token) & VIETNAMESE_DIAGNOSTIC_CHARS) for token in tokens
    )
    # Only Vietnamese-specific letters are discriminative.  Cap their impact
    # so a long Vietnamese email cannot drown out a meaningful English portion
    # in a mixed message.
    rule_vietnamese_score = float(
        2 * len(vietnamese_hits)
        + min(8, 2 * distinctive_diacritic_tokens)
        + 2 * len(unaccented_phrase_hits)
    )
    rule_english_score = float(2 * len(english_hits))
    try:
        from src.language_detection import fasttext_language_evidence

        language_model = fasttext_language_evidence(normalized)
    except Exception as error:
        language_model = {
            "state": "UNAVAILABLE",
            "top_language": None,
            "top_probability": None,
            "predictions": [],
            "detail": f"language model unavailable: {error}",
        }
    model_language = language_model.get("top_language")
    model_probability = float(language_model.get("top_probability") or 0.0)
    model_vietnamese_support = 4.0 if model_language == "vi" and model_probability >= 0.90 else 0.0
    model_english_support = 4.0 if model_language == "en" and model_probability >= 0.90 else 0.0
    vietnamese_score = rule_vietnamese_score + model_vietnamese_support
    english_score = rule_english_score + model_english_support

    if vietnamese_score >= 4 and english_score >= 4:
        language = "mixed"
    elif vietnamese_score >= 4:
        language = "vi"
    elif english_score >= 4:
        language = "en"
    else:
        language = "unknown"

    strongest = max(vietnamese_score, english_score)
    weakest = min(vietnamese_score, english_score)
    if language == "mixed":
        confidence = min(1.0, (weakest + strongest) / 16.0)
    elif language == "unknown":
        confidence = min(0.49, strongest / 8.0)
    else:
        confidence = min(1.0, strongest / 12.0)
        if model_probability >= 0.90 and model_language == language:
            confidence = max(confidence, min(0.99, 0.35 + 0.65 * model_probability))
    return {
        "language": language,
        "confidence": round(confidence, 3),
        "vietnamese_score": vietnamese_score,
        "english_score": english_score,
        "vietnamese_rule_score": rule_vietnamese_score,
        "english_rule_score": rule_english_score,
        "model_vietnamese_support": model_vietnamese_support,
        "model_english_support": model_english_support,
        "vietnamese_marker_hits": vietnamese_hits[:12],
        "vietnamese_unaccented_phrase_hits": unaccented_phrase_hits[:8],
        "english_marker_hits": english_hits[:12],
        # Keep the original field for API compatibility, but it now means
        # Vietnamese-specific diacritic tokens rather than every Latin accent.
        "diacritic_tokens": int(distinctive_diacritic_tokens),
        "distinctive_diacritic_tokens": int(distinctive_diacritic_tokens),
        "language_model": language_model,
    }


def detect_email_language(text: object) -> str:
    """Return the stable compact language route used by existing callers."""
    return str(detect_email_language_details(text)["language"])
LLM_PHISHING_LABEL = "[NHÃN: LỪA ĐẢO]"
LLM_SAFE_LABEL = "[NHÃN: AN TOÀN]"


def limit_email_input(text, max_chars=MAX_EMAIL_CHARS):
    """Return a bounded string before MIME parsing or regular-expression work."""
    if not isinstance(text, str):
        return ""
    return text[:max_chars]


def limit_raw_email_input(text, max_chars=MAX_RAW_EMAIL_CHARS):
    """Bound complete RFC822 input without cutting ordinary MIME messages."""
    if not isinstance(text, str):
        return ""
    return text[:max_chars]


def decode_eml_upload(data, max_bytes=MAX_EML_BYTES):
    """Validate an uploaded RFC822/MIME message and decode it without byte loss."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("EML upload must be bytes.")
    raw_bytes = bytes(data)
    if not raw_bytes:
        raise ValueError("File .eml is empty.")
    if len(raw_bytes) > int(max_bytes):
        raise ValueError(f"File .eml exceeds the {int(max_bytes) // 1_000_000} MB safety limit.")
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except Exception as error:
        raise ValueError(f"Invalid RFC822/MIME message: {error}") from error
    if not any(message.get(name) for name in ("From", "To", "Subject", "Date", "Message-ID")):
        raise ValueError("File does not contain recognizable RFC822 email headers.")
    # Latin-1 provides a one-to-one byte mapping.  Unlike UTF-8 with
    # ``errors=replace``, it preserves original 8-bit MIME payload bytes until
    # the declared part charset is applied by ``parse_raw_email``.
    return raw_bytes.decode("latin-1").replace("\x00", "")


def _decode_mime_payload(payload: bytes | None, declared_charset: str | None) -> str:
    """Decode a MIME text part without silently discarding suspicious content."""

    if not payload:
        return ""
    candidates = [str(declared_charset or "").strip(), "utf-8", "windows-1258", "latin-1"]
    attempted: set[str] = set()
    for charset in candidates:
        if not charset or charset.casefold() in attempted:
            continue
        attempted.add(charset.casefold())
        try:
            return payload.decode(charset, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def _coerce_llm_text(content):
    """Convert common LM Studio message shapes to text without guessing labels."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(
            value
            for value in (_coerce_llm_text(item) for item in content)
            if value
        )
    if isinstance(content, dict):
        for key in ("content", "text", "output", "response", "message"):
            if key in content:
                value = _coerce_llm_text(content[key])
                if value:
                    return value
        return ""
    return "" if content is None else str(content)


def _clean_llm_prefix(content):
    """Remove transport/format wrappers while preserving the analyst report."""
    response = _coerce_llm_text(content).replace("\ufeff", "").strip()
    # Reasoning-capable models sometimes leak their private block into the
    # message.  It must never become an apparent classification prefix.
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.I | re.S).strip()
    if re.search(r"<think>", response, flags=re.I) and not re.search(r"</think>", response, flags=re.I):
        response = re.split(r"<think>", response, maxsplit=1, flags=re.I)[0].strip()

    lines = response.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and lines[0].strip().lower() in {"```", "```json", "```text"}:
        lines.pop(0)
    # These wrappers are generated by some OpenAI-compatible adapters.  Do
    # not strip arbitrary prose: a buried label must remain UNKNOWN.
    while lines and re.match(r"^\s*(?:assistant|final answer|answer|output)\s*:\s*$", lines[0], re.I):
        lines.pop(0)
    response = "\n".join(lines).strip()
    if response.endswith("```"):
        response = response[:-3].rstrip()
    return response


def _llm_label_value(value):
    # Đ/đ does not decompose under NFKD, so transliterate it before removing
    # accents (otherwise ``LỪA ĐẢO`` becomes ``lua ao`` and is rejected).
    value = str(value or "").replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", value).casefold()
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    if any(token in ascii_value for token in ("non_phish", "not phishing", "safe", "benign", "legitimate", "an toan")):
        return "SAFE"
    if any(token in ascii_value for token in ("phish", "lua dao", "credential theft")):
        return "PHISHING"
    return "UNKNOWN"


def extract_llm_label(content):
    """Accept explicit leading labels from text/JSON wrappers only.

    The old implementation required one exact, Vietnamese label byte-for-byte.
    That rejected valid outputs wrapped in Markdown, JSON, or ``LABEL:`` by
    different local models.  This parser remains conservative: a label buried
    after prose is still UNKNOWN and can never silently become SAFE.
    """
    response = _clean_llm_prefix(content)
    if not response:
        return "UNKNOWN"

    # Structured output is common even when response_format was not honoured.
    if response.startswith("{"):
        try:
            parsed = json.loads(response)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            for key in ("label", "verdict", "classification", "decision", "category"):
                if key in parsed:
                    label = _llm_label_value(parsed[key])
                    if label != "UNKNOWN":
                        return label

    # Explicit bracket labels, including the historical NHÃN spelling and
    # English LABEL/CLASSIFICATION variants.
    bracket = re.match(r"^\s*[*_`#>\-]*\s*\[[^:\]]+\s*:\s*([^\]]+)\]", response, re.I)
    if bracket:
        return _llm_label_value(bracket.group(1))

    first_line = response.splitlines()[0].strip(" *_`#>-\t")
    explicit = re.match(r"^(?:label|verdict|classification|decision)\s*:\s*(.+)$", first_line, re.I)
    if explicit:
        return _llm_label_value(explicit.group(1))
    if re.fullmatch(r"(?:phishing|safe|benign|legitimate|non[-_ ]?phishing)", first_line, re.I):
        return _llm_label_value(first_line)
    return "UNKNOWN"


def escape_csv_formula(value):
    """Make untrusted text safe when a CSV is opened by spreadsheet software."""
    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value

def extract_structural_features(text):
    """
    Trích xuất các đặc trưng cấu trúc đã được CHUẨN HÓA THEO TỈ LỆ (Normalized Ratios).
    Giải quyết triệt để mối lo ngại của thầy giáo về việc email dài bị tích điểm spam vô hạn:
    - Thay vì đếm tổng số ký tự thô, hệ thống tính TỈ LỆ xuất hiện trên tổng số ký tự/từ.
    - Dấu hai chấm (:), dấu chấm cảm (!), chữ viết hoa... đều được chia cho tổng độ dài email.
    """
    if not isinstance(text, str) or len(text) == 0:
        return {
            'colon_ratio': 0.0,
            'exclamation_ratio': 0.0,
            'uppercase_ratio': 0.0,
            'char_count': 0,
            'word_count': 0
        }
    
    total_chars = len(text)
    total_words = len(text.split()) if text.split() else 1
    
    colon_count = text.count(':')
    exclamation_count = text.count('!')
    uppercase_count = sum(1 for c in text if c.isupper())
    
    return {
        'colon_ratio': colon_count / total_words,       # Tỉ lệ dấu hai chấm trên số từ
        'exclamation_ratio': exclamation_count / total_words, # Tỉ lệ dấu chấm cảm trên số từ
        'uppercase_ratio': uppercase_count / total_chars,     # Tỉ lệ chữ hoa trên tổng số ký tự
        'char_count': total_chars,
        'word_count': total_words
    }

def decode_and_normalize_text(text):
    """
    Giải mã triệt để các dạng mã hóa văn bản phổ biến trong email và chuẩn hóa Unicode:
    1. Quoted-printable decoding (=E1=BA=A1 -> ạ)
    2. Unicode escape sequences (\\u1ec7 -> ệ)
    3. HTML entities (&nbsp;, &#39; -> ')
    4. NFC Unicode normalization (ghép các ký tự tổ hợp NFD về NFC chuẩn)
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # 1. Quoted-printable decoding nếu có dấu = định dạng byte hex
    if '=E' in text or '=C' in text or '=3D' in text or '=20' in text:
        try:
            text = quopri.decodestring(text.encode('utf-8', errors='ignore')).decode('utf-8', errors='ignore')
        except Exception:
            pass
            
    # 2. Giải mã Unicode Escape (như \\u1ec7, \\u00e1...)
    if '\\u' in text or '\\U' in text:
        try:
            text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
            text = re.sub(r'\\U([0-9a-fA-F]{8})', lambda m: chr(int(m.group(1), 16)), text)
        except Exception:
            pass
            
    # 3. Giải mã HTML Entities (&amp;, &lt;, &#123;...)
    text = html.unescape(text)
    
    # 4. Chuẩn hóa NFC Unicode (chống lỗi mất dấu do NFD)
    text = unicodedata.normalize('NFC', text)
    
    return text

def advanced_cleaning_defense(text):
    """
    Hàm làm sạch văn bản nâng cao và chống các kỹ thuật lách luật của Hacker (Adversarial Defense).
    - Hỗ trợ đầy đủ tiếng Việt Unicode có dấu chuẩn NFC.
    - Giải mã Unicode Escape, Quoted-Printable và HTML Entities trước khi lọc regex.
    - Loại bỏ thẻ HTML, khối CSS <style>, <script> và liên kết URL.
    - Giải mã Leetspeak từ khóa cụ thể (không biến đổi số bừa bãi làm hỏng ngày tháng, mã chứng khoán).
    """
    if not isinstance(text, str):
        return ""
    
    text = limit_email_input(text)
    
    # 0. Giải mã và chuẩn hóa Unicode trước
    text = decode_and_normalize_text(text)
    
    # 1. Chuyển về chữ thường
    text = text.lower()
    
    # 2. Loại bỏ các khối CSS <style> và JavaScript <script> trước khi xóa thẻ HTML
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Loại bỏ thẻ HTML
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # 4. Loại bỏ các liên kết URL (hacker thường chèn link để dẫn dụ nạn nhân)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 5. Giải mã Leetspeak cho các từ khóa lừa đảo cụ thể (không thay đổi số toàn cục)
    leetspeak_words = {
        'v1agra': 'viagra',
        'v1@gra': 'viagra',
        'w@tches': 'watches',
        'p@ssword': 'password',
        'f.r.e.e': 'free',
        'r-e-p-l-i-c-a': 'replica'
    }
    for word, replacement in leetspeak_words.items():
        text = text.replace(word, replacement)
    
    # 6. Xóa các dấu chấm, gạch ngang cố tình chèn vào giữa chữ (f.r.e.e -> free)
    text = re.sub(r'(?<=\w)[.\-_](?=\w)', '', text)
    
    # 7. Loại bỏ ký tự đặc biệt nhưng GIỮ LẠI TIẾNG VIỆT UNICODE VÀ CHỮ SỐ
    text = re.sub(r'[^a-z0-9àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]', ' ', text)
    
    # 8. Thay thế nhiều dấu cách bằng một dấu cách duy nhất
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def build_ml_document(text):
    """Build a robust local feature document without exposing raw URLs as text.

    The lexical portion uses the existing de-obfuscation cleaner.  Bounded
    structural tokens preserve security signals that cleaning would otherwise
    discard (URLs, IP literals, urgency/credential requests and language).
    This makes the same input representation available during training and
    live inference.
    """
    raw_text = limit_email_input(str(text or ""))
    cleaned = advanced_cleaning_defense(raw_text)
    lowered = decode_and_normalize_text(raw_text).lower()
    language_details = detect_email_language_details(raw_text)
    language = language_details["language"]
    urls = re.findall(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+", lowered)
    ip_urls = sum(bool(re.search(r"https?://\d{1,3}(?:\.\d{1,3}){3}", url)) for url in urls)
    feature_tokens = [f"lang_{language}"]
    if float(language_details["confidence"]) >= 0.75:
        feature_tokens.append(f"lang_{language}_confident")
    if urls:
        feature_tokens.extend(["url_present", f"url_count_{min(len(urls), 4)}"])
    if ip_urls:
        feature_tokens.append("url_ip_literal")
    if any(re.search(r"\.(?:xyz|top|click|shop|site|info|tk|gq|ml|cf)(?:[/:?#]|$)", url) for url in urls):
        feature_tokens.append("url_risky_tld")
    if re.search(r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl)\b", lowered):
        feature_tokens.append("url_shortener")
    if re.search(
        r"\b(?:login|sign[ -]?in|password|verify|credential|banking|otp|mật\s*khẩu|đăng\s*nhập|xác\s*minh|tài\s*khoản|ngân\s*hàng)\b",
        lowered,
    ):
        feature_tokens.append("credential_request")
    if re.search(r"\b(?:urgent|immediately|suspend(?:ed)?|khẩn\s*cấp|ngay\s*lập\s*tức|bị\s*khóa)\b", lowered):
        feature_tokens.append("urgency_signal")
    if re.search(r"\b(?:wire\s*transfer|gift\s*card|crypto|chuyển\s*khoản|thẻ\s*cào|tiền\s*điện\s*tử)\b", lowered):
        feature_tokens.append("payment_request")
    if re.search(r"\b(?:\.exe|\.js|\.vbs|\.scr|\.iso|\.img)\b", lowered):
        feature_tokens.append("risky_attachment_name")
    return " ".join(part for part in (cleaned, " ".join(feature_tokens)) if part).strip()


def extract_email_addresses(raw_content):
    """Extract RFC822 address fields as normalized email addresses.

    This does not claim that a visible From address is authentic.  It only
    reports the addresses present in the supplied message for analysts to
    compare with Reply-To, Return-Path and authentication evidence.
    """
    raw_content = limit_raw_email_input(raw_content)
    fields = ("from", "to", "cc", "reply-to", "return-path", "sender")
    result = {field.replace("-", "_"): [] for field in fields}
    if not isinstance(raw_content, str) or not raw_content.strip():
        result["all"] = []
        return result
    try:
        message = Parser(policy=policy.default).parsestr(raw_content)
        for field in fields:
            values = message.get_all(field, [])
            extracted = [address.strip().lower() for _name, address in getaddresses(values) if address.strip()]
            result[field.replace("-", "_")] = list(dict.fromkeys(extracted))
    except Exception:
        for field in fields:
            match = re.search(rf"^{re.escape(field)}:\s*([^\r\n]+)", raw_content, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", match.group(1), re.IGNORECASE)
                result[field.replace("-", "_")] = list(dict.fromkeys(item.lower() for item in extracted))
    result["all"] = list(dict.fromkeys(address for values in result.values() for address in values))
    return result


def extract_authentication_evidence(raw_content, allow_dns=False):
    """Retrieve SPF, DKIM and DMARC evidence without trusting pasted headers.

    ``header_status`` is evidence reported by an upstream gateway and can be
    spoofed in a user-supplied EML. ``dns_record_published`` only says a public
    record exists; it is not cryptographic DKIM verification.  The distinction
    is returned explicitly so the UI and LLM do not overstate certainty.
    """
    raw_content = limit_email_input(raw_content)
    raw_lower = raw_content.lower() if isinstance(raw_content, str) else ""
    header_blob = re.sub(r"\r?\n[ \t]+", " ", raw_content or "")
    statuses = {}
    for protocol in ("spf", "dkim", "dmarc"):
        match = re.search(rf"\b{protocol}=([a-z_]+)", raw_lower)
        statuses[protocol] = match.group(1) if match else "not_reported"
    dkim_match = re.search(r"dkim-signature:\s*([^\n]*(?:\n[ \t]+[^\n]*)*)", raw_content or "", re.IGNORECASE)
    signature = {
        "present": bool(dkim_match),
        "domain": None,
        "selector": None,
        "algorithm": None,
        "header_status": statuses["dkim"],
        "dns_record_published": None,
        "verification": "not_independently_verified",
    }
    if dkim_match:
        value = re.sub(r"\s+", " ", dkim_match.group(1))
        for key, name in (("d", "domain"), ("s", "selector"), ("a", "algorithm")):
            match = re.search(rf"(?:^|;)\s*{key}=([^;\s]+)", value, re.IGNORECASE)
            if match:
                signature[name] = match.group(1).strip().lower()
    if allow_dns and signature["domain"] and signature["selector"]:
        records = perform_dns_lookup(f"{signature['selector']}._domainkey.{signature['domain']}", "TXT")
        signature["dns_record_published"] = bool(records)
    arc_headers = re.findall(
        r"^arc-(?:seal|message-signature):\s*([^\r\n]+)",
        raw_content or "",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    lowered_content = (raw_content or "").casefold()
    smime_present = (
        "application/pkcs7-signature" in lowered_content
        or "application/x-pkcs7-signature" in lowered_content
        or "multipart/signed" in lowered_content and "smime-type=signed-data" in lowered_content
    )
    openpgp_present = "-----begin pgp signed message-----" in lowered_content or "application/pgp-signature" in lowered_content
    return {
        "header_status": statuses,
        "dkim_signature": signature,
        "digital_signatures": {
            "dkim": signature,
            "arc": {
                "present": bool(arc_headers),
                "header_count": len(arc_headers),
                "verification": "not_independently_verified",
            },
            "smime": {
                "present": smime_present,
                "verification": "not_independently_verified",
            },
            "openpgp": {
                "present": openpgp_present,
                "verification": "not_independently_verified",
            },
        },
        "authentication_headers_present": any(
            token in header_blob.lower()
            for token in ("authentication-results:", "received-spf:", "dkim-signature:")
        ),
    }


def _normalise_domain(value: object) -> str | None:
    """Return a canonical hostname for IOC reporting without performing DNS."""
    candidate = str(value or "").strip().strip(".[]<>(){}\"'").casefold()
    if not candidate or len(candidate) > 253 or " " in candidate or "@" in candidate:
        return None
    try:
        # IDNA conversion gives analysts one stable value for a Unicode/punycode
        # domain while preserving no implied trust or reachability claim.
        normalized = candidate.encode("idna").decode("ascii").casefold()
    except UnicodeError:
        return None
    labels = normalized.split(".")
    if len(labels) < 2 or any(not label or len(label) > 63 for label in labels):
        return None
    if any(not re.fullmatch(r"[a-z0-9-]+", label) or label.startswith("-") or label.endswith("-") for label in labels):
        return None
    return normalized


def _url_host(url: str) -> str | None:
    candidate = str(url or "").strip().rstrip(".,;:!?)]}\"'")
    if not candidate:
        return None
    if not re.match(r"^[a-z][a-z0-9+.-]*://", candidate, re.IGNORECASE):
        candidate = "http://" + candidate
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    return _normalise_domain(parsed.hostname) if parsed.hostname else None


def extract_iocs(raw_content: object, allow_dns: bool = False, max_domains: int = 20) -> dict[str, Any]:
    """Extract a bounded forensic IOC profile from an email without trusting it.

    DNS lookups are opt-in because resolving sender/link domains leaks metadata
    outside an offline deployment.  Returned DNS records only establish that a
    record was observed; they are never a phishing/safe verdict.
    """
    content = limit_raw_email_input(raw_content)
    addresses = extract_email_addresses(content)
    urls = re.findall(r"(?:https?://|www\.)[^\s<>\"']+", content, flags=re.IGNORECASE)
    urls = list(dict.fromkeys(url.rstrip(".,;:!?)]}\"'") for url in urls if url.strip()))[:MAX_EXTRACTED_URLS]
    domains: list[str] = []

    def add_domain(value: object) -> None:
        domain = _normalise_domain(value)
        if domain and domain not in domains and len(domains) < max(1, int(max_domains)):
            domains.append(domain)

    for address in addresses.get("all", []):
        if "@" in address:
            add_domain(address.rsplit("@", 1)[1])
    for url in urls:
        host = _url_host(url)
        if host:
            domains.append(host) if host not in domains and len(domains) < max(1, int(max_domains)) else None
    # Header domains add useful routing IOCs without treating arbitrary prose
    # such as a company name as a domain candidate.
    for header_domain in re.findall(
        r"^(?:message-id|return-path|received|dkim-signature):[^\r\n]*?\b([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+)\b",
        content,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        add_domain(header_domain)

    ip_addresses = list(dict.fromkeys(extract_ip_candidates(content)))
    originating_ip = extract_originating_ip(content, direction="backward")
    dns_records: dict[str, dict[str, list[str]]] = {}
    if allow_dns:
        for domain in domains:
            dns_records[domain] = {
                "A": perform_dns_lookup(domain, "A")[:5],
                "AAAA": perform_dns_lookup(domain, "AAAA")[:5],
                "MX": perform_dns_lookup(domain, "MX")[:5],
                "TXT": perform_dns_lookup(domain, "TXT")[:5],
            }
    authentication = extract_authentication_evidence(content, allow_dns=allow_dns)
    return {
        "addresses": addresses,
        "urls": urls,
        "domains": domains,
        "ip_addresses": ip_addresses,
        "originating_ip": originating_ip,
        "dns": {
            "lookup_performed": bool(allow_dns),
            "records": dns_records,
            "state": "observed" if allow_dns else "not_queried_offline_privacy_mode",
        },
        "digital_signatures": authentication["digital_signatures"],
        "authentication": authentication,
    }

def strip_mail_transport_headers(text):
    """
    Loại bỏ các dòng Transport Headers hạ tầng mạng (Received, Delivered-To, DKIM, ARC...)
    để lọc ra đúng nội dung chính của email.
    """
    if not isinstance(text, str):
        return ""
    
    lines = text.splitlines()
    cleaned_lines = []
    in_header_block = True
    
    header_prefixes = (
        'delivered-to:', 'received:', 'x-received:', 'arc-seal:', 
        'arc-message-signature:', 'arc-authentication-results:', 
        'dkim-signature:', 'x-google-smtp-source:', 'mime-version:',
        'content-type:', 'message-id:', 'return-path:', 'x-gm-message-state:',
        'authentication-results:', 'received-spf:', 'x-failed-recipients:'
    )
    
    for line in lines:
        stripped_lower = line.strip().lower()
        if in_header_block:
            if any(stripped_lower.startswith(pref) for pref in header_prefixes) or stripped_lower.startswith('by ') or stripped_lower.startswith('with smtp'):
                continue
            if not stripped_lower:
                in_header_block = False
                continue
            if any(stripped_lower.startswith(p) for p in ('from:', 'to:', 'subject:', 'date:')):
                continue
            in_header_block = False
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
            
    result = "\n".join(cleaned_lines).strip()
    return result if result else text

def parse_raw_email(raw_content):
    """
    Phân tích email hoàn chỉnh đa chiến lược:
    1. Standard RFC822 MIME headers (From:, Subject:)
    2. Webmail UI Copy (Gmail, Outlook, Yahoo) với tự động bóc tách Subject, Sender, loại bỏ rác UI (nhãn, nút tóm tắt, nút trả lời).
    3. Partial Copy & Fallback linh hoạt khi người dùng không copy hết toàn bộ.
    """
    from email import policy
    from email.parser import Parser

    raw_content = limit_raw_email_input(raw_content)

    if not isinstance(raw_content, str):
        return {
            "from": "", "subject": "", "body": "", "links": [], "clean_text": "",
            "detection_type": "Không xác định", "language": "unknown",
            "language_details": detect_email_language_details(""), "addresses": {}, "iocs": {},
        }
    
    sender = ""
    subject = ""
    body = ""
    links = []
    attachments = []
    detection_type = "Đoạn văn bản tự do (Plain Text)"
    
    # ------------------- CHIẾN LƯỢC 1: STANDARD RFC822 / MIME HEADERS -------------------
    # Dùng raw_content gốc chưa decode để Parser chạy chính xác cấu trúc MIME.
    # Trích xuất các dòng Header (trước dòng trống đầu tiên phân tách Header và Body) lên đến 150 dòng
    header_lines = []
    for line in raw_content.splitlines():
        if not line.strip():
            break
        header_lines.append(line)
        if len(header_lines) >= 150:
            break
            
    has_received_header = any(line.lower().strip().startswith(('received:', 'message-id:', 'return-path:', 'dkim-signature:', 'authentication-results:')) for line in header_lines)
    has_from_or_subject = any(line.lower().strip().startswith(('from:', 'subject:', 'to:', 'content-type:')) for line in header_lines)
    has_other_mime_header = any(line.lower().strip().startswith(('mime-version:', 'delivered-to:', 'content-transfer-encoding:')) for line in header_lines)
    
    has_mime_headers = has_from_or_subject and (has_received_header or has_other_mime_header)
    
    if has_mime_headers:
        try:
            # Sử dụng raw_content nguyên bản cho EML Parser
            msg = Parser(policy=policy.default).parsestr(raw_content)
            sender = str(msg.get('from', '') or '')
            subject = str(msg.get('subject', '') or '')
            
            if sender:
                sender = decode_and_normalize_text(sender)
            if subject:
                subject = decode_and_normalize_text(subject)

            plain_body = ""
            html_body = ""
            
            if msg.is_multipart():
                for part_index, part in enumerate(msg.walk()):
                    if part_index >= MAX_MIME_PARTS:
                        attachments.append({
                            "filename": "[MIME part limit reached]",
                            "content_type": "application/x-aic211-limit",
                        })
                        break
                    content_type = part.get_content_type()
                    filename = part.get_filename()
                    disposition = part.get_content_disposition()
                    if filename or disposition == "attachment":
                        attachments.append({
                            "filename": decode_and_normalize_text(filename or "unnamed_attachment"),
                            "content_type": content_type,
                        })
                    if content_type not in ['text/plain', 'text/html']:
                        continue
                    try:
                        # Tự động giải mã Base64 hoặc Quoted-Printable về bytes
                        payload_bytes = part.get_payload(decode=True)
                        if payload_bytes:
                            c = _decode_mime_payload(payload_bytes, part.get_content_charset())
                            if content_type == 'text/plain':
                                plain_body = (plain_body + c + "\n")[:MAX_DECODED_BODY_CHARS]
                            elif content_type == 'text/html':
                                html_body = (html_body + c + "\n")[:MAX_DECODED_BODY_CHARS]
                    except Exception as error:
                        # Retain an explicit forensic marker instead of hiding a
                        # decode failure that could be attacker-induced.
                        plain_body += f"\n[MIME text decode failed: {type(error).__name__}]\n"
            else:
                content_type = msg.get_content_type()
                filename = msg.get_filename()
                if filename or msg.get_content_disposition() == "attachment":
                    attachments.append({
                        "filename": decode_and_normalize_text(filename or "unnamed_attachment"),
                        "content_type": content_type,
                    })
                try:
                    payload_bytes = msg.get_payload(decode=True)
                    if payload_bytes:
                        c = _decode_mime_payload(payload_bytes, msg.get_content_charset())
                        if content_type == 'text/html':
                            html_body = c
                        else:
                            plain_body = c
                except Exception:
                    plain_body = raw_content

            if html_body.strip():
                clean_h = re.sub(r'<style[^>]*>.*?</style>', ' ', html_body, flags=re.DOTALL | re.IGNORECASE)
                clean_h = re.sub(r'<script[^>]*>.*?</script>', ' ', clean_h, flags=re.DOTALL | re.IGNORECASE)
                html_visible = re.sub(r'<[^>]*>', ' ', clean_h).strip()
            else:
                html_visible = ""
            # multipart/alternative may contain a benign plain text decoy and
            # a malicious HTML view.  Preserve both representations for
            # inference rather than choosing one and silently dropping the
            # other.
            body = "\n".join(value for value in (plain_body.strip(), html_visible) if value).strip()
                
            if sender or subject:
                detection_type = "Chuẩn EML / RFC822 Header"
                # Trích xuất URLs từ nội dung đã giải mã (tránh lỗi bọc dòng Quoted-Printable)
                combined_bodies = plain_body + "\n" + html_body
                links = re.findall(r'https?://[^\s<>\'"]+|www\.[^\s<>\'"]+', combined_bodies)[:MAX_EXTRACTED_URLS]
        except Exception:
            pass

    # ------------------- CHIẾN LƯỢC 2: WEBMAIL UI COPY (GMAIL / OUTLOOK WEB) -------------------
    # Nếu không phải RFC822 hoặc thiếu cả sender và subject (Parser thất bại), kích hoạt bộ bóc tách Webmail UI
    if not sender or not subject:
        # Ở đây ta mới decode_and_normalize_text toàn bộ raw_content để xử lý giao diện text
        normalized_content = decode_and_normalize_text(raw_content)
        lines = normalized_content.splitlines()
        non_empty_lines = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]
        
        subject_idx = -1
        # 2a. Trích xuất bằng các tiền tố rõ ràng (Vietnamese / English Headers)
        for idx, line_str in non_empty_lines[:15]:
            lower_line = line_str.lower()
            if not sender and (lower_line.startswith('người gửi:') or lower_line.startswith('từ:') or lower_line.startswith('from:')):
                sender = line_str.split(':', 1)[1].strip()
                sender_idx = idx
                detection_type = "Giao diện Webmail (Manual Prefix)"
            elif not subject and (lower_line.startswith('tiêu đề:') or lower_line.startswith('chủ đề:') or lower_line.startswith('subject:')):
                subject = line_str.split(':', 1)[1].strip()
                subject_idx = idx
                detection_type = "Giao diện Webmail (Manual Prefix)"

        # 2b. Nếu vẫn chưa có Subject, trích xuất từ dòng đầu tiên không phải nút tính năng
        if not subject:
            for idx, line_str in non_empty_lines[:5]:
                if line_str.startswith("✦") or "tóm tắt email" in line_str.lower():
                    continue
                # Làm sạch các nhãn dán ở cuối tiêu đề như "Hộp thư đến x", "Inbox", "Thư nháp"
                clean_sub = re.sub(r'\s*(Hộp thư đến|Inbox|Thư nháp|Drafts|Spam|Thư rác)\s*x?\s*$', '', line_str, flags=re.IGNORECASE).strip()
                if clean_sub:
                    subject = clean_sub
                    subject_idx = idx
                    if detection_type == "Đoạn văn bản tự do (Plain Text)":
                        detection_type = "Giao diện Webmail (Gmail/Outlook Copy)"
                    break
                
        sender_idx = -1
        # 2c. Trích xuất Sender từ 10 dòng đầu tiên (nếu vẫn chưa có)
        if not sender:
            for idx, line_str in non_empty_lines[:10]:
                if idx == subject_idx:
                    continue
                # Tìm email trong dạng <email@domain> hoặc standalone email
                email_match = re.search(r'([^\n<]*?<[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}>|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', line_str)
                if email_match:
                    sender = email_match.group(0).strip()
                    sender_idx = idx
                    if detection_type == "Đoạn văn bản tự do (Plain Text)":
                        detection_type = "Giao diện Webmail / Trích xuất Người gửi"
                    break

        # 2d. Xây dựng phần thân bài (Body) bằng cách lọc bỏ các dòng Header và rác giao diện UI
        body_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            stripped_lower = stripped.lower()
            
            # Bỏ qua dòng Subject và Sender đã trích xuất
            if i == subject_idx or i == sender_idx:
                continue
                
            # Bỏ qua nút tóm tắt AI Gemini/Gmail
            if stripped.startswith("✦") or "tóm tắt email này" in stripped_lower or "summarize this email" in stripped_lower:
                continue
            # Bỏ qua biểu tượng icon Gmail UI
            if re.match(r'^[⭐😀↩️⋮\s]+$', stripped):
                continue
            # Bỏ qua dòng người nhận (vd: "đến Thu", "to me")
            if re.match(r'^(đến|to)\s+.*$', stripped_lower):
                continue
            # Bỏ qua nhãn thư rác độc lập
            if re.match(r'^\s*(hộp thư đến|inbox|thư nháp|drafts)\s*x?\s*$', stripped_lower):
                continue
            # Bỏ qua các nút bấm hành động ở cuối email
            if re.match(r'^\s*↩️?\s*(trả lời|trả lời tất cả|chuyển tiếp|reply|reply all|forward)\s*$', stripped_lower):
                continue
            # Bỏ qua mốc thời gian phụ sát cạnh người gửi (vd: "19:23 5 thg 6, 2026")
            if sender_idx != -1 and abs(i - sender_idx) <= 2 and re.match(r'^\d{1,2}:\d{2}.*?\d{4}$', stripped):
                continue

            body_lines.append(line)
            
        body = "\n".join(body_lines).strip()

    # ------------------- CHIẾN LƯỢC 3: FLEXIBLE FALLBACK -------------------
    if not sender:
        body = raw_content.strip()
        subject = ""
        detection_type = "Đoạn văn bản tự do (Plain Text)"
    elif not body and not subject:
        body = raw_content.strip()
        detection_type = "Đoạn văn bản tự do (Plain Text)"

    body = decode_and_normalize_text(body)
    body = strip_mail_transport_headers(body)
    body = re.sub(r'\n\s*\n', '\n\n', body)

    if not links:
        links = re.findall(r'https?://[^\s<>\'"]+|www\.[^\s<>\'"]+', body)[:MAX_EXTRACTED_URLS]
    clean_text = f"Subject: {subject}\n\n{body}" if subject else body
    
    dkim_selector = None
    dkim_domain = None
    dkim_match = re.search(r'dkim-signature:\s*([^\n]*(?:\n[ \t]+[^\n]*)*)', raw_content, re.IGNORECASE)
    if dkim_match:
        dkim_header_content = dkim_match.group(1)
        s_match = re.search(r'\bs=([a-zA-Z0-9\-]+)', dkim_header_content)
        d_match = re.search(r'\bd=([a-zA-Z0-9\-\.]+)', dkim_header_content)
        if s_match:
            dkim_selector = s_match.group(1)
        if d_match:
            dkim_domain = d_match.group(1)
            
    language_details = detect_email_language_details(clean_text)
    addresses = extract_email_addresses(raw_content)
    authentication = extract_authentication_evidence(raw_content, allow_dns=False)
    iocs = extract_iocs(raw_content, allow_dns=False)
    return {
        "from": decode_and_normalize_text(sender),
        "subject": decode_and_normalize_text(subject),
        "body": body,
        "links": sorted(set(links))[:MAX_EXTRACTED_URLS],
        "attachments": attachments,
        "clean_text": decode_and_normalize_text(clean_text),
        "detection_type": detection_type,
        "dkim_selector": dkim_selector,
        "dkim_domain": dkim_domain,
        "language": language_details["language"],
        "language_details": language_details,
        "addresses": addresses,
        "iocs": iocs,
        "authentication": authentication,
        "digital_signature": authentication["dkim_signature"],
    }


OFFICIAL_DOMAINS = [
    'google.com', 'gmail.com', 'shopee.vn', 'vietcombank.com.vn', 
    'hust.edu.vn', 'microsoft.com', 'outlook.com', 'facebook.com', 
    'paypal.com', 'github.com', 'apple.com', 'netflix.com', 
    'linkedin.com', 'amazon.com', 'youtube.com',
    'google.com.vn', 'google.vn', 'microsoft.com.vn', 'shopee.com', 'shopee.com.vn', 'vietcombank.com'
]

def extract_main_domain_part(domain):
    if not domain:
        return ""
    domain = domain.lower().strip()
    domain = domain.split(':')[0]
    
    multipart_tlds = ['.com.vn', '.edu.vn', '.gov.vn', '.org.vn', '.net.vn', '.co.uk', '.org.uk', '.com.cn', '.net.cn', '.org.cn']
    matched_suffix = None
    for suffix in multipart_tlds:
        if domain.endswith(suffix):
            matched_suffix = suffix
            break
            
    if matched_suffix:
        remaining = domain[:-len(matched_suffix)]
    else:
        parts = domain.split('.')
        if len(parts) > 1:
            remaining = ".".join(parts[:-1])
        else:
            remaining = domain
            
    parts = remaining.split('.')
    return parts[-1] if parts else ""

def normalize_homoglyphs(text):
    if not text:
        return ""
    cyrillic_map = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 
        'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j', 'ԁ': 'd', 'ԝ': 'w'
    }
    for cyr, lat in cyrillic_map.items():
        text = text.replace(cyr, lat)
    text = text.replace('vv', 'w')
    text = text.replace('rn', 'm')
    mapping = {
        '0': 'o',
        '1': 'l',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '@': 'a'
    }
    res = []
    for char in text:
        res.append(mapping.get(char, char))
    return "".join(res)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def check_lookalike_and_typosquatting(domain):
    if not domain:
        return False, ""
        
    domain_lower = domain.lower().strip()
    
    for od in OFFICIAL_DOMAINS:
        if domain_lower == od or domain_lower.endswith('.' + od):
            return False, ""
            
    input_brand = extract_main_domain_part(domain_lower)
    norm_input_brand = normalize_homoglyphs(input_brand)
    
    for od in OFFICIAL_DOMAINS:
        ob = extract_main_domain_part(od)
        
        if norm_input_brand == ob:
            if input_brand == ob:
                return True, f"Phát hiện tên miền giả mạo thương hiệu chính thức '{ob}' (sử dụng tên miền không chính thức)."
            else:
                return True, f"Phát hiện giả mạo thương hiệu '{ob}' qua ký tự tương đồng (Homoglyph Attack)."
                
        dist = levenshtein_distance(norm_input_brand, ob)
        if 1 <= dist <= 2:
            return True, f"Phát hiện giả mạo thương hiệu '{ob}' bằng sai lệch ký tự (Typosquatting, khoảng cách Levenshtein = {dist})."
            
        if ob in norm_input_brand:
            if len(ob) >= 6 or norm_input_brand == ob:
                return True, f"Phát hiện tên miền chứa thương hiệu chính thức '{ob}' để giả mạo."
            
    return False, ""

def _normalise_http_url(url):
    from urllib.parse import urlparse

    if not isinstance(url, str) or not url.strip():
        return None
    candidate = url.strip()
    if not candidate.startswith(("http://", "https://")):
        candidate = "http://" + candidate
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    try:
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return None
    except ValueError:
        return None
    return candidate


def is_safe_external_url(url):
    """Allow only public HTTP(S) destinations; reject local and special-use IPs."""
    from urllib.parse import urlparse

    candidate = _normalise_http_url(url)
    if candidate is None:
        return False, "Invalid HTTP(S) URL."
    host = urlparse(candidate).hostname
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return False, "Host name could not be resolved."
    if not addresses:
        return False, "Host name has no resolved address."
    for address in addresses:
        if not is_public_ip(address):
            return False, "Private, loopback, link-local, or reserved destination is blocked."
    return True, ""


def check_url_activity(url, timeout=2.0, allow_network=False):
    """Return an explicit offline result for dashboard URL reachability.

    A dashboard process is not an egress-security boundary.  The former
    implementation resolved a hostname and then let ``urllib`` resolve it a
    second time while connecting; an attacker could exploit that DNS-rebinding
    window to reach loopback or private services.  It also inherited proxy
    settings from the host.  Rather than claim that a UI-side HTTP probe is
    safe, production keeps this feature offline.  Deploy a separate egress
    scanner with network isolation and an audited, pinned-DNS client if live
    detonation/reachability is a business requirement.
    """
    candidate = _normalise_http_url(url) or str(url)
    return {
        "active": False,
        "status_code": None,
        "final_url": candidate,
        "error": (
            "Live URL probing is unavailable in the dashboard. "
            "Use an isolated egress-scanner service for reachability checks."
        ),
    }


@lru_cache(maxsize=128)
def analyze_url_risk(url, check_activity=False):
    """
    Phân tích độ rủi ro của một liên kết URL nghi vấn.
    """
    from urllib.parse import urlparse
    import re

    # This auxiliary model is deliberately detached from the legacy lexical
    # score below.  It is offline and fail-closed: if no signed/promoted URL
    # artifact exists, the caller receives typed UNAVAILABLE evidence instead
    # of a heuristic relabelled as machine learning.
    try:
        from src.url_ioc_model import score_url_ioc

        url_ml_ioc = score_url_ioc(url, require_promoted=True)
    except Exception as error:
        url_ml_ioc = {
            "state": "UNAVAILABLE",
            "alert": None,
            "probability": None,
            "threshold": None,
            "scope": "Vietnamese URL lexical IOC only; not an email classifier or safe verdict.",
            "detail": f"URL-IOC model could not be evaluated: {error}",
        }
    
    risk_score = 0
    reasons = []
    is_lookalike = False
    
    whitelist_domains = {
        'google.com', 'gmail.com', 'microsoft.com', 'outlook.com', 'live.com', 'yahoo.com',
        'paypal.com', 'github.com', 'facebook.com', 'apple.com', 'netflix.com', 'linkedin.com',
        'twitter.com', 'instagram.com', 'zoom.us', 'amazon.com', 'dropbox.com', 'youtube.com',
        'wikipedia.org', 'cloudflare.com', 'office.com', 'adobe.com', 'spotify.com', 'gitlab.com'
    }
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(':')[0]
        if not domain and parsed.path:
            fake_url = "http://" + url
            parsed_fake = urlparse(fake_url)
            domain = parsed_fake.netloc.lower().split(':')[0]
    except Exception:
        domain = ""
        
    if not domain:
        return {
            "url": url,
            "risk_score": 10,
            "reasons": ["Không thể trích xuất tên miền hợp lệ từ liên kết."],
            "is_lookalike": False,
            "activity_status": {
                "active": False,
                "status_code": None,
                "final_url": url,
                "error": "Không thể trích xuất tên miền hợp lệ từ liên kết.",
                "reachability_state": "ERROR"
            },
            "lexical_state": "RISK",
            "url_ml_ioc": url_ml_ioc,
        }
        
    is_input_whitelisted = False
    for wd in whitelist_domains:
        if domain == wd or domain.endswith('.' + wd):
            is_input_whitelisted = True
            break
            
    if check_activity:
        # ``check_activity`` is the explicit privacy opt-in exposed by the UI.
        # Forward it to the network guard; otherwise the checkbox appears to
        # work while ``check_url_activity`` silently remains offline.
        activity_status = check_url_activity(url, allow_network=True)
        if activity_status.get("active"):
            activity_status["reachability_state"] = "REACHABLE"
        elif activity_status.get("error"):
            activity_status["reachability_state"] = "ERROR"
        else:
            activity_status["reachability_state"] = "UNREACHABLE"
        final_url = activity_status.get("final_url", url)
    else:
        activity_status = {
            "active": False,
            "status_code": None,
            "final_url": url,
            "error": None,
            "reachability_state": "NOT_CHECKED",
            "detail": "Live reachability check was not requested."
        }
        final_url = url
        
    try:
        parsed_final = urlparse(final_url)
        final_domain = parsed_final.netloc.lower().split(':')[0]
        if not final_domain and parsed_final.path:
            fake_url = "http://" + final_url
            parsed_fake = urlparse(fake_url)
            final_domain = parsed_fake.netloc.lower().split(':')[0]
    except Exception:
        final_domain = ""
        
    is_final_whitelisted = False
    if final_domain:
        for wd in whitelist_domains:
            if final_domain == wd or final_domain.endswith('.' + wd):
                is_final_whitelisted = True
                break
    else:
        is_final_whitelisted = is_input_whitelisted
        
    lookalike_input, desc_input = check_lookalike_and_typosquatting(domain)
    if lookalike_input:
        is_lookalike = True
        risk_score += 55
        reasons.append(desc_input)
        
    if final_domain and final_domain != domain:
        lookalike_final, desc_final = check_lookalike_and_typosquatting(final_domain)
        if lookalike_final:
            is_lookalike = True
            risk_score += 55
            if desc_final not in reasons:
                reasons.append(desc_final)
                
    if is_input_whitelisted and (not check_activity or is_final_whitelisted) and not is_lookalike:
        return {
            "url": url,
            "risk_score": 0,
            "reasons": [],
            "is_lookalike": False,
            "activity_status": activity_status,
            "lexical_state": "NO_INDICATOR",
            "url_ml_ioc": url_ml_ioc,
        }
        
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    if re.match(ip_pattern, domain) or (final_domain and re.match(ip_pattern, final_domain)):
        risk_score += 40
        reasons.append("Sử dụng địa chỉ IP trực tiếp thay vì tên miền chuẩn.")
        
    shorteners = {'bit.ly', 'tinyurl.com', 'goo.gl', 'is.gd', 'buff.ly', 'ow.ly', 't.co'}
    if domain in shorteners or any(domain.endswith('.' + s) for s in shorteners):
        risk_score += 30
        reasons.append("Sử dụng dịch vụ rút gọn liên kết để ẩn giấu trang đích.")
        
    free_hosting_domains = {
        'trycloudflare.com', 'vercel.app', 'github.io', 'netlify.app', 'render.com', 
        'herokuapp.com', 'ngrok-free.app', 'ngrok.io', 'supabase.co', 'firebaseapp.com', 
        'glitch.me', 'pages.dev', '000webhostapp.com', 'rf.gd', 'infinityfreeapp.com',
        'web.app', 'repl.co', 'replit.app', 'awmdns-00.net',
        'weebly.com', 'wixsite.com', 'blogspot.com', 'wordpress.com', 
        'githubusercontent.com', 'surge.sh', '000webhost.com', 'pantheonsite.io', 'hostingerapp.com'
    }
    is_free_hosting = False
    matched_hosting = ""
    for fhd in free_hosting_domains:
        if domain == fhd or domain.endswith('.' + fhd) or (final_domain and (final_domain == fhd or final_domain.endswith('.' + fhd))):
            is_free_hosting = True
            matched_hosting = fhd
            break
    if is_free_hosting:
        risk_score += 35
        reasons.append(f"Sử dụng nền tảng lưu trữ/tạo đường hầm miễn phí dễ bị lạm dụng ({matched_hosting}).")
        
    risky_tlds = {
        '.xyz', '.top', '.club', '.work', '.info', '.site', '.tk', '.ml', '.ga', '.cf', '.gq', '.ru',
        '.cc', '.cn', '.fit', '.live', '.icu', '.buzz', '.online', '.vip', '.bid', '.country',
        '.stream', '.download', '.gdn', '.racing', '.win'
    }
    def check_tld(d):
        domain_parts = d.split('.')
        if len(domain_parts) > 1:
            tld = '.' + domain_parts[-1]
            if tld in risky_tlds:
                return tld
        return None
    tld_matched = check_tld(domain) or (check_tld(final_domain) if final_domain else None)
    if tld_matched:
        risk_score += 25
        reasons.append(f"Sử dụng tên miền mở rộng (TLD) có tỉ lệ lừa đảo cao ({tld_matched}).")
        
    sensitive_words = ['login', 'verify', 'secure', 'account', 'banking', 'update', 'signin', 'password', 'pay']
    found_words = [w for w in sensitive_words if w in url.lower() or (final_url and w in final_url.lower())]
    if found_words:
        found_words = list(sorted(set(found_words)))
        risk_score += 20
        reasons.append(f"Chứa từ khóa nhạy cảm gợi dụ đăng nhập/tài khoản: {', '.join(found_words)}")
        
    return {
        "url": url,
        "risk_score": min(100, risk_score),
        "reasons": reasons,
        "is_lookalike": is_lookalike,
        "activity_status": activity_status,
        "lexical_state": "RISK" if risk_score > 0 else "NO_INDICATOR",
        "url_ml_ioc": url_ml_ioc,
    }


def evaluate_12_layers(raw_content, parsed_info, ml_probability=0.0, llm_content=None, evidence_mode="legacy"):
    """
    ĐỘNG CƠ ĐÁNH GIÁ 12 LỚP KIỂM SOÁT EMAIL SECURITY MATRIX (ACTIVE EVALUATION ENGINE)
    Tích hợp trực tiếp logic kiểm tra 12 lớp an ninh mạng vào hệ thống backend.
    """
    # The production dashboard opts into the evidence-oriented contract.  The
    # legacy path remains available for old serialized demonstrations/tests,
    # but new research results must use explicit UNKNOWN/NO_INDICATOR states.
    if evidence_mode == "research":
        from src.security_matrix import build_12_layer_matrix

        llm_probability = None
        if llm_content:
            llm_label = extract_llm_label(llm_content)
            if llm_label == "PHISHING":
                llm_probability = 1.0
            elif llm_label == "SAFE":
                llm_probability = 0.0
        matrix = build_12_layer_matrix(
            raw_content,
            ml_probability,
            llm_probability,
            trusted_gateway_auth=bool(parsed_info.get("_trusted_mail_gateway_auth", False)),
        )
        display_names = {
            1: "1. Network IP Layer",
            2: "2. Domain Reputation",
            3: "3. Authentication Protocols",
            4: "4. MTA Routing Layer",
            5: "5. Header Metadata Layer",
            6: "6. Body Text Content",
            7: "7. Adversarial Defense",
            8: "8. HTML & Visual Layer",
            9: "9. URL & Link Intelligence",
            10: "10. Attachment Metadata Risk (No Sandbox)",
            11: "11. Statistical & ML Layer",
            12: "12. Deep Semantic AI Layer",
        }
        status_names = {
            "RISK": "🚨 RISK — Indicator detected",
            "NO_INDICATOR": "✅ CHECKED — No indicator observed",
            "UNKNOWN": "⚠️ UNKNOWN — Evidence not supplied",
            "MODEL": "🤖 MODEL — Estimate computed",
        }
        return [
            {
                "layer": display_names.get(item["layer"], item["name"]),
                "status": status_names.get(item["state"], item["state"]),
                "detail": item["detail"],
                "evidence_state": item["state"],
                "risk": item["risk"],
                "observed": item["observed"],
            }
            for item in matrix
        ]

    raw_content = limit_email_input(raw_content)
    body = parsed_info.get('body', '')
    subject = parsed_info.get('subject', '')
    sender = parsed_info.get('from', '')
    links = parsed_info.get('links', [])
    clean_text = parsed_info.get('clean_text', '')
    
    layers = []
    
    # 1. Network IP Layer
    if parsed_info.get('detection_type') != "Chuẩn EML / RFC822 Header":
        layers.append({
            "layer": "1. Network IP Layer",
            "status": "N/A (Bỏ qua - Thiếu thông tin Header)",
            "detail": "Bỏ qua do đây là email dạng Plain Text thiếu thông tin Header."
        })
    else:
        originating_ip = extract_originating_ip(raw_content, direction='backward')
        is_blacklisted, reason = is_ip_blacklisted(originating_ip)
        if is_blacklisted:
            layers.append({
                "layer": "1. Network IP Layer",
                "status": "🚨 Cảnh báo (Blacklist)",
                "detail": f"Địa chỉ IP gửi ({originating_ip}) nằm trong danh sách đen. Chi tiết: {reason}"
            })
        else:
            ip_in_links = [link for link in links if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', link)]
            if ip_in_links:
                layers.append({
                    "layer": "1. Network IP Layer",
                    "status": "🚨 Cảnh báo",
                    "detail": f"Phát hiện liên kết trực tiếp chứa địa chỉ IP: {', '.join(ip_in_links[:2])}"
                })
            elif originating_ip and originating_ip != "N/A":
                layers.append({
                    "layer": "1. Network IP Layer",
                    "status": "🟢 Kiểm duyệt",
                    "detail": f"Địa chỉ IP máy chủ chuyển tiếp (Hop IP): {originating_ip}"
                })
            else:
                layers.append({
                    "layer": "1. Network IP Layer",
                    "status": "🟢 An toàn",
                    "detail": "Không phát hiện địa chỉ IP bất thường trong luồng kết nối."
                })
        
    # 2. Domain Reputation
    risky_tlds = ['.xyz', '.top', '.club', '.work', '.info', '.site', '.tk', '.ml', '.ga', '.cf', '.gq', '.ru']
    sender_domain = sender.split('@')[-1].strip(' <>()') if '@' in sender else ''
    is_risky_domain = any(sender_domain.endswith(tld) for tld in risky_tlds)
    if is_risky_domain:
        layers.append({"layer": "2. Domain Reputation", "status": "🚨 Cảnh báo", "detail": f"Tên miền người gửi ({sender_domain}) nằm trong danh sách TLD rủi ro cao."})
    elif sender_domain:
        layers.append({"layer": "2. Domain Reputation", "status": "🟢 Đạt tiêu chuẩn", "detail": f"Tên miền người gửi: {sender_domain}"})
    else:
        layers.append({"layer": "2. Domain Reputation", "status": "⚠️ Đáng ngờ", "detail": "Không xác định được tên miền người gửi hợp lệ từ Header."})
        
    # 3. Authentication Protocols (SPF / DKIM / DMARC)
    # Pasted EML headers are attacker-controlled. Do not trust a literal
    # "dkim=pass" or the existence of a DNS record as authentication proof.
    if parsed_info.get('detection_type') == "Chuẩn EML / RFC822 Header" and not parsed_info.get('_trusted_mail_gateway_auth'):
        auth_headers_present = bool(re.search(r'^(?:authentication-results|received-spf|dkim-signature):', raw_content, re.IGNORECASE | re.MULTILINE))
        layers.append({
            "layer": "3. Authentication Protocols",
            "status": "⚠️ Chưa xác thực độc lập",
            "detail": (
                "Header SPF/DKIM/DMARC không được coi là bằng chứng khi EML do người dùng cung cấp; cần kết quả từ mail gateway/SIEM."
                if auth_headers_present else "Không có bằng chứng xác thực đáng tin cậy từ mail gateway/SIEM."
            )
        })
    elif parsed_info.get('detection_type') != "Chuẩn EML / RFC822 Header":
        layers.append({
            "layer": "3. Authentication Protocols",
            "status": "N/A (Bỏ qua - Thiếu thông tin Header)",
            "detail": "Bỏ qua do đây là email dạng Plain Text thiếu thông tin Header."
        })
    else:
        dns_validated = []
        spf_status = "Missing"
        
        originating_ip = extract_originating_ip(raw_content, direction='backward')
        
        # 1. Perform actual SPF validation if we have sender domain and originating IP
        if sender_domain and originating_ip and originating_ip != "N/A":
            try:
                spf_passed = check_ip_in_spf(originating_ip, sender_domain)
                if spf_passed:
                    spf_status = "Pass"
                    dns_validated.append("SPF Pass")
                else:
                    # Check if an SPF record exists at all
                    txt_records = perform_dns_lookup(sender_domain, 'TXT')
                    if any('v=spf1' in r.lower() for r in txt_records):
                        spf_status = "Fail (IP Mismatch)"
                    else:
                        spf_status = "Missing"
            except Exception:
                pass
        
        # 2. Check MX records
        if sender_domain:
            try:
                mx_records = perform_dns_lookup(sender_domain, 'MX')
                if mx_records:
                    dns_validated.append("MX Pass")
            except Exception:
                pass
                
        # 3. Check DMARC records
        if sender_domain:
            try:
                dmarc_records = perform_dns_lookup(f"_dmarc.{sender_domain}", 'TXT')
                if any('v=dmarc1' in r.lower() for r in dmarc_records):
                    dns_validated.append("DMARC Pass")
            except Exception:
                pass
                
        # 4. Check DKIM records from DKIM-Signature header
        try:
            dkim_match = re.search(r'dkim-signature:\s*([^\n]*(?:\n[ \t]+[^\n]*)*)', raw_content, re.IGNORECASE)
            if dkim_match:
                dkim_header_content = dkim_match.group(1)
                s_match = re.search(r'\bs=([a-zA-Z0-9\-]+)', dkim_header_content)
                d_match = re.search(r'\bd=([a-zA-Z0-9\-\.]+)', dkim_header_content)
                if s_match and d_match:
                    selector = s_match.group(1)
                    dkim_domain = d_match.group(1)
                    dkim_records = perform_dns_lookup(f"{selector}._domainkey.{dkim_domain}", 'TXT')
                    if dkim_records:
                        dns_validated.append("DKIM Pass")
        except Exception:
            pass
            
        # Determine the overall status
        if spf_status == "Fail (IP Mismatch)":
            layers.append({
                "layer": "3. Authentication Protocols",
                "status": "🚨 Cảnh báo (IP giả mạo SPF)",
                "detail": f"IP nguồn phát ({originating_ip}) không khớp với bản ghi SPF của tên miền người gửi ({sender_domain})."
            })
        elif "SPF Pass" in dns_validated or "DKIM Pass" in dns_validated:
            layers.append({
                "layer": "3. Authentication Protocols",
                "status": "🟢 Đã xác thực",
                "detail": f"Chữ ký số và hạ tầng đạt (DNS): {', '.join(dns_validated)}"
            })
        else:
            # Fallback to header parsing
            auth_found = []
            raw_lower = raw_content.lower()
            if 'received-spf: pass' in raw_lower or 'spf=pass' in raw_lower:
                auth_found.append('SPF Pass')
            if 'dkim=pass' in raw_lower or 'header.i=' in raw_lower or 'dkim-signature:' in raw_lower:
                auth_found.append('DKIM Pass')
            if 'dmarc=pass' in raw_lower:
                auth_found.append('DMARC Alignment')
                
            if auth_found:
                layers.append({
                    "layer": "3. Authentication Protocols",
                    "status": "🟢 Đã xác thực (Fallback Header)",
                    "detail": f"Xác thực qua Header (DNS ngoại tuyến/lỗi): {', '.join(auth_found)}"
                })
            else:
                layers.append({
                    "layer": "3. Authentication Protocols",
                    "status": "⚠️ Thiếu xác thực",
                    "detail": f"Không tìm thấy bằng chứng xác thực SPF/DKIM/DMARC qua DNS hoặc trong Header. SPF check: {spf_status}."
                })
        
    # 4. MTA Routing Layer
    if parsed_info.get('detection_type') != "Chuẩn EML / RFC822 Header":
        layers.append({
            "layer": "4. MTA Routing Layer",
            "status": "N/A (Bỏ qua - Thiếu thông tin Header)",
            "detail": "Bỏ qua do đây là email dạng Plain Text thiếu thông tin Header."
        })
    else:
        received_count = len(re.findall(r'received:', raw_content, re.IGNORECASE))
        if received_count > 5:
            layers.append({"layer": "4. MTA Routing Layer", "status": "⚠️ Bất thường", "detail": f"Email trải qua quá nhiều nốt chuyển tiếp MTA ({received_count} hops)."})
        else:
            layers.append({"layer": "4. MTA Routing Layer", "status": "🟢 Bình thường", "detail": f"Số lượng hop chuyển tiếp tiêu chuẩn: {received_count} hops."})
        
    # 5. Header Metadata Layer
    reply_to_match = re.search(r'reply-to:\s*([^\n\r]+)', raw_content, re.IGNORECASE)
    reply_to_val = reply_to_match.group(1).strip() if reply_to_match else ''
    is_subject_caps = subject.isupper() and len(subject) > 5
    if reply_to_val and sender and reply_to_val.lower() not in sender.lower():
        layers.append({"layer": "5. Header Metadata Layer", "status": "🚨 Cảnh báo giả mạo", "detail": f"Địa chỉ Reply-To ({reply_to_val}) không khớp với From ({sender})."})
    elif is_subject_caps:
        layers.append({"layer": "5. Header Metadata Layer", "status": "⚠️ Nghi vấn", "detail": "Tiêu đề email viết hoa toàn bộ (Full-Caps Attack)."})
    else:
        layers.append({"layer": "5. Header Metadata Layer", "status": "🟢 Hợp lệ", "detail": "Cấu trúc Header Metadata đồng nhất."})
        
    # 6. Body Text Content
    urgency_keywords = ['urgency', 'urgent', 'khẩn cấp', '24h', 'immediate', 'suspended', 'tài khoản bị khóa', 'khoá tài khoản', 'xác minh ngay']
    has_urgency = any(k in clean_text.lower() for k in urgency_keywords)
    if has_urgency:
        layers.append({"layer": "6. Body Text Content", "status": "⚠️ Đáng ngờ", "detail": "Chứa từ ngữ tạo áp lực thời gian khẩn cấp (vd: urgent, immediate, 24h)."})
    else:
        layers.append({"layer": "6. Body Text Content", "status": "🟢 Bình thường", "detail": "Nội dung văn bản không có dấu hiệu tạo áp lực tâm lý đột ngột."})
        
    # 7. Adversarial Defense
    leetspeak_detected = any(k in raw_content.lower() for k in ['w@tches', 'f.r.e.e', 'v1agra', 'p@ssword', 'r-e-p-l-i-c-a'])
    if leetspeak_detected:
        layers.append({"layer": "7. Adversarial Defense", "status": "🛡️ Đã giải mã lách luật", "detail": "Phát hiện và tự động giải mã các từ khóa ký tự biến hình (Leetspeak Obfuscation)."})
    else:
        layers.append({"layer": "7. Adversarial Defense", "status": "🟢 Sạch sẽ", "detail": "Văn bản tiêu chuẩn, không sử dụng kỹ thuật lách luật ký tự ẩn."})
        
    # 8. HTML & Visual Layer
    has_hidden_html = bool(re.search(r'display:\s*none|font-size:\s*0px|visibility:\s*hidden', raw_content, re.IGNORECASE))
    has_script = bool(re.search(r'<script', raw_content, re.IGNORECASE))
    if has_hidden_html or has_script:
        layers.append({"layer": "8. HTML & Visual Layer", "status": "🚨 Chứa mã/Thẻ ẩn", "detail": "Phát hiện thẻ HTML ẩn (Zero-font size) hoặc mã script can thiệp giao diện."})
    else:
        layers.append({"layer": "8. HTML & Visual Layer", "status": "🟢 An toàn", "detail": "Cấu trúc HTML hiển thị tiêu chuẩn, không chứa phần tử ẩn nghi vấn."})
        
    # 9. URL & Link Intelligence
    url_risks = [analyze_url_risk(link, check_activity=False) for link in links]
    max_url_risk = max([r['risk_score'] for r in url_risks], default=0)
    if max_url_risk >= 40:
        layers.append({"layer": "9. URL & Link Intelligence", "status": "🚨 Độc hại cao", "detail": f"Liên kết có độ rủi ro lớn ({max_url_risk}%). Chứa tên miền độc hại hoặc IP trực tiếp."})
    elif max_url_risk > 0:
        layers.append({"layer": "9. URL & Link Intelligence", "status": "⚠️ Đáng ngờ", "detail": f"Phát hiện dịch vụ rút gọn liên kết hoặc TLD lạ ({max_url_risk}%)."})
    else:
        layers.append({"layer": "9. URL & Link Intelligence", "status": "🟢 An toàn", "detail": "Các liên kết URL đều hướng về tên miền tiêu chuẩn."})
        
    # 10. Attachment Sandboxing
    suspicious_exts_pattern = r'\b[\w\-]+\.(exe|vbs|scr|iso|docm|xlsm|bat|cmd|ps1|js|jar|msi|lnk)\b'
    attachments = parsed_info.get('attachments', [])
    dangerous_attachments = [
        item for item in attachments
        if re.search(suspicious_exts_pattern, str(item.get('filename', '')), re.IGNORECASE)
    ]
    # Retain the body heuristic for pasted UI text, but inspect MIME metadata for
    # real EML attachments instead of silently discarding non-text parts.
    has_suspicious_att = bool(dangerous_attachments) or bool(re.search(suspicious_exts_pattern, body, re.IGNORECASE))
    if has_suspicious_att:
        names = ', '.join(str(item.get('filename')) for item in dangerous_attachments[:3])
        detail = "Cảnh báo tệp đính kèm thực thi hoặc macro nguy hiểm."
        if names:
            detail += f" Phát hiện trong MIME: {names}."
        layers.append({"layer": "10. Attachment Sandboxing", "status": "🚨 Tệp đính kèm nguy hại", "detail": detail})
    else:
        layers.append({"layer": "10. Attachment Sandboxing", "status": "🟢 An toàn", "detail": "Không tìm thấy tham chiếu tệp đính kèm chứa mã độc."})
        
    # 11. Statistical & ML Layer
    ml_pct = ml_probability * 100
    if ml_pct >= 50:
        layers.append({"layer": "11. Statistical & ML Layer", "status": "🚨 Đỏ (Critical)", "detail": f"Mô hình Logistic Regression dự đoán xác suất lừa đảo cao: {ml_pct:.2f}% (Sigmoid Score)."})
    elif ml_pct >= 30:
        layers.append({"layer": "11. Statistical & ML Layer", "status": "⚠️ Vàng (Suspicious)", "detail": f"Mô hình nhận diện dấu hiệu đáng ngờ: {ml_pct:.2f}%."})
    else:
        layers.append({"layer": "11. Statistical & ML Layer", "status": "🟢 Xanh (Safe)", "detail": f"Độ rủi ro máy học nằm trong ngưỡng an toàn: {ml_pct:.2f}%."})
        
    # 12. Deep Semantic AI Layer
    if llm_content:
        llm_label = extract_llm_label(llm_content)
                
        if llm_label == "PHISHING":
            layers.append({
                "layer": "12. Deep Semantic AI Layer",
                "status": "🚨 Đỏ (Lừa đảo)",
                "detail": "Mô hình LLM phân tích ngữ nghĩa phát hiện các yếu tố lừa đảo, thao túng tâm lý."
            })
        elif llm_label == "SAFE":
            layers.append({
                "layer": "12. Deep Semantic AI Layer",
                "status": "🟢 An toàn (Đã kiểm duyệt)",
                "detail": "Mô hình LLM xác nhận nội dung email an toàn sau khi phân tích sâu ngữ nghĩa."
            })
        else:
            layers.append({
                "layer": "12. Deep Semantic AI Layer",
                "status": "⚠️ Phản hồi LLM không hợp lệ",
                "detail": "LLM không trả về nhãn bắt buộc ở đầu phản hồi; không dùng phản hồi này để kết luận an toàn."
            })
    elif ml_pct >= 30 or max_url_risk >= 30 or has_urgency:
        layers.append({"layer": "12. Deep Semantic AI Layer", "status": "🤖 Chờ LLM Phân tích", "detail": "Email được kích hoạt tầng đọc hiểu ngữ nghĩa sâu bằng Local LLM (LM Studio)."})
    else:
        layers.append({"layer": "12. Deep Semantic AI Layer", "status": "🟢 Thấu hiểu ngữ nghĩa", "detail": "Ngữ cảnh email thể hiện các hoạt động giao tiếp bình thường."})
        
    return layers

def classify_cyber_threat(
    raw_content,
    parsed_info,
    layers_eval,
    ml_probability=0.0,
    evidence_mode="legacy",
    ml_threshold=0.5,
):
    """
    MA TRẬN RỦI RO AN NINH MẠNG VÀ PHÂN LOẠI MỐI ĐE DỌA (CYBER RISK MATRIX & THREAT CLASSIFICATION)
    Phân định bản chất: Credential Harvesting, BEC (CEO Fraud), Ransomware, hay Commercial Spam.
    """
    clean_text = parsed_info.get('clean_text', '').lower()
    links = parsed_info.get('links', [])
    
    # 1. Credential Harvesting
    login_kw = ['login', 'signin', 'password', 'mật khẩu', 'đăng nhập', 'verify account', 'xác minh tài khoản', 'banking', 'ngân hàng']
    has_login_terms = any(k in clean_text for k in login_kw)
    has_links = len(links) > 0
    
    # 2. Business Email Compromise (BEC / CEO Fraud)
    bec_kw = ['ceo', 'cfo', 'giám đốc', 'chuyển tiền gấp', 'wire transfer', 'gift card', 'thẻ quà tặng', 'bảo mật thông tin', 'chuyển khoản khẩn']
    has_bec_terms = any(k in clean_text for k in bec_kw)
    
    # 3. Malware / Ransomware Delivery
    att_layer = next(
        (
            layer_entry
            for layer_entry in layers_eval
            if str(layer_entry.get('layer', '')).startswith('10. Attachment')
        ),
        None,
    )
    has_bad_att = att_layer and '🚨' in att_layer['status']
    
    # Filter out layers with "N/A" status when deciding threat indicators
    active_layers = [layer_entry for layer_entry in layers_eval if "N/A" not in layer_entry['status']]
    
    # Xác định các cảnh báo kỹ thuật nghiêm trọng
    critical_warnings = False
    for layer_entry in active_layers:
        status = layer_entry.get("status", "")
        layer = layer_entry.get("layer", "")
        if layer_entry.get("evidence_state") == "RISK" and any(
            token in layer
            for token in (
                "1. Network IP Layer",
                "3. Authentication Protocols",
                "5. Header Metadata Layer",
                "9. URL & Link Intelligence",
                "10. Attachment Sandboxing",
                "10. Attachment Metadata Risk",
            )
        ):
            critical_warnings = True
            continue
        # Các chỉ số kỹ thuật thực sự nguy hiểm:
        # - Lớp 1: IP nguồn nằm trong Blacklist
        # - Lớp 3: IP giả mạo SPF
        # - Lớp 5: Reply-To bị giả mạo / không khớp với From
        # - Lớp 9: URL độc hại cao (risk >= 40%)
        # - Lớp 10: File đính kèm nguy hiểm
        if "1. Network IP Layer" in layer and "Blacklist" in status:
            critical_warnings = True
        elif "3. Authentication Protocols" in layer and "giả mạo SPF" in status:
            critical_warnings = True
        elif "5. Header Metadata Layer" in layer and "giả mạo" in status:
            critical_warnings = True
        elif "9. URL & Link Intelligence" in layer and "Độc hại cao" in status:
            critical_warnings = True
        elif layer.startswith("10. Attachment") and (
            "nguy hại" in status or layer_entry.get("evidence_state") == "RISK"
        ):
            critical_warnings = True

    # Quyết định xem có coi email là mối đe dọa lừa đảo (phishing) hay không
    # Technical evidence can escalate a message to Suspicious, but it cannot
    # by itself prove phishing. An authenticated internal account may be
    # compromised (BEC/CEO fraud), so authentication is not a substitute for
    # behavioural/contextual evidence.
    warning_threshold = float(ml_threshold) * 0.6
    is_phishing_threat = ml_probability >= ml_threshold or (
        critical_warnings and evidence_mode != "research"
    )

    # Phân loại mối đe dọa
    if is_phishing_threat:
        if has_bad_att:
            return {
                "vector": "3. Phát tán mã độc / Ransomware (Malware / Ransomware Delivery)",
                "severity": "🔴 Rất Cao (Critical Risk)",
                "description": "Email chứa hoặc liên kết tới tập tin đính kèm độc hại (Macro/Exe) nhằm mã hóa dữ liệu đòi tiền chuộc hoặc chiếm quyền máy tính.",
                "action": "Chặn ngay lập tức, cách ly tệp đính kèm và gửi báo cáo cho SOC."
            }
        elif has_bec_terms:
            return {
                "vector": "2. Giả mạo email doanh nghiệp (Business Email Compromise - BEC)",
                "severity": "🔴 Rất Cao (Critical Risk)",
                "description": "Giả mạo lãnh đạo/đối tác yêu cầu chuyển tiền gấp hoặc tiết lộ thông tin nhạy cảm của doanh nghiệp.",
                "action": "Cảnh báo người dùng kiểm tra qua kênh điện thoại nội bộ, không thực hiện lệnh chuyển tiền."
            }
        elif has_login_terms and has_links:
            return {
                "vector": "1. Thu thập thông tin xác thực (Credential Harvesting)",
                "severity": "🔴 Rất Cao (Critical Risk)",
                "description": "Dụ dỗ nạn nhân nhấp vào liên kết trang web giả mạo để chiếm đoạt tài khoản email/ngân hàng.",
                "action": "Chặn email, thêm URL vào danh sách đen (Blacklist) của Firewall."
            }
        else:
            # Nếu ML dự đoán lừa đảo cao (>=0.5) hoặc có cảnh báo kỹ thuật nghiêm trọng đi kèm với ML >= 0.5
            if ml_probability >= ml_threshold:
                return {
                    "vector": "Email Lừa Đảo Tổng Hợp (General Phishing)",
                    "severity": "🟠 Cao (High Risk)",
                    "description": "Email sử dụng các chiêu trò thao túng tâm lý xã hội để lừa đảo.",
                    "action": "Đưa email vào thư mục Spam/Junk và cảnh báo nguy cơ."
                }
            else:
                return {
                    "vector": "Email Đáng Ngờ (Suspicious / Potential Spam)",
                    "severity": "🟡 Trung bình (Medium Risk)",
                    "description": "Email có dấu hiệu đáng ngờ hoặc chứa các yếu tố quảng cáo, lôi kéo (Spam/Noise).",
                    "action": "Đưa vào hộp thư rác (Spam) hoặc cảnh báo người dùng cẩn trọng."
                }
    else:
        # ML là tín hiệu quyết định mặc định khi không có bằng chứng kỹ thuật nghiêm trọng.
        # Không nâng một email lên "Đáng ngờ" chỉ vì một cảnh báo yếu/heuristic đơn lẻ
        # (ví dụ HTML có display:none, TLD ít phổ biến, hoặc tiêu đề viết hoa). Các tín hiệu
        # này vẫn được giữ nguyên trong bảng 12 lớp để người dùng kiểm tra, nhưng chưa đủ
        # để phủ định kết luận Safe của ML.
        if ml_probability >= warning_threshold or (critical_warnings and evidence_mode == "research"):
            return {
                "vector": "Email Đáng Ngờ (Suspicious / Potential Spam)",
                "severity": "🟡 Trung bình (Medium Risk)",
                "description": "Mô hình máy học đánh giá nguy cơ ở mức trung bình; cần kiểm tra thêm trước khi thực hiện thao tác.",
                "action": "Khuyến nghị người dùng kiểm tra kỹ các liên kết trước khi thực hiện thao tác."
            }
        else:
            return {
                "vector": "Email Công Việc An Toàn (Legitimate Mail)",
                "severity": "🟢 An Toàn (Safe)",
                "description": "ML đánh giá email an toàn và không có bằng chứng kỹ thuật nghiêm trọng. Các cảnh báo nhẹ (nếu có) chỉ được hiển thị để tham khảo, không làm đổi kết luận an toàn.",
                "action": "Cho phép chuyển trực tiếp vào Hộp thư đến (Inbox)."
            }

def is_public_ip(ip_str):
    try:
        ip = ipaddress.ip_address(ip_str)
        # ``is_global`` covers documentation, multicast, CGNAT, loopback and
        # reserved ranges.  A hand-maintained allow list inevitably drifts and
        # previously treated several non-routable addresses as public.
        return bool(
            ip.is_global
            and not ip.is_multicast
            and not ip.is_unspecified
            and not ip.is_reserved
            and not ip.is_loopback
            and not ip.is_link_local
        )
    except Exception:
        return False

def extract_ip_candidates(text):
    if not text:
        return []
    cleaned_text = re.sub(r'[^0-9a-fA-F.:]', ' ', text)
    tokens = cleaned_text.split()
    
    valid_ips = []
    for token in tokens:
        if '.' not in token and ':' not in token:
            continue
        try:
            ip = ipaddress.ip_address(token)
            valid_ips.append(str(ip))
        except ValueError:
            continue
    return valid_ips

def extract_originating_ip(raw_content, direction='backward'):
    if not raw_content:
        return "N/A"
    
    received_headers = []
    current_header = []
    for line in raw_content.splitlines():
        if line.lower().startswith('received:'):
            if current_header:
                received_headers.append("\n".join(current_header))
            current_header = [line]
        elif current_header and (line.startswith(' ') or line.startswith('\t')):
            current_header.append(line)
        else:
            if current_header:
                received_headers.append("\n".join(current_header))
                current_header = []
    if current_header:
        received_headers.append("\n".join(current_header))
    
    if not received_headers:
        all_ips = extract_ip_candidates(raw_content)
        if not all_ips:
            return "N/A"
        
        for ip in all_ips:
            if is_public_ip(ip):
                return ip
        return all_ips[0]
    
    headers_to_scan = received_headers
    if direction == 'backward':
        headers_to_scan = list(reversed(received_headers))
        
    found_ips = []
    for header in headers_to_scan:
        candidates = extract_ip_candidates(header)
        for ip in candidates:
            if ip not in found_ips:
                found_ips.append(ip)
                
    if not found_ips:
        return "N/A"
        
    for ip in found_ips:
        if is_public_ip(ip):
            return ip
            
    return found_ips[0]

@lru_cache(maxsize=256)
def perform_dns_lookup(domain, record_type='A'):
    if dns is None:
        return []
    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = 2.0
        resolver.lifetime = 2.0
        answers = resolver.resolve(domain, record_type)
        return [str(rdata) for rdata in answers]
    except Exception:
        # Do not fall back to public resolvers: sending sender domains to them
        # violates the on-premise/offline privacy promise.
        if record_type == 'A':
            try:
                ip = socket.gethostbyname(domain)
                return [ip]
            except Exception:
                pass
        return []

def check_ip_in_spf(ip_str, domain, depth=0):
    if depth > 3:
        return False
    if not ip_str or ip_str == "N/A":
        return False
        
    try:
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        return False
        
    txt_records = perform_dns_lookup(domain, 'TXT')
    spf_record = None
    for r in txt_records:
        if r.lower().startswith('v=spf1'):
            spf_record = r
            break
            
    if not spf_record:
        return False
        
    tokens = spf_record.split()
    
    for token in tokens[1:]:
        token = token.lower()
        
        # Strip mechanism qualifiers (+, -, ~, ?)
        qualifier = ''
        if token and token[0] in ['+', '-', '~', '?']:
            qualifier = token[0]
            token = token[1:]
            
        if token.startswith('ip4:'):
            cidr = token[4:]
            try:
                if '/' in cidr:
                    network = ipaddress.ip_network(cidr, strict=False)
                    if ip in network:
                        return qualifier != '-'
                else:
                    if ip == ipaddress.ip_address(cidr):
                        return qualifier != '-'
            except Exception:
                pass
                
        elif token.startswith('ip6:'):
            cidr = token[4:]
            try:
                if '/' in cidr:
                    network = ipaddress.ip_network(cidr, strict=False)
                    if ip in network:
                        return qualifier != '-'
                else:
                    if ip == ipaddress.ip_address(cidr):
                        return qualifier != '-'
            except Exception:
                pass
                
        elif token.startswith('include:'):
            sub_domain = token[8:]
            if check_ip_in_spf(ip_str, sub_domain, depth + 1):
                return True
                
        elif token.startswith('redirect='):
            sub_domain = token[9:]
            if check_ip_in_spf(ip_str, sub_domain, depth + 1):
                return True
                
        elif token == 'a' or token == '+a':
            a_ips = perform_dns_lookup(domain, 'A')
            for a_ip in a_ips:
                try:
                    if ip == ipaddress.ip_address(a_ip):
                        return True
                except Exception:
                    pass
        elif token.startswith('a:'):
            sub_domain = token[2:]
            a_ips = perform_dns_lookup(sub_domain, 'A')
            for a_ip in a_ips:
                try:
                    if ip == ipaddress.ip_address(a_ip):
                        return True
                except Exception:
                    pass
        elif token == 'mx' or token == '+mx':
            mx_records = perform_dns_lookup(domain, 'MX')
            for mx in mx_records:
                mx_domain = mx.split()[-1]
                mx_ips = perform_dns_lookup(mx_domain, 'A')
                for mx_ip in mx_ips:
                    try:
                        if ip == ipaddress.ip_address(mx_ip):
                            return True
                    except Exception:
                        pass
        elif token.startswith('mx:'):
            sub_domain = token[3:]
            mx_records = perform_dns_lookup(sub_domain, 'MX')
            for mx in mx_records:
                mx_domain = mx.split()[-1]
                mx_ips = perform_dns_lookup(mx_domain, 'A')
                for mx_ip in mx_ips:
                    try:
                        if ip == ipaddress.ip_address(mx_ip):
                            return True
                    except Exception:
                        pass
                        
    return False

def _load_blacklist():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_try = [
        os.path.join(current_dir, "..", "data", "ip_blacklist.json"),
        os.path.join("data", "ip_blacklist.json"),
        os.path.join("..", "data", "ip_blacklist.json"),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    "specific_ips": set(data.get("specific_ips", [])),
                    "cidr_ranges": list(data.get("cidr_ranges", []))
                }
            except Exception:
                pass
    return {
        "specific_ips": {"192.168.1.50", "198.51.100.42", "203.0.113.15", "192.0.2.23"},
        "cidr_ranges": ["45.227.254.0/24", "185.220.101.0/24", "190.115.18.0/24"]
    }

def is_ip_blacklisted(ip_str):
    if not ip_str or ip_str == "N/A":
        return False, None
    blacklist = _load_blacklist()
    if ip_str in blacklist["specific_ips"]:
        return True, "IP listed in specific offline blacklist."
    try:
        ip = ipaddress.ip_address(ip_str)
        for range_str in blacklist["cidr_ranges"]:
            if ip in ipaddress.ip_network(range_str):
                return True, f"IP belongs to blacklisted range {range_str}."
    except Exception:
        pass
    return False, None

def group_network_security_features(raw_content, allow_dns=False):
    """
    Groups network security features (Originating IP, DNS lookup status, Blacklist status)
    and handles missing/empty values by returning 'N/A' or localized equivalents.
    """
    # 1. Extract Originating IP (using backward scan as default)
    originating_ip = extract_originating_ip(raw_content, direction='backward')
    
    # 2. Extract sender domain to check DNS
    parsed_info = parse_raw_email(raw_content)
    sender = parsed_info.get('from', '')
    sender_domain = "N/A"
    dns_status = "N/A"
    dns_records = {"A": [], "MX": [], "TXT": []}
    
    if '@' in sender:
        sender_domain = sender.split('@')[-1].strip(' <>')
        if sender_domain and allow_dns:
            # Keep the individual records available for the forensic view.
            mx_records = perform_dns_lookup(sender_domain, 'MX')
            a_records = perform_dns_lookup(sender_domain, 'A')
            txt_records = perform_dns_lookup(sender_domain, 'TXT')
            dns_records = {"A": a_records[:5], "MX": mx_records[:5], "TXT": txt_records[:5]}
            if mx_records:
                dns_status = f"Valid MX records: {', '.join(mx_records[:2])}"
            else:
                if a_records:
                    dns_status = f"Valid A records: {', '.join(a_records[:2])}"
                else:
                    dns_status = "No DNS records found"
        elif sender_domain:
            dns_status = "Skipped (offline privacy mode)"
                    
    # 3. Check Blacklist status
    is_blacklisted, reason = is_ip_blacklisted(originating_ip)
    blacklist_status = "Clean"
    if originating_ip == "N/A":
        blacklist_status = "N/A"
    elif is_blacklisted:
        blacklist_status = f"Blacklisted ({reason})"
        
    authentication = extract_authentication_evidence(raw_content, allow_dns=allow_dns)
    iocs = extract_iocs(raw_content, allow_dns=allow_dns)
    return {
        "originating_ip": originating_ip if originating_ip else "N/A",
        "sender_domain": sender_domain if sender_domain else "N/A",
        "dns_status": dns_status if dns_status else "N/A",
        "dns_records": dns_records,
        "blacklist_status": blacklist_status if blacklist_status else "N/A",
        "blacklist_reason": reason,
        "addresses": parsed_info.get("addresses", {}),
        "authentication": authentication,
        "iocs": iocs,
    }

if __name__ == "__main__":

    demo_text = "From: security@paypal-verification.com\nSubject: URGENT!! Reset your password\n\nReset your password at: http://192.168.1.50/verify! Details: f.r.e.e w@tches!"
    print("Email gốc: ", demo_text)
    parsed = parse_raw_email(demo_text)
    print("Bóc tách Raw Email:", parsed)
    print("Đặc trưng cấu trúc chuẩn hóa:", extract_structural_features(demo_text))
    print("Sau khi qua bộ lọc văn bản:", advanced_cleaning_defense(demo_text))
    layers = evaluate_12_layers(demo_text, parsed, 0.85)
    print("Đánh giá 12 Lớp Security Matrix:", layers)
    print("Phân loại Mối đe dọa Cyber Risk:", classify_cyber_threat(demo_text, parsed, layers, 0.85))
