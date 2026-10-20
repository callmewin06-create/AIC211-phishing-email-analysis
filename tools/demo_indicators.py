"""Run the original technical evidence extractor on fabricated local email.

This is not a trained ML/LLM classifier and produces no final verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.security_matrix import build_12_layer_matrix  # noqa: E402


def main() -> int:
    # Keep machine-readable output consistent on Windows and Unix terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "email", nargs="?", type=Path,
        default=ROOT / "examples" / "synthetic_phishing.eml",
        help="Local text/EML file; defaults to the included fabricated example.",
    )
    parser.add_argument("--json", action="store_true", help="Print structured evidence as JSON.")
    args = parser.parse_args()
    try:
        if args.email.stat().st_size > 5_000_000:
            parser.error("Use an example smaller than 5 MB.")
        raw = args.email.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        parser.error(f"Cannot read the local example: {error}")
    evidence = build_12_layer_matrix(raw)
    if args.json:
        print(json.dumps({
            "mode": "offline_technical_evidence_only",
            "final_verdict": None,
            "ml_available": False,
            "llm_available": False,
            "layers": evidence,
        }, indent=2, ensure_ascii=False))
    else:
        print("Offline technical evidence only. No ML/LLM inference or final phishing verdict.")
        print("No email links are opened and no attachments are executed.\n")
        for item in evidence:
            print(f"{item['layer']:>2}. {item['state']:<12} {item['name']}")
            print(f"    {item['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
