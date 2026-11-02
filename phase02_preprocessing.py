"""Phase 02 — clean text before it is counted or tokenized."""

import re


def clean_text(text: str) -> str:
    """Strip edges, lowercase, collapse spaces, and drop punctuation."""
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def main():
    print("=== Phase 02 — text preprocessing ===")
    samples = [
        "   I NEED HELP!!!   ",
        "The API is NOT responding...",
        "  Where   can I submit my application? ",
    ]
    for sample in samples:
        print(f"Original: {sample!r}")
        print(f"Cleaned:  {clean_text(sample)!r}")
    print("Preprocessing makes the same words look the same, so capitalization,")
    print("extra spaces, and punctuation do not create extra vocabulary entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
