from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("source_id")
    parser.add_argument("pattern")
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--chars", type=int, default=2500)
    args = parser.parse_args()

    path = Path(__file__).resolve().parent / "corpus_text" / f"{args.source_id}.txt"
    content = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^=== PAGE (\d+) ===\s*$", content)
    matches = 0
    rx = re.compile(args.pattern, re.IGNORECASE | re.MULTILINE)
    for index in range(1, len(parts), 2):
        page_number = parts[index]
        page_text = parts[index + 1].strip()
        hit = rx.search(page_text)
        if not hit:
            continue
        start = max(0, hit.start() - args.chars // 3)
        end = min(len(page_text), hit.end() + args.chars)
        print(f"=== PAGE {page_number} ===")
        print(page_text[start:end])
        print()
        matches += 1
        if matches >= args.max_pages:
            break


if __name__ == "__main__":
    main()
