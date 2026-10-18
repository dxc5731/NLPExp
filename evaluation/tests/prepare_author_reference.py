"""Prepare the pinned external entity-scoring reference used only by tests."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

from author_entity_reference import AUTHOR_PATH, AUTHOR_SHA256, AUTHOR_URL

RAW_URL = AUTHOR_URL.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional existing copy for offline preparation.")
    args = parser.parse_args()
    if AUTHOR_PATH.exists():
        raw = AUTHOR_PATH.read_bytes()
    elif args.source is not None:
        raw = args.source.read_bytes()
    else:
        with urllib.request.urlopen(RAW_URL, timeout=30) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != AUTHOR_SHA256:
        raise SystemExit("FAIL: reference SHA-256 differs from the pinned version; no file was replaced.")
    if not AUTHOR_PATH.exists():
        AUTHOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUTHOR_PATH.open("xb") as stream:
            stream.write(raw)
    print("PASS: pinned reference SHA-256 verified:", AUTHOR_PATH)
    print("External test-only source; not part of the published repository. No model was run.")


if __name__ == "__main__":
    main()
