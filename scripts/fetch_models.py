"""Download the two ONNX models this project runs on.

The weights are ~300MB, which does not belong in git, so they are fetched instead.
Each file's SHA256 is printed and (once pinned in config.EXPECTED_SHA256) checked, so a
truncated or swapped download fails loudly here rather than silently producing garbage
embeddings three stages later.

Usage:  uv run python scripts/fetch_models.py
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile

from facepipe.config import DET_MODEL, EXPECTED_SHA256, MODEL_BUNDLE_URL, MODEL_DIR, REC_MODEL

WANTED = {DET_MODEL.name, REC_MODEL.name}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str) -> bytes:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        buf = io.BytesIO()
        read = 0
        while chunk := resp.read(1 << 20):
            buf.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r  {read / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    return buf.getvalue()


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if DET_MODEL.exists() and REC_MODEL.exists():
        print("models already present, skipping download")
    else:
        blob = download(MODEL_BUNDLE_URL)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for member in zf.namelist():
                name = member.rsplit("/", 1)[-1]
                if name in WANTED:
                    target = MODEL_DIR / name
                    print(f"extracting {name}")
                    target.write_bytes(zf.read(member))

    ok = True
    for path in (DET_MODEL, REC_MODEL):
        if not path.exists():
            print(f"MISSING: {path.name} was not in the bundle")
            ok = False
            continue
        digest = sha256(path)
        expected = EXPECTED_SHA256.get(path.name)
        size_mb = path.stat().st_size / 1e6
        if expected is None:
            print(f"{path.name}  {size_mb:6.1f} MB  sha256={digest}  (not pinned yet)")
        elif expected != digest:
            print(f"{path.name}  SHA MISMATCH\n  expected {expected}\n  got      {digest}")
            ok = False
        else:
            print(f"{path.name}  {size_mb:6.1f} MB  sha256 ok")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
