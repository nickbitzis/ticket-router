"""Capture README screenshots with headless Chrome. Start the server first:

make run                                   # in one terminal, wait for the "ready" log line
uv run python scripts/screenshot.py --port 8836
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def chrome() -> str:
    for c in CHROME_PATHS:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Chrome not found")


def shoot(url: str, out: Path, width: int, height: int, dark: bool = False) -> None:
    args = [
        chrome(),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",
        f"--screenshot={out}",
    ]
    # Pin prefers-color-scheme so the result does not depend on the OS theme (0 dark, 1 light).
    args.append(f"--blink-settings=preferredColorScheme={0 if dark else 1}")
    subprocess.run([*args, url], check=True, capture_output=True, timeout=90)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8836)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}/"
    DOCS.mkdir(exist_ok=True)
    shoot(base, DOCS / "screenshot.png", 1400, 1900)
    shoot(base, DOCS / "screenshot-dark.png", 1400, 1900, dark=True)


if __name__ == "__main__":
    main()
