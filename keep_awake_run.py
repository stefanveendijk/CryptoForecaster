from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def keep_awake(enabled: bool) -> None:
    if os.name != "nt":
        return
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enabled else 0)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()

    app_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(app_dir / "crypto_forecaster_ultimate.py"),
        "--workdir",
        str(Path(args.workdir)),
    ]
    if args.deep:
        cmd.append("--deep")
    if args.no_news:
        cmd.append("--no-news")

    keep_awake(True)
    try:
        return subprocess.call(cmd, cwd=str(app_dir), env=os.environ.copy())
    finally:
        keep_awake(False)

if __name__ == "__main__":
    raise SystemExit(main())
