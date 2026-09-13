from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
STATE_FILE = DATA_DIR / "cloud_state.json"
MODEL_VERSION = os.getenv("MODEL_VERSION", "").strip()


def main():
    if not MODEL_VERSION:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        state = {}

    if state.get("model_version") == MODEL_VERSION:
        return

    # Keep the existing forecast files available while a new model version runs,
    # but make the scheduler regard the model as due for a fresh run.
    state["model_version"] = MODEL_VERSION
    state["running"] = False
    state["last_success"] = None
    state["version_refresh_requested_at"] = datetime.now(timezone.utc).isoformat()
    state["reason"] = "model-version-upgrade"
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
