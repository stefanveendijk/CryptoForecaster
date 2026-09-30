from __future__ import annotations

import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

import forecast_proof


def _row_key(row) -> tuple[str, str, int, str] | None:
    try:
        return (
            str(pd.Timestamp(row.get("date", row.get("forecast_date"))).date()),
            str(row.get("coin", "")).upper(),
            int(row.get("horizon")),
            str(row.get("model", "META")).upper(),
        )
    except Exception:
        return None


def _existing_keys(path: Path) -> set[tuple[str, str, int, str]]:
    if not path.exists():
        return set()
    try:
        ledger = pd.read_csv(path)
    except Exception:
        return set()
    keys: set[tuple[str, str, int, str]] = set()
    for _, row in ledger.iterrows():
        key = _row_key(row)
        if key is not None:
            keys.add(key)
    return keys


def _safe_to_recover(row, file_mtime_utc: datetime) -> bool:
    """Recover only when the file timestamp proves the forecast existed before its outcome.

    Daily crypto bars can legitimately lag the computer date by one day. Therefore
    we do not require forecast_date == today. Instead, the modification time of the
    preserved latest_forecasts.csv must be earlier than the end of the forecast's
    target day. This keeps recovery leakage-safe while allowing a morning run whose
    latest completed candle is dated yesterday.
    """
    try:
        forecast_date = pd.Timestamp(row.get("date")).date()
        horizon = int(row.get("horizon"))
    except Exception:
        return False
    if horizon <= 0 or forecast_date > date.today():
        return False
    target_date = forecast_date + timedelta(days=horizon)
    outcome_cutoff = datetime.combine(
        target_date + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    return file_mtime_utc < outcome_cutoff


def _restore_issuance_time(
    ledger_path: Path,
    newly_added_keys: set[tuple[str, str, int, str]],
    file_mtime_utc: datetime,
) -> None:
    if not newly_added_keys or not ledger_path.exists():
        return
    try:
        ledger = pd.read_csv(ledger_path)
    except Exception:
        return

    changed = False
    for idx, row in ledger.iterrows():
        key = _row_key(row)
        if key in newly_added_keys:
            ledger.at[idx, "issued_at_utc"] = file_mtime_utc.isoformat()
            changed = True

    if changed:
        tmp = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        ledger.to_csv(tmp, index=False)
        os.replace(tmp, ledger_path)


def main() -> int:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "local_data")
    output = data_dir / "ultimate_run" / "output"
    latest_path = output / "latest_forecasts.csv"
    price_path = output / "price_history.csv"
    ledger_path = output / forecast_proof.LEDGER_FILE

    if not latest_path.exists():
        print("PROOF_REPAIR: nog geen latest_forecasts.csv; niets te herstellen.")
        return 0

    try:
        latest = pd.read_csv(latest_path)
    except Exception as exc:
        print(f"PROOF_REPAIR: latest_forecasts.csv niet leesbaar: {exc}")
        return 0

    if "date" not in latest.columns or "horizon" not in latest.columns:
        print("PROOF_REPAIR: forecastdatum of horizon ontbreekt; niets geregistreerd.")
        return 0

    try:
        file_mtime_utc = datetime.fromtimestamp(latest_path.stat().st_mtime, timezone.utc)
    except Exception as exc:
        print(f"PROOF_REPAIR: bestandsdatum niet leesbaar: {exc}")
        return 0

    if "model" in latest.columns:
        latest = latest[latest["model"].astype(str).str.upper() == "META"].copy()

    safe_mask = latest.apply(lambda row: _safe_to_recover(row, file_mtime_utc), axis=1)
    recoverable = latest.loc[safe_mask].copy()

    if recoverable.empty:
        print(
            "PROOF_REPAIR: geen veilig herstelbare voorspellingen; "
            "er wordt niets achteraf toegevoegd nadat een uitkomst bekend kan zijn."
        )
    else:
        before = _existing_keys(ledger_path)
        recoverable_keys = {k for k in (_row_key(r) for _, r in recoverable.iterrows()) if k is not None}
        info = forecast_proof.record_forecasts(output, recoverable)
        newly_added = recoverable_keys - before
        _restore_issuance_time(ledger_path, newly_added, file_mtime_utc)
        print(
            "PROOF_REPAIR: veilig toegevoegd="
            f"{info['added']} bestaand={info['existing']} "
            f"bewijs-tijd={file_mtime_utc.isoformat()}"
        )

    if price_path.exists():
        try:
            prices = pd.read_csv(price_path, parse_dates=["date"]).set_index("date").sort_index()
            result = forecast_proof.evaluate_ledger(output, prices)
            print(f"PROOF_REPAIR: beoordeelde voorspellingen={len(result)}")
        except Exception as exc:
            print(f"PROOF_REPAIR: beoordelen overgeslagen: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
