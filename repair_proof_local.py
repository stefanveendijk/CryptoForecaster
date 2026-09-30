from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

import forecast_proof


def main() -> int:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "local_data")
    output = data_dir / "ultimate_run" / "output"
    latest_path = output / "latest_forecasts.csv"
    price_path = output / "price_history.csv"

    if not latest_path.exists():
        print("PROOF_REPAIR: nog geen latest_forecasts.csv; niets te herstellen.")
        return 0

    try:
        latest = pd.read_csv(latest_path)
    except Exception as exc:
        print(f"PROOF_REPAIR: latest_forecasts.csv niet leesbaar: {exc}")
        return 0

    if "date" not in latest.columns:
        print("PROOF_REPAIR: forecastdatum ontbreekt; niets geregistreerd.")
        return 0

    # Integrity rule: never backfill an old prediction after its outcome may already
    # be known. We only recover forecasts whose model date equals today's local date.
    dates = pd.to_datetime(latest["date"], errors="coerce").dt.date
    fresh = latest.loc[dates == date.today()].copy()
    if fresh.empty:
        print("PROOF_REPAIR: geen voorspelling van vandaag; oude voorspellingen worden bewust niet achteraf geregistreerd.")
    else:
        info = forecast_proof.record_forecasts(output, fresh)
        print(f"PROOF_REPAIR: vandaag toegevoegd={info['added']} bestaand={info['existing']}")

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
