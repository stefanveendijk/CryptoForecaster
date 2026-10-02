from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

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

    if "date" not in latest.columns or "horizon" not in latest.columns:
        print("PROOF_REPAIR: forecastdatum of horizon ontbreekt; niets geregistreerd.")
        return 0

    try:
        file_mtime_utc = datetime.fromtimestamp(
            latest_path.stat().st_mtime, timezone.utc
        )
    except Exception as exc:
        print(f"PROOF_REPAIR: bestandsdatum niet leesbaar: {exc}")
        return 0

    # De bestandsdatum is bewijs van wanneer deze voorspellingen uiterlijk bestonden.
    # forecast_proof bewaakt zelf dat niets wordt toegevoegd nadat de uitkomst al
    # bekend kon zijn. Daardoor kan een ochtendrun met modeldatum gisteren veilig
    # worden hersteld zonder achteraf bewijs te fabriceren.
    info = forecast_proof.record_forecasts(
        output,
        latest,
        issued_at_utc=file_mtime_utc.isoformat(),
    )
    print(
        "PROOF_REPAIR: toegevoegd="
        f"{info['added']} bestaand={info['existing']} "
        f"te-laat-geweigerd={info.get('skipped_late', 0)} "
        f"bewijs-tijd={file_mtime_utc.isoformat()}"
    )

    if price_path.exists():
        try:
            prices = (
                pd.read_csv(price_path, parse_dates=["date"])
                .set_index("date")
                .sort_index()
            )
            result = forecast_proof.evaluate_ledger(output, prices)
            print(f"PROOF_REPAIR: beoordeelde uitgiftes={len(result)}")
        except Exception as exc:
            print(f"PROOF_REPAIR: beoordelen overgeslagen: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
