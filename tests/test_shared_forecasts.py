"""Fast checks that laptop forecast cards and GitHub Pages use one source."""
import os
from datetime import date, datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["LOCAL_MODE"] = "true"
os.environ["AUTO_RUN_MODEL"] = "false"

import cloud_server
import trading_strategy
from forecast_quality import matured_history


class SharedForecastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_output = cloud_server.OUTPUT_DIR
        cloud_server.OUTPUT_DIR = Path(self.tmp.name) / "output"
        cloud_server.shared_forecast_cache.update(ts=0.0, payload=None)
        cloud_server.shared_history_cache.clear()

    def tearDown(self):
        cloud_server.OUTPUT_DIR = self.original_output
        cloud_server.shared_forecast_cache.update(ts=0.0, payload=None)
        cloud_server.shared_history_cache.clear()
        self.tmp.cleanup()

    @patch.object(cloud_server.requests, "get")
    def test_laptop_returns_online_forecasts_unchanged(self, get):
        expected = {
            "source": "github-pages", "updatedAt": "2026-10-26",
            "baseCurrency": "USD", "forecasts": [
                {"coin": "BTC", "horizon": 7, "date": "2026-10-26",
                 "probUp": 0.479, "expectedPrice": 70000.0}
            ], "metrics": [], "events": [],
        }
        response = Mock()
        response.json.return_value = expected
        response.raise_for_status.return_value = None
        get.return_value = response
        result = cloud_server.dashboard()
        self.assertEqual(result["forecasts"], expected["forecasts"])
        self.assertEqual(result["baseCurrency"], "USD")
        self.assertEqual(result["forecastOrigin"], "github-pages")
        self.assertEqual(result["strategyOrigin"], "local-model")
        self.assertEqual(cloud_server.dashboard()["forecasts"], expected["forecasts"])
        self.assertEqual(get.call_count, 1, "Cache prevents repeated requests")

    @patch.object(cloud_server.requests, "get", side_effect=Exception("offline"))
    def test_offline_does_not_forge_online_forecasts(self, get):
        # Simulate the bounded transport raising a requests-specific exception.
        import requests
        get.side_effect = requests.Timeout("offline")
        result = cloud_server.dashboard()
        self.assertNotEqual(result["source"], "github-pages")
        self.assertEqual(result["forecasts"], [])
        self.assertIn("niet bereikbaar", result["warning"])

    @patch.object(cloud_server.requests, "get")
    def test_shared_7_day_chart_uses_complete_online_history(self, get):
        response = Mock()
        response.json.return_value = {
            "coin": "BTC", "horizon": 7,
            "points": [
                {"date": "2026-10-13", "probUp": .60, "predReturn": .03, "actualReturn": .01},
                {"date": "2026-10-14", "probUp": .55, "predReturn": .02, "actualReturn": None},
                {"date": "2026-10-15", "probUp": .52, "predReturn": -.01, "actualReturn": -.03},
            ],
        }
        response.raise_for_status.return_value = None
        get.return_value = response
        result = cloud_server.history("BTC", 7)
        self.assertEqual(result["source"], "github-pages-oos")
        self.assertEqual(result["lastHistoryDate"], "2026-10-15")
        self.assertEqual(result["lastObservedTargetDate"], "2026-10-22")
        self.assertEqual(len(result["points"]), 2)
        self.assertIsNone(result["points"][-1]["actualPrice"])
        self.assertEqual(cloud_server.history("BTC", 7)["points"], result["points"])
        self.assertEqual(get.call_count, 1)

    @patch.object(cloud_server.requests, "get")
    def test_history_offline_does_not_forge_recent_points(self, get):
        import requests
        get.side_effect = requests.Timeout("offline")
        self.assertIsNone(cloud_server.load_shared_history("BTC", 7))
        with self.assertRaises(Exception):
            cloud_server.history("BTC", 7)

    def test_october_11_local_signal_blocked_on_october_26(self):
        import pandas as pd
        directory = Path(self.tmp.name)
        rows = [{
            "date": "2026-10-11", "coin": "BTC", "horizon": h,
            "model": "META", "prob_up": .85, "pred_ret": .08,
            "confidence_score": .99,
        } for h in (1, 7, 30, 90)]
        pd.DataFrame(rows).to_csv(directory / "latest_forecasts.csv", index=False)
        result = trading_strategy.build_strategy(
            directory, pd.DataFrame(), "BTC", current_position=0.,
            as_of_utc_date=date(2026, 10, 26),
        )
        self.assertEqual(result["modelDate"], "2026-10-11")
        self.assertEqual(result["signalAgeDays"], 15)
        self.assertEqual(result["signal"], "GEEN ACTUEEL SIGNAAL")
        self.assertFalse(result["signalFresh"])
        self.assertEqual(result["targetAllocation"], 0)
        self.assertIsNone(result["signalScore"])

    def test_fresh_online_signal_uses_same_forecast_values(self):
        import pandas as pd
        directory = Path(self.tmp.name)
        latest = {
            h: {
                "date": "2026-10-26", "prob_up": .85,
                "pred_ret": .08, "confidence": .80,
            }
            for h in (1, 7, 30, 90)
        }
        result = trading_strategy.build_strategy(
            directory, pd.DataFrame(), "BTC", current_position=0.,
            latest_override=latest, signal_source="github-pages",
            as_of_utc_date=date(2026, 10, 26),
        )
        self.assertTrue(result["signalFresh"])
        self.assertEqual(result["signalSource"], "github-pages")
        self.assertEqual(result["modelDate"], "2026-10-26")
        self.assertIn(result["signal"], ("KOPEN", "WACHTEN"))
        self.assertFalse(result["signalHistoryComparable"])

    @patch.object(cloud_server, "load_shared_forecast")
    def test_strategy_endpoint_uses_online_data(self, shared):
        today = datetime.now(timezone.utc).date().isoformat()
        shared.return_value = {
            "source": "github-pages",
            "updatedAt": today,
            "forecasts": [
                {"coin": "BTC", "horizon": h, "date": today,
                 "probUp": .84, "predReturn": .07, "confidenceScore": .92}
                for h in (1, 7, 30, 90)
            ],
        }
        result = cloud_server.strategy(coin="BTC", current_position=0.)
        self.assertTrue(result["signalFresh"])
        self.assertEqual(result["signalSource"], "github-pages")
        self.assertEqual(result["modelDate"], today)
        self.assertNotEqual(result["signal"], "GEEN ACTUEEL SIGNAAL")

    def test_stale_local_portfolio_recommendation_is_blocked(self):
        import pandas as pd
        directory = Path(self.tmp.name) / "output"
        directory.mkdir(parents=True, exist_ok=True)
        old = (datetime.now(timezone.utc).date() - timedelta(days=15)).isoformat()
        rows = [
            {"date": old, "coin": coin, "horizon": horizon,
             "model": "META", "prob_up": 0.8, "pred_ret": .05}
            for coin in ("BTC", "ETH") for horizon in (1, 7, 30, 90)
        ]
        pd.DataFrame(rows).to_csv(directory / "latest_forecasts.csv", index=False)
        result = cloud_server.portfolio_strategy_endpoint()
        self.assertFalse(result["available"])
        self.assertIn("uitgeschakeld", result["reason"])

    def test_local_trade_model_date_is_preserved(self):
        import pandas as pd
        directory = Path(self.tmp.name)
        pd.DataFrame([{
            "date": "2026-10-25", "coin": "BTC", "horizon": 7,
            "model": "META", "prob_up": 0.50, "pred_ret": 0.01,
        }]).to_csv(directory / "latest_forecasts.csv", index=False)
        d = trading_strategy._load_latest(directory, "BTC")
        self.assertEqual(d[7]["date"], "2026-10-25")

    def test_7_day_weighting_never_sees_unmatured_outcomes(self):
        import pandas as pd
        dates = pd.date_range("2026-10-01", periods=15, freq="D")
        bt = pd.DataFrame({
            "pred_ret": [0.01]*15,
            "actual_ret": [0.10]*15,
            "prob_up": [0.6]*15,
            "actual_up": [1]*15,
        }, index=dates)
        hist = matured_history(bt, pd.Timestamp("2026-10-15"), 7, 730)
        self.assertEqual(hist.index.max(), pd.Timestamp("2026-10-08"))
        self.assertNotIn(pd.Timestamp("2026-10-09"), hist.index)
        hist_tomorrow = matured_history(bt, pd.Timestamp("2026-10-16"), 7, 730)
        self.assertEqual(hist_tomorrow.index.max(), pd.Timestamp("2026-10-09"))
        # The same maturity check is used for historical and latest expert weighting.
        source = (Path(__file__).resolve().parents[1] / "modeling.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("matured_history(bt, dt, horizon, cfg.recent_weight_window_days)"), 2)

    def test_meta_predictions_ignore_unmatured_outcomes(self):
        """Changing outcomes that are still in the future must not change today's weights."""
        import pandas as pd
        import modeling
        from config import Config

        dates = pd.date_range("2026-06-01", periods=125, freq="D")
        cfg = Config()
        def expert(prob, predicted_return):
            return pd.DataFrame({
                "prob_up": [prob] * len(dates),
                "pred_ret": [predicted_return] * len(dates),
                "actual_up": [1] * len(dates),
                "actual_ret": [0.03] * len(dates),
                "low80": [-0.1] * len(dates),
                "high80": [0.1] * len(dates),
            }, index=dates)

        a = expert(0.8, 0.04)
        b = expert(0.2, -0.04)
        baseline = modeling.meta_ensemble({"A": a, "B": b}, cfg, horizon=7)
        a_changed, b_changed = a.copy(), b.copy()
        a_changed.loc[dates[-6:], "actual_up"] = 0
        b_changed.loc[dates[-6:], "actual_up"] = 0
        a_changed.loc[dates[-6:], "actual_ret"] = -0.50
        b_changed.loc[dates[-6:], "actual_ret"] = -0.50
        modified = modeling.meta_ensemble({"A": a_changed, "B": b_changed}, cfg, horizon=7)
        self.assertEqual(baseline.iloc[-1].prob_up, modified.iloc[-1].prob_up)
        self.assertEqual(baseline.iloc[-1].pred_ret, modified.iloc[-1].pred_ret)

        lp = [
            {"date": dates[-1], "expert": "A", "prob_up": 0.8,
             "pred_ret": 0.04, "low80": -0.1, "high80": 0.1},
            {"date": dates[-1], "expert": "B", "prob_up": 0.2,
             "pred_ret": -0.04, "low80": -0.1, "high80": 0.1},
        ]
        current = modeling.latest_meta(lp, {"A": a, "B": b}, cfg, horizon=7)
        changed = modeling.latest_meta(lp, {"A": a_changed, "B": b_changed}, cfg, horizon=7)
        self.assertEqual(current["prob_up"], changed["prob_up"])
        self.assertEqual(current["pred_ret"], changed["pred_ret"])

    def test_ui_clearly_discloses_sources_and_currencies(self):
        root = Path(__file__).resolve().parents[1]
        online = (root / "pages_site/index.html").read_text(encoding="utf-8")
        local = (root / "local_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("currency:'USD'", online)
        self.assertIn("Kans op daling", online)
        self.assertIn("forecastOrigin", local)
        self.assertIn("kans op daling", local)
        self.assertIn("actuele gedeelde online voorspellingen", local)
        self.assertIn("Geen actueel handelsadvies", local)
        self.assertIn("lastHistoryDate", local)
        self.assertIn("historyFreshness", local)
        self.assertIn("const validNumber=", local)


if __name__ == "__main__":
    unittest.main()
