from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    # Forecasts
    coins: Dict[str, str] = field(default_factory=lambda: {"BTC": "BTC-USD", "ETH": "ETH-USD"})
    horizons: List[int] = field(default_factory=lambda: [1, 7, 30, 90])
    history_years: int = 10
    min_train_years: int = 4
    retrain_every_days: int = 60
    validation_days: int = 365
    recent_weight_window_days: int = 730
    transaction_cost: float = 0.001

    # Prediction semantics: forecast is made after the UTC daily crypto close.
    # Exogenous sources are delayed to avoid using information not safely known then.
    exogenous_lag_days: int = 1
    onchain_lag_days: int = 1
    news_lag_days: int = 1
    etf_lag_days: int = 1
    trends_lag_days: int = 7

    # Compute mode: "balanced" or "deep".
    # Deep adds XGBoost when installed and increases tree counts.
    mode: str = "balanced"

    # Minimum observations for an expert to participate.
    min_expert_rows: int = 800
    min_validation_rows: int = 180

    # Ensemble thresholds
    long_threshold: float = 0.57
    strong_long_threshold: float = 0.64
    risk_off_threshold: float = 0.43

    # Optional data sources
    enable_news: bool = True
    enable_google_trends: bool = False       # unofficial pytrends; off by default
    enable_etf_flows: bool = True
    enable_binance_derivatives: bool = True
    enable_fear_greed: bool = True
    enable_coinmetrics: bool = True
    enable_fred: bool = True
    enable_cross_assets: bool = True

    # Optional historical files supplied by the user.
    # CSV format: first column "date"; all remaining numeric columns become features.
    custom_feature_files: List[str] = field(default_factory=list)

    # Research lockbox: report the last N OOS days separately.
    lockbox_days: int = 365

DEFAULT_CONFIG = Config()
