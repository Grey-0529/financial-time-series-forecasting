from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


Order = Tuple[int, int, int]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)



def chronological_split(series: pd.Series, train_ratio: float, val_ratio: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]
    return train, val, test



def rolling_forecast(train: pd.Series, test: pd.Series, order: Order) -> np.ndarray:
    history = train.astype(float).tolist()
    predictions = []
    for actual in test.astype(float):
        model = ARIMA(history, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=1)
        pred = float(forecast[0])
        predictions.append(pred)
        history.append(float(actual))
    return np.array(predictions)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single-ticker ARIMA baseline for next-day closing price forecasting."
    )
    parser.add_argument("--input", required=True, help="Path to raw or cleaned CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--ticker", required=True, help="Ticker to filter, e.g. AAPL.")
    parser.add_argument("--date_col", default="Date", help="Date column name.")
    parser.add_argument("--ticker_col", default="Ticker", help="Ticker column name.")
    parser.add_argument(
        "--target_col",
        default="Next_Day_Close",
        help="Target series column used by ARIMA. Default is Next_Day_Close.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--p", type=int, default=5, help="ARIMA p parameter.")
    parser.add_argument("--d", type=int, default=1, help="ARIMA d parameter.")
    parser.add_argument("--q", type=int, default=0, help="ARIMA q parameter.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {args.date_col, args.target_col}
    if args.ticker_col in df.columns:
        required.add(args.ticker_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    if args.ticker_col in df.columns:
        df = df[df[args.ticker_col] == args.ticker].copy()
    if df.empty:
        raise ValueError(f"No rows found for ticker '{args.ticker}'.")

    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df.dropna(subset=[args.date_col, args.target_col]).sort_values(args.date_col).reset_index(drop=True)

    series = df[args.target_col].astype(float)
    dates = df[args.date_col]

    train, val, test = chronological_split(series, args.train_ratio, args.val_ratio)
    _, _, test_dates = chronological_split(dates, args.train_ratio, args.val_ratio)

    history_series = pd.concat([train, val], axis=0)
    order: Order = (args.p, args.d, args.q)
    preds = rolling_forecast(history_series, test, order=order)

    rmse = float(sqrt(mean_squared_error(test, preds)))
    mae = float(mean_absolute_error(test, preds))
    mape_value = float(mape(test.to_numpy(), preds))

    pred_df = pd.DataFrame(
        {
            args.date_col: test_dates.dt.strftime("%Y-%m-%d").to_list(),
            "actual": test.to_numpy(dtype=float),
            "predicted": preds,
        }
    )
    pred_df.to_csv(output_dir / f"arima_predictions_{args.ticker}.csv", index=False)

    metrics = {
        "model": "ARIMA",
        "ticker": args.ticker,
        "target_col": args.target_col,
        "order": {"p": args.p, "d": args.d, "q": args.q},
        "rmse": rmse,
        "mae": mae,
        "mape": mape_value,
        "train_points": int(len(train)),
        "val_points": int(len(val)),
        "test_points": int(len(test)),
    }
    with open(output_dir / f"arima_metrics_{args.ticker}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, test.to_numpy(dtype=float), label="Actual")
    plt.plot(test_dates, preds, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel(args.target_col)
    plt.title(f"ARIMA Actual vs Predicted ({args.ticker}, {args.target_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"arima_actual_vs_predicted_{args.ticker}.png", dpi=200)
    plt.close()

    print("ARIMA baseline completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
