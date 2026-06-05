from __future__ import annotations

import argparse
import json
import warnings
from math import sqrt
from pathlib import Path
from typing import Iterable, List, Tuple

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



def chronological_split(values: np.ndarray, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return values[:train_end], values[train_end:val_end], values[val_end:]



def parse_orders(order_string: str) -> List[Order]:
    orders: List[Order] = []
    for chunk in order_string.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        p_str, d_str, q_str = [x.strip() for x in chunk.split(",")]
        orders.append((int(p_str), int(d_str), int(q_str)))
    if not orders:
        raise ValueError("No valid ARIMA orders were provided.")
    return orders



def rolling_forecast_append(train_values: np.ndarray, test_values: np.ndarray, order: Order) -> np.ndarray:
    warnings.filterwarnings("ignore")
    result = ARIMA(train_values, order=order).fit()
    predictions = []
    for actual in test_values:
        pred = float(result.forecast(steps=1)[0])
        predictions.append(pred)
        result = result.append([float(actual)], refit=False)
    return np.array(predictions)



def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
    }



def save_prediction_plot(
    dates: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    ticker: str,
    target_col: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.title(f"ARIMA Actual vs Predicted ({ticker}, {target_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"arima_actual_vs_predicted_{ticker}.png", dpi=200)
    plt.close()



def run_for_one_ticker(
    df_all: pd.DataFrame,
    ticker: str,
    date_col: str,
    ticker_col: str,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    candidate_orders: Iterable[Order],
    output_root: Path,
) -> dict:
    df = df_all[df_all[ticker_col] == ticker].copy()
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' not found.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    values = df[target_col].to_numpy(dtype=float)
    dates = df[date_col].reset_index(drop=True)
    train_values, val_values, test_values = chronological_split(values, train_ratio, val_ratio)
    _, val_dates, test_dates = chronological_split(dates.to_numpy(), train_ratio, val_ratio)

    ticker_dir = output_root / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    validation_rows = []
    best_order = None
    best_val_rmse = None
    for order in candidate_orders:
        val_pred = rolling_forecast_append(train_values, val_values, order)
        metrics = evaluate_predictions(val_values, val_pred)
        validation_rows.append(
            {
                "ticker": ticker,
                "stage": "validation",
                "target_col": target_col,
                "order": str(order),
                **metrics,
            }
        )
        if best_val_rmse is None or metrics["rmse"] < best_val_rmse:
            best_val_rmse = metrics["rmse"]
            best_order = order

    pd.DataFrame(validation_rows).to_csv(ticker_dir / f"arima_validation_orders_{ticker}.csv", index=False)

    history = np.concatenate([train_values, val_values])
    test_pred = rolling_forecast_append(history, test_values, best_order)
    test_metrics = evaluate_predictions(test_values, test_pred)

    pred_df = pd.DataFrame(
        {
            date_col: pd.to_datetime(test_dates).strftime("%Y-%m-%d"),
            "actual": test_values,
            "predicted": test_pred,
        }
    )
    pred_df.to_csv(ticker_dir / f"arima_predictions_{ticker}.csv", index=False)
    save_prediction_plot(
        dates=pd.Series(pd.to_datetime(test_dates)),
        actual=test_values,
        predicted=test_pred,
        ticker=ticker,
        target_col=target_col,
        output_dir=ticker_dir,
    )

    result = {
        "ticker": ticker,
        "target_col": target_col,
        "best_order": {"p": best_order[0], "d": best_order[1], "q": best_order[2]},
        **test_metrics,
        "train_points": int(len(train_values)),
        "val_points": int(len(val_values)),
        "test_points": int(len(test_values)),
    }
    with open(ticker_dir / f"arima_metrics_{ticker}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune and run ARIMA baselines for multiple tickers using Next_Day_Close by default."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker list, e.g. AAPL MSFT NVDA.")
    parser.add_argument("--date_col", default="Date", help="Date column name.")
    parser.add_argument("--ticker_col", default="Ticker", help="Ticker column name.")
    parser.add_argument(
        "--target_col",
        default="Next_Day_Close",
        help="Target series column used by ARIMA. Default is Next_Day_Close.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument(
        "--candidate_orders",
        default="3,1,0;5,1,0",
        help="Semicolon-separated ARIMA orders, e.g. '3,1,0;5,1,0;5,1,1'",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_orders = parse_orders(args.candidate_orders)

    df_all = pd.read_csv(args.input)
    required = {args.date_col, args.ticker_col, args.target_col}
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_results = []
    for ticker in args.tickers:
        print(f"Running ARIMA tuning for {ticker}...")
        result = run_for_one_ticker(
            df_all=df_all,
            ticker=ticker,
            date_col=args.date_col,
            ticker_col=args.ticker_col,
            target_col=args.target_col,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            candidate_orders=candidate_orders,
            output_root=output_root,
        )
        all_results.append(result)
        print(json.dumps(result, indent=2))

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_root / "arima_batch_summary.csv", index=False)
    with open(output_root / "arima_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("ARIMA batch tuning completed.")
    print(summary_df)


if __name__ == "__main__":
    main()
