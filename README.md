# Financial Time Series Forecasting

## Overview

This repository contains the ARIMA baseline and tuning scripts from a group project on financial time series forecasting. The project compares ARIMA, LSTM, and GRU for next-day closing price prediction on AAPL, MSFT, and NVDA under the same experimental setting.

The repository focuses on the ARIMA implementation and includes the final group presentation in the `docs/` folder to summarize the full model comparison, including ARIMA, LSTM, and GRU results.

## Project Attribution

This repository is based on a group project completed during my Master's study in Data Science. The project was developed collaboratively by our team.

My main contributions included dataset selection, data description, preprocessing pipeline explanation, ARIMA baseline revision, result interpretation, and presentation preparation. This repository is used as a portfolio record to demonstrate my participation in the project and understanding of the end-to-end time series forecasting workflow.

## Research Objective

The project aims to answer the following questions:

* Can ARIMA, LSTM, and GRU predict next-day closing prices effectively?
* Which model performs better on AAPL, MSFT, and NVDA under the same forecasting setup?
* How does stock volatility affect forecasting performance and model stability?

## Dataset

The project uses FAANG-related historical stock price data with technical indicators.

Selected tickers:

* AAPL
* MSFT
* NVDA

Main feature groups include:

* Price and volume features
* Moving averages
* Momentum indicators
* Volatility-related indicators

The forecasting target is `Next_Day_Close`.

## Methods

### ARIMA

ARIMA is used as a classical statistical baseline for next-day closing price forecasting. The scripts support rolling one-step forecasting and evaluation using RMSE, MAE, and MAPE.

### LSTM and GRU

LSTM and GRU were included in the final group comparison as deep learning sequence models. Their results and discussion are summarized in the final presentation under the `docs/` folder.

## Key Features

* Prepared a next-day stock price forecasting task for AAPL, MSFT, and NVDA
* Used chronological train / validation / test splitting to avoid data leakage
* Implemented ARIMA baseline forecasting for individual tickers
* Implemented batch ARIMA tuning across multiple tickers
* Evaluated model performance using RMSE, MAE, and MAPE
* Generated prediction CSV files, metrics JSON files, and actual-vs-predicted plots
* Summarized full ARIMA / LSTM / GRU comparison in the final group presentation

## Results Summary

In the final group presentation, ARIMA achieved the strongest and most stable performance among the compared models.

| Ticker | Best ARIMA Order | RMSE |  MAE |  MAPE |
| ------ | ---------------: | ---: | ---: | ----: |
| AAPL   |          (3,1,0) | 3.89 | 2.60 | 1.16% |
| MSFT   |          (5,1,0) | 6.26 | 4.43 | 1.00% |
| NVDA   |          (3,1,0) | 4.15 | 3.10 | 2.24% |

The results suggest that a well-designed classical baseline can be highly competitive in a short-term one-step forecasting task. In this project, LSTM and GRU did not outperform ARIMA under the current data setup and hyperparameter settings.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── faang_stock_prices.csv
├── src/
│   ├── arima_baseline_nextday_fixed.py
│   └── arima_tuning_batch_nextday_fixed.py
├── docs/
│   └── stock_price_prediction_presentation.pdf
└── results/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run ARIMA baseline for a single ticker:

```bash
python src/arima_baseline_nextday_fixed.py \
  --input data/faang_stock_prices.csv \
  --output_dir results/AAPL \
  --ticker AAPL
```

Run ARIMA tuning for multiple tickers:

```bash
python src/arima_tuning_batch_nextday_fixed.py \
  --input data/faang_stock_prices.csv \
  --output_dir results/arima_batch \
  --tickers AAPL MSFT NVDA \
  --candidate_orders "3,1,0;5,1,0"
```

The scripts will save prediction results, metrics files, and actual-vs-predicted plots under the `results/` folder.

## Outputs

The scripts generate:

* Prediction CSV files
* Metrics JSON files
* Actual-vs-predicted plots
* Batch summary CSV / JSON files

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Statsmodels
* Time Series Forecasting
* ARIMA

## What I Learned

Through this project, I practiced financial time series preprocessing, chronological data splitting, ARIMA baseline modeling, rolling one-step forecasting, model evaluation, and result interpretation. I also learned how to compare classical statistical methods with deep learning sequence models under a fair experimental setup.
