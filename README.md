# BTC Algorithmic Trading Research
### End-to-end quantitative strategy research pipeline on BTC/USD M1

---

## Overview

This project documents a complete quantitative trading research cycle — from raw market data to backtesting, signal filtering with machine learning, and a rigorous out-of-sample evaluation. The strategy investigated is **The Trident**: a 3-step engulfing candle pattern anchored to the Hull Moving Average color cycle.

The honest conclusion: the strategy has insufficient edge on M1 BTC for live prop firm trading. This repo shows the full research process, including why that conclusion was reached and how it was validated.

---

## Research Pipeline

```
Bybit API → Data Pipeline → Indicators → Backtest Engine → ML Filter → OOS Validation
   ↓              ↓              ↓               ↓               ↓              ↓
Raw M1 data   Clean CSV     Pine Script    Event-driven    CatBoost      Challenge
via REST API  + M5 resample  TradingView   bar-by-bar      classifier    simulation
```

---

## The Trident Strategy

### Concept
A directional state machine that detects 3-step engulfing candle patterns signalling potential trend reversals, anchored to Hull MA color cycles.

### Long Pattern (BEC1 → BEC2 → BEC3)
```
BEC1: Bullish engulfing candle + close below Hull + Hull RED
      ↓ Hull turns GREEN (one cycle allowed)
BEC2: Bullish engulfing + lower low than BEC1 (no Hull restriction)
      ↓ Hull must stay GREEN
BEC3: Bullish engulfing + close above BEC2 high + Hull GREEN → ENTRY
```

Short pattern is the exact mirror. Any Hull color violation resets the entire machine.

### Engulfing Detection
Original HPotter body-based logic — no high/low range requirement:
```python
bull_engulf = (prev_red and this_green
               and close >= open[1]       # close at/above prev open
               and close[1] >= open       # prev close at/above this open
               and body > prev_body)      # current body strictly larger
```

### Entry / Risk
```
Entry : Signal candle close ± offset (market order)
SL    : Signal candle low + 1.5 × ATR(14)
TP    : 2:1 RR from actual SL distance
Risk  : Fixed $50 per trade
```

---

## Results

### Baseline (no filter)
| Metric | Value |
|---|---|
| Total trades | 2,523 |
| Win rate | 32.74% |
| Profit factor | 0.97 |
| Max drawdown | -73% (limits disabled) |
| Trades/month | ~62 |

### With CatBoost ML Filter (threshold=0.525)
| Metric | Training (2023–2025) | Out-of-Sample (2025–2026) |
|---|---|---|
| Total trades | 710 | 203 |
| Win rate | ~55% | 33.5% |
| Profit factor | ~2.0 | ~1.1 |
| Challenge passed | ✅ 3 months | ❌ Not passed (8 months) |

**Key finding:** The training period performance was partially driven by the `month` feature encoding specific calendar periods. After removing it and applying TimeSeriesSplit cross-validation, the honest out-of-sample AUC was 0.53 — weak but consistent signal.

---

## Notebooks

### `00_data_pipeline.ipynb`
Downloads BTC/USDT M1 OHLCV data from the Bybit public REST API, validates data quality, and resamples to M5 for the HTF Hull filter.

**Demonstrates:**
- API pagination and rate limit handling
- Data quality checks (duplicates, OHLCV sanity, gap analysis)
- Time series resampling with validation

### `01_trident_backtest.ipynb`
Event-driven bar-by-bar backtester with the full Trident state machine, feature logging, and challenge simulation.

**Demonstrates:**
- Event-driven backtesting (no vectorised look-ahead bias)
- 4-state machine implementation preventing same-bar bleed
- Prop firm guardrails (daily/total loss limits)
- Feature extraction at signal time (no data leakage)
- Challenge simulation with monthly P&L tracking

### `02_features.ipynb`
CatBoost signal classifier trained on the trade log, with Optuna hyperparameter tuning and TimeSeriesSplit cross-validation.

**Demonstrates:**
- Feature engineering on financial time series
- Class imbalance handling with CatBoost weights
- TimeSeriesSplit CV to detect temporal overfitting
- Optuna Bayesian hyperparameter optimisation
- Probability threshold sweep for trade filtering
- Feature importance analysis

---

## Key Learnings

**1. The state machine bug that cost weeks**
Using separate `if` blocks instead of `if/else if` chains caused the state machine to reset and immediately re-enter on the same bar — effectively surviving multiple Hull cycles instead of resetting. Fixed by converting to a single `if/else if` chain.

**2. Data leakage from the `month` feature**
Including calendar month as a feature caused the model to memorise specific periods seen in training (e.g. "March 2023 was a good month"). AUC jumped to 0.57 with it, dropped to 0.53 without. The 0.53 is the honest number.

**3. Linear correlation ≠ predictive power**
All features showed linear correlation < 0.05 with the win/loss label. This doesn't mean ML can't find signal — CatBoost finds non-linear combinations. But when even non-linear signal is weak (AUC 0.53), the strategy itself needs rethinking.

**4. Loss diagnosis before changing parameters**
Before tuning, we diagnosed WHY trades were losing — 91% wrong direction vs 9% SL too tight. This prevented wasting time widening the SL, which would only have increased loss size. The problem was in the signal, not the risk management.

**5. Regime dependency**
TimeSeriesSplit CV revealed AUC ranging from 0.48 to 0.56 across folds — the model works in trending markets (BTC bull runs) and fails in ranging/correcting periods. This is a fundamental strategy limitation, not a model limitation.

---

## Tech Stack

```
Python 3.11      pandas, numpy, plotly
CatBoost         gradient boosting classifier
Optuna           Bayesian hyperparameter optimisation
scikit-learn     TimeSeriesSplit, metrics
requests         Bybit REST API client
TradingView      Pine Script v6 indicators
```

---

## Repository Structure

```
btc-algo-trading/
├── notebooks/
│   ├── 00_data_pipeline.ipynb     # API download, clean, resample
│   ├── 01_trident_backtest.ipynb  # backtester + challenge simulation
│   └── 02_features.ipynb          # CatBoost ML filter
├── indicators/
│   ├── TheTrident.pine            # strict HPotter engulfing, one Hull cycle
│   └── ReTriForce.pine            # range-based engulfing, flexible Hull
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/mad52XD/btc-algo-trading.git
cd btc-algo-trading
pip install -r requirements.txt
```

Run notebooks in order: `00` → `01` → `02`

Data is downloaded automatically in `00_data_pipeline.ipynb` via the Bybit public API — no API key required.

---

## Why This Strategy Was Not Deployed

The out-of-sample evaluation showed:
- Win rate of 33.5% on the test period (Sep 2025 – May 2026)
- Challenge not passed after 8 months of simulated trading
- High monthly variance (8% WR in bad months, 75% in good months)
- 91% of losing trades had price continuing in the wrong direction — a signal quality problem, not a parameter problem

A separate SMC (Smart Money Concepts) strategy showed 55.97% win rate and 4.95 profit factor on the same dataset and is deployed live. The Trident research informed that project's development.

---

## License
MIT
