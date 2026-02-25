# Portfolio Optimization on S&P 500 Stocks — Predicting Returns with ML

In the [first article](https://medium.com/@alexandre.durand/portfolio-optimisation-on-s-p-500-stocks-46f03732b030), we explored the theoretical foundations of portfolio optimization.
In the [second article](https://medium.com/@alexandre.durand/portfolio-optimization-on-s-p-500-stocks-with-backtest-61da87ed91ff), we backtested three portfolio strategies using past quarterly returns as our "prediction" — essentially a momentum bet.

The obvious weakness? Using last quarter's return to predict next quarter's return is a rough heuristic at best. In this third notebook, we build actual predictive models — **Linear Regression** and **Gradient Boosted Trees** — trained on a set of features to forecast quarterly log returns. We then plug those predictions into the same portfolio optimization framework and compare against the momentum baseline from article 2.

## Contents:

**Feature Engineering:** Build predictive features from price and volume data.

**Model Training:** Fit Linear Regression and Gradient Boosted Trees on historical data.

**Return Prediction:** Generate predicted quarterly log returns on test period.

**Portfolio Allocation & Backtest:** For each of the 3 prediction methods (Previous Quarter Return, Linear Regression, GBT), allocate weights using 4 strategies — Random, Max Sharpe Ratio, Min Variance, and Markowitz Mean-Variance.

**Comparison:** 3 × 4 = 12 strategy combinations, plus analysis of which prediction + allocation pairing works best.

---

## Feature Engineering

The idea is simple — give the model enough historical signal to form a view on next quarter's return. We compute the following features at each rebalancing date, per ticker:

| Feature | Definition |
|---|---|
| `ret_q1` | Quarterly log return, lag 1 (previous quarter) |
| `ret_q2` | Quarterly log return, lag 2 |
| `ret_q4` | Quarterly log return, lag 4 (1 year ago) |
| `volatility_63d` | Rolling 63-day std of daily log returns |
| `volatility_252d` | Rolling 252-day std of daily log returns |
| `mom_12_1` | 12-month return minus last month return (classic momentum signal) |
| `volatility_ratio` | Ratio of 63d / 252d volatility (volatility regime indicator) |
| `mean_reversion` | Deviation of current price from 252d moving average |

These are standard quantitative finance features — nothing exotic, but they capture momentum, mean-reversion, volatility regimes and relative activity shifts.

```python
# Calculate daily log returns — per ticker to avoid cross-ticker contamination
df['Log_Return'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: np.log(x / x.shift(1)))

days = 63 # quarterly rebalancing window
df['Quarterly_Log_Return'] = df.groupby('Ticker')['Log_Return']\
    .rolling(window=days, min_periods=days).sum()\
    .reset_index(0, drop=True)

# Features per ticker
def compute_features(group):
    g = group.sort_values('Date').copy()

    # Lagged quarterly returns
    g['ret_q1'] = g['Quarterly_Log_Return'].shift(days)
    g['ret_q2'] = g['Quarterly_Log_Return'].shift(days * 2)
    g['ret_q4'] = g['Quarterly_Log_Return'].shift(days * 4)

    # Volatility
    g['volatility_63d'] = g['Log_Return'].rolling(63).std()
    g['volatility_252d'] = g['Log_Return'].rolling(252).std()

    # Momentum 12-1
    ret_12m = g['Log_Return'].rolling(252).sum()
    ret_1m = g['Log_Return'].rolling(21).sum()
    g['mom_12_1'] = ret_12m - ret_1m

    # Volatility ratio (short-term vs long-term vol regime)
    g['volatility_ratio'] = g['volatility_63d'] / g['volatility_252d']

    # Mean reversion signal
    g['mean_reversion'] = g['Adj Close'] / g['Adj Close'].rolling(252).mean() - 1

    return g

df = df.groupby('Ticker', group_keys=False).apply(compute_features)
```

Quick sanity check — let's look at the correlation of individual features with future quarterly return:

```python
feature_cols = ['ret_q1','ret_q2','ret_q4','volatility_63d',
                'volatility_252d','mom_12_1','volatility_ratio','mean_reversion']

# Target : next quarter return
df['target'] = df.groupby('Ticker')['Quarterly_Log_Return'].shift(-days)

corr = df[feature_cols + ['target']].corr()['target'].drop('target')
display(corr.sort_values(ascending=False))
```

![Feature Correlation with Future Quarterly Return](images/article3_feature_correlation.png)

The correlations are small (as expected in financial data), but not zero. `volatility_252d` (+0.114) and `volatility_63d` (+0.067) show the strongest positive correlation with future returns — higher-vol stocks tend to deliver higher returns (the classic risk premium). `ret_q4` is mildly positive. On the negative side, `mom_12_1` (−0.031) and `ret_q1` (−0.027) suggest short-term reversal effects.

---

## Split Data

Same split logic as article 2 : Train / Valid / Test. The covariance matrix is estimated on the validation set. Models are trained on training data only.

```python
dates = list(df['Date'].unique())

dates_train = dates[:int(len(dates) * 0.7)]
dates_valid = dates[int(len(dates) * 0.7) : int(len(dates) * 0.85)]
dates_test  = dates[int(len(dates) * 0.85):]
dates_test_rebalance = dates_test[0::days]  # 1 rebalancing every 63 days

train = df[df['Date'].isin(dates_train)].dropna(subset=feature_cols + ['target'])
valid = df[df['Date'].isin(dates_valid)]
test  = df[df['Date'].isin(dates_test)]
```

Output :

```
train :  1411253 rows | 2006-02-24 -> 2020-02-14
valid :   379262 rows | 2020-02-18 -> 2023-02-13
test  :   379765 rows | 2023-02-14 -> 2026-02-18
test rebalancing dates : 12
```

So we train on ~14 years of data, validate on ~3 years, and test on the most recent ~3 years (2023–2026). This test window covers the post-inflation recovery, the 2023–2024 AI-driven rally, and the 2025 consolidation — a fairly turbulent stretch.

Covariance matrice on validation data (same as article 2) :

```python
pivot_returns_valid = valid.pivot_table(values='Quarterly_Log_Return',
                                         columns='Ticker', index='Date').fillna(0)
matrix_covariance = calculate_shrink_cov_matrix(pivot_returns_valid)
matrix_covariance = pd.DataFrame(matrix_covariance,
                                  columns=pivot_returns_valid.columns,
                                  index=pivot_returns_valid.columns)
```

---

## Model Training

### Linear Regression

Nothing fancy. We standardize features (important for regularization stability) and fit a simple OLS. We could add Ridge or Lasso but let's keep it minimal first.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])
y_train = train['target'].values

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Feature importance (coefficients)
coef_df = pd.DataFrame({'feature': feature_cols, 'coef': lr_model.coef_})
display(coef_df.sort_values('coef', ascending=False))

print(f"Train R² : {lr_model.score(X_train, y_train):.4f}")
```

![Linear Regression — Standardized Coefficients](images/article3_lr_coefficients.png)

**Train R² = 0.0195** — about 1.9%. The R² is low, as expected for quarterly stock return predictions. Don't be alarmed. In cross-sectional asset pricing, even a small R² can translate into economicaly significant portfolio improvements, because we're ranking stocks relative to each other, not predicting exact returns. The largest positive coefficient is `volatility_252d` (long-term vol premium), followed by `volatility_ratio` and `ret_q4`. Negative coefficients on `volatility_63d` and `ret_q1` suggest that short-term high-vol and recent winners tend to revert.

### Gradient Boosted Trees

`GradientBoostingRegressor` from sklearn is well suited here because it handles non-linear interactions between features (e.g. momentum behaves differently in high vs low volatility regimes).

We use modest hyperparameters to avoid overfitting — financial data is notoriously noisy.

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    min_samples_leaf=50,
    random_state=42
)

gbr_model.fit(X_train, y_train)
print(f"GBR Train R² : {gbr_model.score(X_train, y_train):.4f}")
```

**GBR Train R² = 0.1026** — about 10%, which is substantially higher than the linear model. That's partly because the tree model can fit non-linear patterns, but also partly overfitting. The conservative hyperparameters (`min_samples_leaf=50`, `subsample=0.7`) help, but we should remain skeptical about out-of-sample performance.

Let's check feature importance :

```python
fi = pd.DataFrame({'feature': feature_cols, 'importance': gbr_model.feature_importances_})
fi = fi.sort_values('importance', ascending=True)
fi.plot(kind='barh', x='feature', y='importance')
plt.title('Gradient Boosted Trees — Feature Importance (Impurity)')
plt.show()
```

![Gradient Boosted Trees — Feature Importance](images/article3_gbr_feature_importance.png)

`volatility_ratio` dominates (≈0.21 importance), followed by `ret_q1` and `volatility_252d`. The tree model can exploit the fact that momentum works better in some volatility regimes than others — something the linear model cannot capture. Interestingly, `mom_12_1` ranks lowest in importance for the GBR, while `ret_q1` (which was negative in the linear model) is the second most important split feature. The tree model is finding non-linear structure in the interaction between recent returns and volatility regimes.

---

## Prediction Function

We wrap prediction in a helper that takes a date's feature data and returns predicted returns per ticker for both models.

```python
def predict_returns(df_date, feature_cols, scaler, lr_model, gbr_model):
    """Predict quarterly log returns for all tickers at a given date."""
    valid_mask = df_date[feature_cols].notna().all(axis=1)
    df_valid = df_date[valid_mask]
    if len(df_valid) == 0:
        return pd.DataFrame()

    X = df_valid[feature_cols].values
    X_scaled = scaler.transform(X)

    preds_lr  = lr_model.predict(X_scaled)
    preds_gbr = gbr_model.predict(X_scaled)

    return pd.DataFrame({
        'Ticker': df_valid['Ticker'].values,
        'pred_lr': preds_lr,
        'pred_gbr': preds_gbr
    }).set_index('Ticker')
```

---

## Portfolio Optimization Functions

Same as the previous articles — reused directly (see [article 1](https://medium.com/@alexandre.durand/portfolio-optimisation-on-s-p-500-stocks-46f03732b030) for detailled explanations).

```python
def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_portfolio_returns(weights, returns):
    return np.dot(weights, returns)

def neg_sharpe_ratio_objective(weights, returns, cov_matrix, risk_free_rate=0.03):
    portfolio_returns = np.squeeze(calculate_portfolio_returns(weights, returns))
    portfolio_variance = np.squeeze(calculate_portfolio_variance(weights, cov_matrix))
    return -((portfolio_returns - risk_free_rate) / np.sqrt(portfolio_variance))

def neg_markowitz_objective(weights, returns, cov_matrix, gamma=0.2):
    portfolio_returns = np.squeeze(calculate_portfolio_returns(weights, returns))
    portfolio_variance = np.squeeze(calculate_portfolio_variance(weights, cov_matrix))
    return gamma * portfolio_variance - portfolio_returns

def optimize_weights(log_returns, covariance_matrix, fun=neg_markowitz_objective, x0=None):
    number_of_tickers = len(log_returns)
    if x0 is None:
        x0 = np.array([1/number_of_tickers for _ in range(number_of_tickers)])
    if fun == calculate_portfolio_variance:
        args = (covariance_matrix,)
    else:
        args = (log_returns, covariance_matrix)
    result = sp_opt.minimize(
        fun=fun, args=args, x0=x0, method='SLSQP',
        bounds=tuple((0, 0.3) for _ in range(number_of_tickers)),
        constraints=({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    )
    return result.x
```

---

## Backtest Loop

This is where it gets interesting. For each rebalancing date in the test period, we:

1. **Predict returns** using 3 methods : previous quarter return (momentum), Linear Regression, Gradient Boosted Trees
2. **Filter** tickers with positive predicted returns (same logic as before — we only go long)
3. **Optimize weights** using 4 allocation strategies per prediction method :
   - **Random** : uniform random weights (sanity check baseline)
   - **Max Sharpe Ratio** : maximize risk-adjusted return
   - **Min Variance** : minimize portfolio variance (ignores predicted returns entirely)
   - **Markowitz Mean-Variance** : maximize return − γ × variance tradeoff
4. **Compute actual realised return** of the portfolio

This gives us 3 prediction × 4 allocation = 12 strategy combinations.

```python
pivot_returns_test = test_rebalance.pivot_table(
    values='Quarterly_Log_Return', columns='Ticker', index='Date').fillna(0)

np.random.seed(42)

results = {}
for idx in range(0, len(pivot_returns_test) - 1):
    date = pivot_returns_test.iloc[idx].name
    results[date] = {}
    tickers_returns_future = pivot_returns_test.iloc[idx + 1]   # forward return (next quarter)
    tickers_returns_momentum = pivot_returns_test.iloc[idx]     # momentum signal (last quarter)

    # Get ML predictions for this date
    df_date = test_rebalance[test_rebalance['Date'] == date]
    preds = predict_returns(df_date, feature_cols, scaler, lr_model, gbr_model)
    if len(preds) == 0:
        continue

    for pred_name, pred_series in [('momentum', tickers_returns_momentum),
                                     ('lr', preds['pred_lr']),
                                     ('gbr', preds['pred_gbr'])]:

        # Align tickers
        common_tickers = list(set(pred_series.index) & set(tickers_returns_future.index)
                              & set(matrix_covariance.columns))
        if len(common_tickers) < 5: continue

        pred_filtered = pred_series.loc[common_tickers]
        future_filtered = tickers_returns_future.loc[common_tickers]

        # Keep only positive predictions
        mask_positive = pred_filtered.values > 0
        if mask_positive.sum() < 5: continue

        t = np.array(common_tickers)[mask_positive]
        pred_pos = pred_filtered.values[mask_positive]
        future_pos = future_filtered.values[mask_positive]
        cov_filtered = matrix_covariance.loc[t, t].values

        # Random allocation
        w_random = np.random.rand(len(t))
        w_random = w_random / w_random.sum()
        results[date][f'returns_random_{pred_name}'] = (w_random * future_pos).sum()

        # Max Sharpe optimized
        w_sharpe = optimize_weights(pred_pos, cov_filtered, fun=neg_sharpe_ratio_objective)
        results[date][f'returns_sharpe_{pred_name}'] = (w_sharpe * future_pos).sum()

        # Min Variance optimized
        w_minvar = optimize_weights(pred_pos, cov_filtered, fun=calculate_portfolio_variance)
        results[date][f'returns_minvar_{pred_name}'] = (w_minvar * future_pos).sum()

        # Markowitz Mean-Variance optimized
        w_mv = optimize_weights(pred_pos, cov_filtered, fun=neg_markowitz_objective)
        results[date][f'returns_mv_{pred_name}'] = (w_mv * future_pos).sum()
```

Backtest runs over 11 quarterly periods (mid-2023 to end-2025).

---

## Results Analysis

Let's plot the cumulative returns for all strategies :

![Cumulative Quarterly Log Returns — Momentum vs ML Predictions](images/article3_cumulative_returns.png)

All strategies end up positive (except one) — the 2023–2026 test window coincided with a strong bull market. `mv_momentum` dominates on raw return, reaching ~1.47 cumulative log return. But the spread between strategies and allocation methods is what matters.

Now the key comparisons — total returns and realised Sharpe ratios across all 12 strategies :

| Strategy | Total Return | Avg Q Return | Std Q Return | Realised Sharpe | Max Q Drawdown | Best Q Return |
|---|---|---|---|---|---|---|
| mv_momentum | 1.4663 | 0.1333 | 0.1807 | 0.7378 | −0.1659 | 0.3669 |
| minvar_gbr | 0.2321 | 0.0211 | 0.0335 | 0.6308 | −0.0099 | 0.0786 |
| minvar_lr | 0.2272 | 0.0207 | 0.0375 | 0.5512 | −0.0158 | 0.0782 |
| sharpe_lr | 0.3370 | 0.0306 | 0.0563 | 0.5438 | −0.0870 | 0.1445 |
| random_gbr | 0.2483 | 0.0226 | 0.0485 | 0.4658 | −0.0565 | 0.1127 |
| random_momentum | 0.2486 | 0.0226 | 0.0493 | 0.4587 | −0.0594 | 0.1231 |
| random_lr | 0.2312 | 0.0210 | 0.0502 | 0.4188 | −0.0609 | 0.1094 |
| minvar_momentum | 0.2133 | 0.0194 | 0.0567 | 0.3421 | −0.0488 | 0.1217 |
| mv_gbr | 0.8678 | 0.0789 | 0.2378 | 0.3317 | −0.2426 | 0.5924 |
| sharpe_gbr | 0.5645 | 0.0513 | 0.1851 | 0.2772 | −0.2935 | 0.4004 |
| sharpe_momentum | 0.2005 | 0.0182 | 0.0793 | 0.2300 | −0.1058 | 0.1255 |
| mv_lr | −0.1050 | −0.0095 | 0.1291 | −0.0739 | −0.1812 | 0.2298 |

![Total Return and Realised Sharpe — All Strategies](images/article3_total_return_sharpe_comparison.png)

Focused comparison — grouping by allocation method across prediction strategies :

![Sharpe Allocation: Momentum vs LR vs GBR](images/article3_sharpe_portfolios_comparison.png)

The Sharpe-GBR line (green dot-dash) climbs steeply through Q4 2023, then stays around 0.45–0.57. LR (blue dashed) builds steadily and finishes at 0.34. The Sharpe-momentum line (red) stays relatively flat around 0.20.

![Min Variance Allocation: Momentum vs LR vs GBR](images/article3_minvar_portfolios_comparison.png)

Min variance strategies show much tighter clustering — all three prediction methods converge around 0.21–0.23 total return, but with very different stability profiles. `minvar_lr` leads mid-test before all three converge.

![Markowitz Allocation: Momentum vs LR vs GBR](images/article3_mv_portfolios_comparison.png)

The Markowitz chart shows the widest spread — `mv_momentum` dominates at 1.47, `mv_gbr` at 0.87, while `mv_lr` goes negative. This illustrates how agressive concentration amplifies prediction errors.

---

## Discussion

`mv_momentum` leads on raw return (1.47 log return, 0.74 Sharpe) — momentum + aggressive concentration works in trending markets. But it comes with 0.18 quarterly std.

Min Variance + ML is the risk-adjusted standout. `minvar_gbr` (0.63 Sharpe, 0.034 std) and `minvar_lr` (0.55 Sharpe) use predictions only as a stock filter, ignoring return magnitudes for allocation. This sidesteps noisy return estimates entirely.

Markowitz amplifies everything — `mv_gbr` reaches 0.87 return but `mv_lr` goes *negative* (−0.10), showing how concentration destroys value when predictions are wrong. Random allocation (~0.25 return, ~0.45 Sharpe across all predictions) proves that the stock selection step already captures most of the value.

Bottom line: ML predictions improve stock filtering. `minvar_gbr` (0.63) > `minvar_lr` (0.55) > `minvar_momentum` (0.34) on Sharpe. The signal is real, it's the allocation method that determines whether you exploit it or blow up.

---

## Conclusion

The 3 × 4 strategy grid shows that prediction quality and allocation method interact in non-obvious ways. Min-variance doesn't use predicted magnitudes at all — yet `minvar_gbr` is the second-best risk-adjusted strategy, proving GBR selects the right stocks even if the exact return forecasts are noisy.

**Key takeaways :** Small R² ≠ useless (cross-sectional ranking drives alpha). Min Variance + ML stock selection is surprisingly robust. Markowitz amplifies both signal and errors. Always benchmark against random allocation.

## Current Limitations

- **Survivorship bias** — only current S&P 500 constituents
- **Static covariance** — estimated once on validation data
- **No transaction costs** — high turnover strategies look better on paper
- **Price-only features** — no fundamentals or alternative data
- **Short test period** — 11 quarters is not statistically robust

## Next Steps

- Include historical S&P 500 constituents (survivorship bias)
- Rolling / exponentially-weighted covariance estimation
- Fundamental features (P/E, earnings growth, dividend yield)
- Transaction cost penalty in the objective function
- Ensemble stacking (LR + GBR meta-learner)
- Black-Litterman framework — blend ML predictions with market-implied equilibrium returns for more stable weights

---

**Full Notebook / Code available :**

[https://github.com/alexandreib/medium/blob/main/notebooks/3_SP500_Portfolio_Allocation_ML_predictions.ipynb](https://github.com/alexandreib/medium/blob/main/notebooks/3_SP500_Portfolio_Allocation_ML_predictions.ipynb)
