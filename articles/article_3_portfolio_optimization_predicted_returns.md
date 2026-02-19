# Portfolio Optimization on S&P 500 Stocks — Predicting Returns with ML

In the [first notebook](https://medium.com/@alexandre.durand/portfolio-optimisation-on-s-p-500-stocks-46f03732b030), we explored the theoretical foundations of portfolio optimization.
In the [second notebook](https://medium.com/@alexandre.durand/portfolio-optimization-on-s-p-500-stocks-with-backtest-61da87ed91ff), we backtested three portfolio strategies using past quarterly returns as our "prediction" — essentially a momentum bet.

The obvious weakness? Using last quarter's return to predict next quarter's return is a rough heuristic at best. In this third notebook, we build actual predictive models — **Linear Regression** and **Gradient Boosted Trees** — trained on a set of features to forecast quarterly log returns. We then plug those predictions into the same portfolio optimization framework and compare against the momentum baseline from article 2.

## Contents:

**Feature Engineering:** Build predictive features from price and volume data.

**Model Training:** Fit Linear Regression and XGBoost on historical data.

**Return Prediction:** Generate predicted quarterly log returns on test period.

**Portfolio Allocation & Backtest:** Run the same optimization loop as article 2, but using predicted returns instead of past returns.

**Comparison:** Momentum baseline vs ML-predicted allocations.

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
| `volume_ratio` | Current 63d avg volume / 252d avg volume |
| `mean_reversion` | Deviation of current price from 252d moving average |

These are standard quantitative finance features — nothing exotic, but they capture momentum, mean-reversion, volatility regimes and relative volume shifts.

```python
# Calculate daily log returns
df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

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

    # Volume ratio
    g['volume_ratio'] = g['Volume'].rolling(63).mean() / g['Volume'].rolling(252).mean()

    # Mean reversion signal
    g['mean_reversion'] = g['Adj Close'] / g['Adj Close'].rolling(252).mean() - 1

    return g

df = df.groupby('Ticker', group_keys=False).apply(compute_features)
```

Quick sanity check — let's look at the correlation of individual features with future quarterly return:

```python
feature_cols = ['ret_q1','ret_q2','ret_q4','volatility_63d',
                'volatility_252d','mom_12_1','volume_ratio','mean_reversion']

# Target : next quarter return
df['target'] = df.groupby('Ticker')['Quarterly_Log_Return'].shift(-days)

corr = df[feature_cols + ['target']].corr()['target'].drop('target')
display(corr.sort_values(ascending=False))
```

The correlations are small (as expected in financial data), but not zero. `mom_12_1` and `ret_q1` typically show mild positive correlation with future returns, while `mean_reversion` tends to be slightly negative — stocks that have run up far above their moving average tend to revert.

---

## Split Data

Same split logic as article 2 : Train / Valid / Test. The covariance matrix is estimated on the validation set. Models are trained on training data only.

```python
dates = list(df['Date'].unique())

dates_train = dates[:int(len(dates) * 0.7)]
dates_valid = dates[int(len(dates) * 0.7) : int(len(dates) * 0.85)]
dates_test  = dates[int(len(dates) * 0.85):][0::days] # 1 rebalancing every 63 days

train = df[df['Date'].isin(dates_train)].dropna(subset=feature_cols + ['target'])
valid = df[df['Date'].isin(dates_valid)]
test  = df[df['Date'].isin(dates_test)]

print(f"train : {train['Date'].min()} -> {train['Date'].max()} | {len(train)} rows")
print(f"valid : {valid['Date'].min()} -> {valid['Date'].max()}")
print(f"test  : {test['Date'].min()} -> {test['Date'].max()}")
```

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

The R² will be low — typically 1-3% for quarterly stock return predictions. Don't be alarmed. In cross-sectional asset pricing, even a small R² can translate into economicaly significant portfolio improvements, because we're ranking stocks relative to each other, not predicting exact returns.

### Gradient Boosted Trees (XGBoost)

XGBoost is well suited here because it handles non-linear interactions between features (e.g. momentum behaves differently in high vs low volatility regimes).

We use modest hyperparameters to avoid overfitting — financial data is notoriously noisy.

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=50)

print(f"Train R² : {xgb_model.score(X_train, y_train):.4f}")
```

Let's check feature importance :

```python
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10)
plt.title('XGBoost Feature Importance (Gain)')
plt.show()
```

Typically, `volatility_252d` and `mom_12_1` dominate. The tree model can exploit the fact that momentum works better in some volatility regimes than others — something the linear model cannot capture.

---

## Prediction Function

We wrap prediction in a helper that takes a date's feature data and returns predicted returns per ticker for both models.

```python
def predict_returns(df_date, feature_cols, scaler, lr_model, xgb_model):
    """Predict quarterly log returns for all tickers at a given date."""
    X = df_date[feature_cols].values
    X_scaled = scaler.transform(X)

    preds_lr  = lr_model.predict(X_scaled)
    preds_xgb = xgb_model.predict(X)

    return pd.DataFrame({
        'Ticker': df_date['Ticker'].values,
        'pred_lr': preds_lr,
        'pred_xgb': preds_xgb
    }).set_index('Ticker')
```

---

## Portfolio Optimization Functions

Same as the previous articles — reused directly.

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
    result = sp.optimize.minimize(
        fun=fun, args=args, x0=x0, method='SLSQP',
        bounds=tuple((0, 0.3) for _ in range(number_of_tickers)),
        constraints=({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    )
    return result.x
```

---

## Backtest Loop

This is where it gets interesting. For each rebalancing date in the test period, we:

1. **Predict returns** using both models
2. **Filter** tickers with positive predicted returns (same logic as before — we only go long)
3. **Optimize weights** using Sharpe and Mean-Variance, based on predicted returns
4. **Compute actual realised return** of the portfolio

We run this for 4 strategies per model (LR and XGB), plus the momentum basline from article 2.

```python
pivot_returns_test = test.pivot_table(values='Quarterly_Log_Return',
                                       columns='Ticker', index='Date').fillna(0)

results = {}
for idx in range(1, len(pivot_returns_test)):
    date = pivot_returns_test.iloc[idx].name
    results[date] = {}
    tickers = np.array(pivot_returns_test.iloc[idx].index)
    tickers_returns_future = pivot_returns_test.iloc[idx]
    tickers_returns_momentum = pivot_returns_test.iloc[idx - 1]  # Previous quarter (momentum baseline)

    # Get ML predictions for this date
    df_date = test[test['Date'] == date].dropna(subset=feature_cols)
    if len(df_date) == 0:
        continue
    preds = predict_returns(df_date, feature_cols, scaler, lr_model, xgb_model)

    # Loop over prediction sources
    for pred_name, pred_series in [('momentum', tickers_returns_momentum),
                                     ('lr', preds['pred_lr']),
                                     ('xgb', preds['pred_xgb'])]:

        # Align tickers
        common_tickers = list(set(pred_series.index) & set(tickers_returns_future.index)
                              & set(matrix_covariance.columns))
        pred_filtered = pred_series.loc[common_tickers]
        future_filtered = tickers_returns_future.loc[common_tickers]

        # Keep only positive predictions
        mask_positive = pred_filtered.values > 0
        if mask_positive.sum() < 5:
            continue

        t = np.array(common_tickers)[mask_positive]
        pred_pos = pred_filtered.values[mask_positive]
        future_pos = future_filtered.values[mask_positive]
        cov_filtered = matrix_covariance.loc[t, t]

        # Sharpe optimized weights
        w_sharpe = optimize_weights(pred_pos, cov_filtered.values,
                                     fun=neg_sharpe_ratio_objective)
        results[date][f'returns_sharpe_{pred_name}'] = (w_sharpe * future_pos).sum()

        # Mean-Variance optimized weights
        w_mv = optimize_weights(pred_pos, cov_filtered.values,
                                 fun=neg_markowitz_objective)
        results[date][f'returns_mv_{pred_name}'] = (w_mv * future_pos).sum()
```

---

## Results Analysis

```python
results_df = pd.DataFrame(results).T
display(results_df.head())
```

Let's plot the cumulative returns for all strategies :

```python
l_returns_cols = [x for x in results_df.columns if 'returns_' in x[:9]]
cumsum_df = results_df[l_returns_cols].cumsum()

fig, ax = plt.subplots(figsize=(14, 7))
sns.lineplot(data=cumsum_df, dashes=False, ax=ax)
plt.title('Cumulative Quarterly Log Returns — Momentum vs ML Predictions')
plt.ylabel('Cumulative Log Return')
plt.xlabel('')
plt.xticks(rotation=30)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
```

Now the key comparisons — total returns and realised Sharpe ratios :

```python
# Total returns
print("=== Total Returns ===")
display(results_df[l_returns_cols].sum().sort_values(ascending=False))

# Realised Sharpe Ratio (mean / std of quarterly returns)
print("\n=== Realised Sharpe Ratio ===")
realised_sharpe = results_df[l_returns_cols].mean() / results_df[l_returns_cols].std()
display(realised_sharpe.sort_values(ascending=False))
```

```python
# Side by side comparison
comparison = pd.DataFrame({
    'Total Return': results_df[l_returns_cols].sum(),
    'Avg Quarterly Return': results_df[l_returns_cols].mean(),
    'Std Quarterly Return': results_df[l_returns_cols].std(),
    'Realised Sharpe': realised_sharpe,
    'Max Drawdown (quarterly)': results_df[l_returns_cols].min()
})
display(comparison.sort_values('Realised Sharpe', ascending=False))
```

---

## Discussion

A few things worth noting from the results :

**Linear Regression** tends to produce moderate, stable predictions. Because it's a simple weighted average of features, it doesn't overfit as aggressively — but it also misses non-linear regime effects. Combined with Sharpe optimization, it usually delivers a decent risk-adjusted return.

**XGBoost** can capture feature interactions (e.g. momentum works differentely in high vs low vol environments) which sometimes leads to better stock selection. However, it's more prone to overfitting on training data, which is why we kept the hyperparameters conservative.

**Both ML approaches should outperform the momentum baseline** in terms of realised Sharpe, because they incorporate more information (volatility, volume dynamics, mean-reversion) rather than relying solely on past returns.

The Sharpe-optimized portfolio consistently tends to prodce the best risk-adjusted returns accross prediction methods, which is consistent with findings from article 1 and 2.

---

## Conclusion

We've moved from a naive momentum heuristic to actual return predictions using ML models. The improvement may not be dramatic — and it shouldn't be. Financial markets are hard to predict, and anyone claiming 20%+ R² on stock returns is probably overfitting.

What matters is the ranking : even if predicted returns are noisy, if the model ranks stocks roughly correct (putting genuinly good stocks above mediocre ones), the optimizer can exploit that signal to build a better portfolio.

**Key takeaways :**

- Small R² ≠ useless. Cross-sectional ranking is what drives portfolio alpha.
- Feature engineering matters more than model complexity. Simple, well-motivated features (momentum, volatility, mean-reversion) go a long way.
- Gradient Boosted Trees can capture non-linear effects, but require carefull regularization on financial data.
- The portfolio optimization layer amplifies even weak predictive signal into meaningfull risk-adjusted returns.

## Current Limitations

- **Survivorship Bias**: We're still only using current S&P 500 constituents. Stocks that were removed (often due to poor performance) are missing from our training data.
- **Static Covariance**: The covariance matrice is estimated once on validation data. A rolling or exponentially-weighted covariance might adapt better.
- **No Transaction Costs**: Real portfolios incur costs at each rebalancing. High turnover strategies look better on paper than in practice.
- **Feature Set**: We use price/volume features only. Fundamental data (earnings, book value, etc.) and alternative data (sentiment, news) could further improve predictions.

## Next Steps

- **Address Survivorship Bias**: Include historical S&P 500 constituents.
- **Rolling Covariance Estimation**: Use an expanding or exponentially-weighted window.
- **Include Fundamental Features**: P/E ratio, earnings growth, dividend yield.
- **Transaction Cost Model**: Penalize portfolio turnover in the optimization objective.
- **Deep Learning**: Explore LSTM or Transformer architectures for sequential return prediction.

---

**Full Notebook / Code available :**

[https://github.com/alexandreib/QuantDesign/blob/main/3_SP500_Portfolio_Allocation_ML_predictions.ipynb](https://github.com/alexandreib/QuantDesign/blob/main/3_SP500_Portfolio_Allocation_ML_predictions.ipynb)
