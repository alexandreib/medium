import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sp_opt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings, os
warnings.filterwarnings('ignore')
np.random.seed(42)

# ===== DATA LOADING =====
df = pd.read_csv('data/sp500_20years.csv')
print(f"Data loaded: {df.shape[0]} rows, {df['Ticker'].nunique()} tickers")
print(f"Date range: {df['Date'].min()} -> {df['Date'].max()}")

# Calculate daily log returns
df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

days = 63
df['Quarterly_Log_Return'] = df.groupby('Ticker')['Log_Return']\
    .rolling(window=days, min_periods=days).sum()\
    .reset_index(0, drop=True)

# ===== FEATURE ENGINEERING =====
def compute_features(group):
    g = group.sort_values('Date').copy()
    g['ret_q1'] = g['Quarterly_Log_Return'].shift(days)
    g['ret_q2'] = g['Quarterly_Log_Return'].shift(days * 2)
    g['ret_q4'] = g['Quarterly_Log_Return'].shift(days * 4)
    g['volatility_63d'] = g['Log_Return'].rolling(63).std()
    g['volatility_252d'] = g['Log_Return'].rolling(252).std()
    ret_12m = g['Log_Return'].rolling(252).sum()
    ret_1m = g['Log_Return'].rolling(21).sum()
    g['mom_12_1'] = ret_12m - ret_1m
    # volatility_ratio instead of volume_ratio (no volume data available)
    g['volatility_ratio'] = g['volatility_63d'] / g['volatility_252d']
    g['mean_reversion'] = g['Adj Close'] / g['Adj Close'].rolling(252).mean() - 1
    return g

df = df.groupby('Ticker', group_keys=False).apply(compute_features)

feature_cols = ['ret_q1','ret_q2','ret_q4','volatility_63d',
                'volatility_252d','mom_12_1','volatility_ratio','mean_reversion']

df['target'] = df.groupby('Ticker')['Quarterly_Log_Return'].shift(-days)

# ===== CORRELATION =====
corr = df[feature_cols + ['target']].corr()['target'].drop('target')
print("\n=== Feature Correlations ===")
print(corr.sort_values(ascending=False).round(6))

# Plot correlation
corr_sorted = corr.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
corr_sorted.plot(kind='barh', ax=ax, color=['#2ecc71' if x > 0 else '#e74c3c' for x in corr_sorted])
ax.set_title('Feature Correlation with Future Quarterly Return')
ax.set_xlabel('Pearson Correlation')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('articles/images/article3_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== SPLIT DATA =====
dates = sorted(df['Date'].unique())
dates_train = dates[:int(len(dates) * 0.7)]
dates_valid = dates[int(len(dates) * 0.7) : int(len(dates) * 0.85)]
dates_test  = dates[int(len(dates) * 0.85):]
dates_test_rebalance = dates_test[0::days]

train = df[df['Date'].isin(dates_train)].dropna(subset=feature_cols + ['target'])
valid = df[df['Date'].isin(dates_valid)]
test  = df[df['Date'].isin(dates_test)]
test_rebalance = df[df['Date'].isin(dates_test_rebalance)]

print(f"\ntrain : {len(train):>8} rows | {dates_train[0]} -> {dates_train[-1]}")
print(f"valid : {len(valid):>8} rows | {dates_valid[0]} -> {dates_valid[-1]}")
print(f"test  : {len(test):>8} rows | {dates_test[0]} -> {dates_test[-1]}")
print(f"test rebalancing dates : {len(dates_test_rebalance)}")

# ===== COVARIANCE =====
def calculate_shrink_cov_matrix(df_values):
    masked_arr = np.ma.array(df_values, mask=np.isnan(df_values))
    cov_numpy = np.ma.cov(masked_arr, rowvar=False, allow_masked=True, ddof=1).data
    n_samples, n_features = df_values.shape
    alpha = np.mean(cov_numpy**2)
    mu = np.trace(cov_numpy) / n_features
    mu_squared = mu**2
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    shrunk_cov = (1.0 - shrinkage) * cov_numpy
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov

pivot_returns_valid = valid.pivot_table(values='Quarterly_Log_Return',
                                         columns='Ticker', index='Date').fillna(0)
matrix_covariance = calculate_shrink_cov_matrix(pivot_returns_valid.values)
matrix_covariance = pd.DataFrame(matrix_covariance,
                                  columns=pivot_returns_valid.columns,
                                  index=pivot_returns_valid.columns)

# ===== LINEAR REGRESSION =====
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])
y_train = train['target'].values

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

coef_df = pd.DataFrame({'feature': feature_cols, 'coef': lr_model.coef_})
print("\n=== LR Coefficients ===")
print(coef_df.sort_values('coef', ascending=False).to_string(index=False))
lr_r2 = lr_model.score(X_train, y_train)
print(f"Train R2 : {lr_r2:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
coef_sorted = coef_df.sort_values('coef', ascending=False)
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in coef_sorted['coef']]
ax.barh(coef_sorted['feature'], coef_sorted['coef'], color=colors)
ax.set_title('Linear Regression -- Standardized Coefficients')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('articles/images/article3_lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== GBR =====
gbr_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.7, min_samples_leaf=50, random_state=42
)
gbr_model.fit(X_train, y_train)
gbr_r2 = gbr_model.score(X_train, y_train)
print(f"\nGBR Train R2 : {gbr_r2:.4f}")

fi = pd.DataFrame({'feature': feature_cols, 'importance': gbr_model.feature_importances_})
fi = fi.sort_values('importance', ascending=True)
print("=== GBR Feature Importance ===")
print(fi.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(fi['feature'], fi['importance'], color='#3498db')
ax.set_title('Gradient Boosted Trees -- Feature Importance (Impurity)')
plt.tight_layout()
plt.savefig('articles/images/article3_gbr_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== PREDICTION + BACKTEST =====
def predict_returns(df_date, feature_cols, scaler, lr_model, gbr_model):
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

def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_portfolio_returns(weights, returns):
    return np.dot(weights, returns)

def neg_sharpe_ratio_objective(weights, returns, cov_matrix, risk_free_rate=0.03):
    pr = np.squeeze(calculate_portfolio_returns(weights, returns))
    pv = np.squeeze(calculate_portfolio_variance(weights, cov_matrix))
    return -((pr - risk_free_rate) / np.sqrt(pv))

def neg_markowitz_objective(weights, returns, cov_matrix, gamma=0.2):
    pr = np.squeeze(calculate_portfolio_returns(weights, returns))
    pv = np.squeeze(calculate_portfolio_variance(weights, cov_matrix))
    return gamma * pv - pr

def optimize_weights(log_returns, covariance_matrix, fun=neg_markowitz_objective, x0=None):
    n = len(log_returns)
    if x0 is None:
        x0 = np.array([1/n]*n)
    if fun == calculate_portfolio_variance:
        args = (covariance_matrix,)
    else:
        args = (log_returns, covariance_matrix)
    result = sp_opt.minimize(
        fun=fun, args=args, x0=x0, method='SLSQP',
        bounds=tuple((0, 0.3) for _ in range(n)),
        constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    )
    return result.x

pivot_returns_test = test_rebalance.pivot_table(
    values='Quarterly_Log_Return', columns='Ticker', index='Date').fillna(0)

results = {}
for idx in range(1, len(pivot_returns_test)):
    date = pivot_returns_test.iloc[idx].name
    results[date] = {}
    tickers_returns_future = pivot_returns_test.iloc[idx]
    tickers_returns_momentum = pivot_returns_test.iloc[idx - 1]

    df_date = test_rebalance[test_rebalance['Date'] == date]
    preds = predict_returns(df_date, feature_cols, scaler, lr_model, gbr_model)
    if len(preds) == 0:
        continue

    for pred_name, pred_series in [('momentum', tickers_returns_momentum),
                                     ('lr', preds['pred_lr']),
                                     ('gbr', preds['pred_gbr'])]:
        common_tickers = list(set(pred_series.index) & set(tickers_returns_future.index)
                              & set(matrix_covariance.columns))
        if len(common_tickers) < 5: continue
        pred_filtered = pred_series.loc[common_tickers]
        future_filtered = tickers_returns_future.loc[common_tickers]
        mask_positive = pred_filtered.values > 0
        if mask_positive.sum() < 5: continue
        t = np.array(common_tickers)[mask_positive]
        pred_pos = pred_filtered.values[mask_positive]
        future_pos = future_filtered.values[mask_positive]
        cov_filtered = matrix_covariance.loc[t, t].values

        if pred_name == 'momentum':
            w_random = np.random.rand(len(t))
            w_random = w_random / w_random.sum()
            results[date]['returns_random'] = (w_random * future_pos).sum()

        w_sharpe = optimize_weights(pred_pos, cov_filtered, fun=neg_sharpe_ratio_objective)
        results[date][f'returns_sharpe_{pred_name}'] = (w_sharpe * future_pos).sum()
        w_mv = optimize_weights(pred_pos, cov_filtered, fun=neg_markowitz_objective)
        results[date][f'returns_mv_{pred_name}'] = (w_mv * future_pos).sum()

print(f"\nBacktest complete. {len(results)} periods.")

results_df = pd.DataFrame(results).T.sort_index()
results_df = results_df.dropna(how='all')
l_returns_cols = [x for x in results_df.columns if 'returns_' in x]

print("\n=== Results Head ===")
print(results_df.head().round(4).to_string())

# ===== PLOTS =====
# Quarterly scatter
fig, ax = plt.subplots(figsize=(14, 6))
for col in l_returns_cols:
    ax.scatter(results_df.index, results_df[col], label=col.replace('returns_', ''), s=25, alpha=0.7)
ax.set_ylabel('Quarterly Log Return')
ax.set_title('Quarterly Returns per Strategy')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('articles/images/article3_plot_4.png', dpi=150, bbox_inches='tight')
plt.close()

# Cumulative
cumsum_df = results_df[l_returns_cols].cumsum()
fig, ax = plt.subplots(figsize=(14, 7))
for col in cumsum_df.columns:
    label = col.replace('returns_', '')
    ax.plot(cumsum_df.index, cumsum_df[col], label=label, linewidth=1.5)
ax.set_ylabel('Cumulative Log Return')
ax.set_title('Cumulative Quarterly Log Returns -- Momentum vs ML Predictions')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('articles/images/article3_cumulative_returns.png', dpi=150, bbox_inches='tight')
plt.close()

# Total returns + sharpe bar charts
realised_sharpe = results_df[l_returns_cols].mean() / results_df[l_returns_cols].std()
comparison = pd.DataFrame({
    'Total Return': results_df[l_returns_cols].sum(),
    'Avg Q Return': results_df[l_returns_cols].mean(),
    'Std Q Return': results_df[l_returns_cols].std(),
    'Realised Sharpe': realised_sharpe,
    'Max Q Drawdown': results_df[l_returns_cols].min(),
    'Best Q Return': results_df[l_returns_cols].max()
})
comparison = comparison.sort_values('Realised Sharpe', ascending=False)
comparison.index = comparison.index.str.replace('returns_', '')

print("\n=== Comparison Table ===")
print(comparison.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
comparison['Total Return'].plot(kind='barh', ax=axes[0], color='#3498db')
axes[0].set_title('Total Cumulative Return')
axes[0].set_xlabel('Log Return')
comparison['Realised Sharpe'].plot(kind='barh', ax=axes[1], color='#2ecc71')
axes[1].set_title('Realised Sharpe Ratio')
axes[1].set_xlabel('Sharpe')
plt.tight_layout()
plt.savefig('articles/images/article3_total_return_sharpe_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Sharpe-optimized comparison
sharpe_cols = [c for c in l_returns_cols if 'sharpe' in c]
cumsum_sharpe = results_df[sharpe_cols].cumsum()
fig, ax = plt.subplots(figsize=(12, 6))
styles = ['-', '--', '-.']
colors = ['#e74c3c', '#3498db', '#2ecc71']
for i, col in enumerate(cumsum_sharpe.columns):
    label = col.replace('returns_sharpe_', 'Sharpe -- ')
    ax.plot(cumsum_sharpe.index, cumsum_sharpe[col],
            label=label, linewidth=2, linestyle=styles[i % 3], color=colors[i % 3])
ax.set_ylabel('Cumulative Log Return')
ax.set_title('Sharpe-Optimized Portfolios: Momentum vs Linear Regression vs GBR')
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('articles/images/article3_sharpe_portfolios_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll plots saved to articles/images/")
