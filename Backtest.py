import numpy as np
import pandas as pd

import Strategy as strat
from Metrics import plot_pnl

# read data
data = pd.read_csv('data/crafted_data.csv')

# declare the model and hyper parameters
model = strat.ridgeModel
lookback = 4 * 365
cooldown = 365
retrain_window = 365
optimizer = strat.mean_variance_optimization
n_estimators = 100
max_depth = 1

# define signals
all_signals = ['pmt_long_term',
               'pmt_mid_term', 'pmt_short_term', 'rsi_short_term', 'rsi_long_term',
               'rsi_mid_term', 'price_acc_short_term', 'price_acc_long_term',
               'price_acc_mid_term', 'vwap_signal', 'pvt_signal',
               'cross_sectional_mean_reversion_signal']

backtester = strat.rollingStrategy(cooldown=cooldown, lookback=lookback, retrain_window=retrain_window)
backtester.roll(data=data, model=model, signal_names=all_signals, penalty=1)
ridge_pnl_curve = np.cumsum(backtester.get_pnl('predictions'))

gbdt_backtester = strat.rollingStrategy(cooldown=cooldown, lookback=lookback, retrain_window=retrain_window)
backtester.roll(data=data, model=strat.gBDTModel, signal_names=all_signals, n_estimators=n_estimators,
                max_depth=max_depth)
gbdt_pnl_curve = np.cumsum(backtester.get_pnl('predictions'))

# data analysis

# compare gbdt and ridge regression
# Assuming df is a DataFrame where each column is a pnl curve
pnl_curves = pd.DataFrame({
    'ridge PnL curve': ridge_pnl_curve[cooldown + 1:],
    'gbdt PnL curve': gbdt_pnl_curve[cooldown + 1:]
})
plot_pnl(pnl_curves)

# Hyperparameter selection of ridge via cross validation
ridge_pnl_curves = {}
lam = [0, 0.1, 0.5, 1, 5]
for k in lam:
    print(f'running ridge {k}...')
    backtester = strat.rollingStrategy(cooldown=cooldown, lookback=lookback, retrain_window=retrain_window)
    backtester.roll(data=data, model=model, signal_names=all_signals, penalty=k)
    ridge_pnl_curve = np.cumsum(backtester.get_pnl('predictions'))
    ridge_pnl_curves[f'Penalty = {k}'] = ridge_pnl_curve[cooldown + 1:]
ridge_pnl_curves = pd.DataFrame(ridge_pnl_curves)
plot_pnl(ridge_pnl_curves)

gbdt_pnl_curves = {}
trees = [50, 100]
depth = [1, 2]
for t in trees:
    for d in depth:
        print(f'calculating tree {t} and depth {d}...')
        backtester = strat.rollingStrategy(cooldown=cooldown, lookback=lookback, retrain_window=retrain_window)
        backtester.roll(data=data, model=strat.gBDTModel, signal_names=all_signals, n_estimators=t, max_depth=d)
        gbdt_pnl_curve = np.cumsum(backtester.get_pnl('predictions'))
        gbdt_pnl_curves[f'Tree number = {t}, max depth = {d}'] = gbdt_pnl_curve[cooldown + 1:]
gbdt_pnl_curves = pd.DataFrame(gbdt_pnl_curves)
plot_pnl(gbdt_pnl_curves)

## Signal Performance
