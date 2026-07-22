Results

This document summarizes the empirical results from three perspectives: overall model comparison, regime-specific performance, and rolling-window stability.

1. Overall Results

Overall out-of-sample results compare the supervised models with simple benchmarks using classification metrics such as accuracy, balanced accuracy, F1 score, and ROC-AUC where applicable.

The main takeaway is that aggregate predictive gains are limited and should be interpreted cautiously. In short-horizon financial prediction, benchmark rules remain difficult to outperform consistently, and stronger in-sample fit does not necessarily imply better out-of-sample reliability.

2. Regime-Specific Results

Performance is also evaluated separately across volatility regimes to examine whether model behavior changes under different market conditions.

This analysis shows that predictive performance is not uniform across regimes. A model that appears reasonable in aggregate may perform less reliably in certain volatility states, which suggests that a single overall metric can hide important instability.

3. Rolling-Window Findings

Rolling-window validation is used to assess performance across sequential historical periods rather than relying on one train-test split.

The results show meaningful variation over time, indicating that model quality is period-dependent. This supports the broader conclusion that predictive models in financial markets should be evaluated by their stability across changing environments, not only by one overall test result.

Summary

Taken together, the results show that predictive reliability is conditional on both market regime and historical period. The main contribution of the project is therefore a regime-aware evaluation framework for studying model robustness under non-stationary conditions.
