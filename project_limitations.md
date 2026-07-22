Project Limitations

This repository is intended as a research-oriented empirical project on predictive model reliability under changing market conditions. It should not be interpreted as a production-ready trading system or as evidence of a robust, deployable alpha signal.

1. Simplified Prediction Target

The project studies next-period return direction as a binary classification problem. This is a useful research setting for comparing model behavior across regimes, but it is a simplified objective relative to real investment decision-making. Correct directional classification does not necessarily translate into economic value, especially when return magnitudes are small.

2. Evaluation Focuses on Statistical Metrics

The analysis emphasizes classification metrics such as accuracy, balanced accuracy, F1 score, and ROC-AUC. These measures are appropriate for model comparison, but they do not directly capture trading relevance. The project does not evaluate portfolio construction, transaction costs, turnover, drawdown, or risk-adjusted returns.

3. Regime Definitions Are Limited

Market regimes are defined using volatility-based grouping and Gaussian mixture clustering. These are useful for studying conditional model performance, but they remain simplified representations of changing market states. Richer alternatives, such as hidden Markov models, Markov-switching processes, or structural-break methods, may capture regime dynamics more fully.

4. Volatility Proxy Is a Supporting Feature

The LSTM-based volatility proxy is included as an uncertainty representation within the broader evaluation framework. It should not be interpreted as a fully optimized or definitive volatility-forecasting model. In addition, the workflow uses an EWMA fallback when TensorFlow is unavailable or the sample is too small, so the volatility proxy is best viewed as a practical research feature rather than a central standalone contribution.

5. Limited Model Scope

The core workflow compares a small set of supervised models: Logistic Regression, Decision Tree, and Random Forest. These models provide a useful range of linear and nonlinear methods, but they are not exhaustive. The project does not attempt a comprehensive comparison across all relevant forecasting architectures.

6. Limited Hyperparameter Search

Model settings are chosen to support a stable and interpretable workflow rather than a fully exhaustive tuning exercise. As a result, the reported results should be understood as evidence from a structured empirical pipeline, not as the outcome of a maximal model-optimization effort.

7. Potential Extensions to Leakage Control

The workflow preserves time ordering through chronological splitting and rolling-window validation, but some components of regime construction and feature generation could be made even stricter by fitting all transformations exclusively within training windows. Future extensions could strengthen this point further.

8. Data Scope and Generalizability

The conclusions of the project are tied to the data used in the analysis and should not be generalized without caution. Financial prediction is highly sensitive to sample period, market conditions, asset universe, and feature design. The synthetic fallback data included in the repository is provided only for reproducibility and demonstration of the workflow; it is not a substitute for real empirical evidence.

9. Research Scope

The contribution of the project is primarily methodological. Its purpose is to study how predictive reliability changes across regimes and over time, not to claim that any specific model consistently outperforms in live trading conditions.

Future Directions

Natural extensions include:

* richer regime-detection methods
* train-only transformation pipelines for stricter leakage control
* probability calibration and threshold optimization
* broader model comparison
* economic evaluation with transaction costs and portfolio metrics
* analysis of regime transitions using macro or cross-asset variables

Summary

These limitations do not invalidate the project’s contribution, but they define the scope of its claims. The repository should be read as a reproducible empirical study of regime-aware model evaluation under market uncertainty, rather than as a finalized forecasting or trading system.
