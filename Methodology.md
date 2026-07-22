
Methodology

This project evaluates whether short-horizon predictive models remain reliable when market conditions shift across different uncertainty regimes.

Data and Preprocessing

The pipeline loads financial market data from local CSV files when available, with a synthetic-data fallback for reproducibility. Returns are inferred when necessary, timestamps are parsed, and observations are sorted chronologically.

Feature Engineering

The modeling dataset includes:

* lagged returns
* rolling return statistics
* momentum and moving-average features
* price-range measures
* volume-based variables
* volatility-related features

The target is next-period return direction.

Volatility Proxy

Market uncertainty is represented using an LSTM-based volatility proxy trained on recent return sequences. When TensorFlow is unavailable or the sample is too small, the workflow falls back to EWMA volatility.

Regime Construction

Market regimes are defined in two ways:

* quantile-based volatility regimes
* Gaussian mixture clustering on return and volatility-related features

This combines interpretability with a more flexible unsupervised grouping method.

Predictive Models

The main supervised models are:

* Logistic Regression
* Decision Tree
* Random Forest

These are compared with two simple benchmarks:

* majority-class prediction
* random-walk persistence

Evaluation

The project uses:

* chronological train-test splitting
* accuracy, balanced accuracy, F1, and ROC-AUC
* regime-specific performance comparison
* rolling-window validation
* train-test gap analysis for overfitting detection

Main Methodological Point

The project is designed to show that predictive models should be evaluated not only by overall out-of-sample performance, but also by their stability across time and across market regimes.

Limitations

The framework is research-oriented rather than production-oriented. It does not directly evaluate trading profitability, transaction costs, or portfolio construction, and the regime definitions remain simplified relative to richer state-space approaches.
