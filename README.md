# regime-aware-market-uncertainty-prediction
 Regime-Dependent Prediction under Financial Market Uncertainty


This project studies how machine learning models perform under changing financial market regimes. Instead of treating market prediction as a simple price-direction task, the project reframes the problem as a study of predictive reliability under uncertainty, with a focus on regime-dependent model performance, robustness, and decision-making under non-stationary conditions.

The project was developed as an applied machine learning research project and later extended into a more research-oriented format suitable for demonstrating analytical preparation for doctoral-level work in Operations Management, Operations Research, and data-driven decision-making.

⸻

Research Motivation

Financial markets are highly non-stationary. A model that performs well during calm periods may fail during volatile or structurally different market conditions. This project investigates whether predictive models behave differently across market regimes and whether model reliability can be evaluated more rigorously through regime-specific analysis and rolling-window validation.

The central research question is:

How does predictive model performance change across different market uncertainty regimes, and what does this imply for data-driven decision-making under uncertainty?

Rather than focusing only on maximizing prediction accuracy, this project emphasizes:

* Model robustness
* Regime-dependent performance
* Overfitting detection
* Benchmark comparison
* Rolling-window stability
* Predictive decision-making under uncertainty

⸻

Project Overview

The project predicts the directional movement of financial market returns using historical market features. It includes both traditional machine learning models and uncertainty-aware extensions.

The workflow includes:

1. Data loading and preprocessing
2. Feature engineering
3. Market regime construction
4. LSTM-based volatility proxy generation
5. Baseline benchmark models
6. Machine learning model training
7. Overfitting detection
8. Regime-specific performance comparison
9. Rolling-window validation
10. Research interpretation and summary

⸻

If no external dataset is available, the notebook includes a synthetic-data fallback so that the full workflow can still be executed and reviewed.

⸻

Methods

1. Feature Engineering

The project constructs predictive features from historical market data, including:

* Lagged returns
* Rolling volatility
* Momentum indicators
* Trading range measures
* Volume-based features
* Uncertainty proxies

These features are used to predict future return direction.

⸻

2. Market Regime Construction

Market regimes are constructed based on volatility and uncertainty conditions. The project separates observations into regimes such as:

* Low-volatility regime
* Medium-volatility regime
* High-volatility regime

This allows the analysis to evaluate whether model performance is stable across different market environments.

⸻

3. LSTM Volatility Proxy

An LSTM model is used to estimate a forward-looking volatility proxy from sequences of past returns. This proxy is treated as a learned representation of short-term market uncertainty.

If TensorFlow is unavailable, the workflow automatically falls back to an exponentially weighted moving average volatility proxy to ensure that the notebook remains executable.

⸻

4. Benchmark Models

The project includes simple benchmark models to evaluate whether machine learning models provide meaningful improvement over naive prediction rules.

Benchmarks include:

* Majority-class benchmark
* Random-walk persistence benchmark

These benchmarks are important because financial prediction tasks often have weak signal-to-noise ratios. A machine learning model should be compared against simple decision rules before being interpreted as useful.

⸻

5. Machine Learning Models

The project evaluates several supervised learning models, including:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

The models are compared using train and test performance, with emphasis on both accuracy and overfitting risk.

⸻

6. Overfitting Detection

For each model, the project reports:

* Training accuracy
* Test accuracy
* Overfitting gap

The overfitting gap is calculated as the difference between training and test performance. This helps identify models that fit historical data too closely but fail to generalize.

⸻

7. Regime-Specific Performance Comparison

The project evaluates model performance separately within each market regime. This helps answer whether a model performs consistently across different uncertainty environments.

This section is central to the research framing because it moves the project beyond a single aggregate accuracy score and toward a more nuanced understanding of model reliability.

⸻

8. Rolling-Window Validation

Rolling-window validation is used to evaluate model stability over time. Instead of relying on one static train-test split, the model is repeatedly trained and tested across different time windows.

This approach better reflects real-world forecasting conditions, where models are trained on historical data and applied to future periods with potentially changing data-generating processes.

⸻

Key Outputs

The project generates the following output files:

outputs/model_performance_summary.csv
outputs/regime_specific_performance.csv
outputs/rolling_window_results.csv
outputs/rolling_window_summary.csv

These files summarize:

* Overall model performance
* Overfitting gaps
* Benchmark comparisons
* Regime-specific accuracy
* Rolling-window stability

⸻

Research Contribution

The main contribution of this project is not simply predicting market direction. Instead, the project demonstrates how predictive models can be evaluated under changing uncertainty conditions.

The project shows how a financial forecasting task can be reframed as a broader research problem in:

* Predictive modeling under uncertainty
* Regime-dependent decision-making
* Model robustness
* Non-stationary environments
* Data-driven operational decision support

This framing is relevant to Operations Management and Operations Research because many real-world operational systems face similar uncertainty and regime shifts, including:

* Staffing demand
* Service operations
* Queueing systems
* Supply chain disruptions
* Capacity planning
* Dynamic resource allocation

⸻


The project addresses a broader question:

How should decision-makers evaluate predictive models when the environment changes over time?

This question is relevant in many OM contexts where historical data may not remain stable, including demand forecasting, workforce planning, inventory management, and service system design.

The project demonstrates preparation for research involving:

* Empirical modeling
* Machine learning
* Forecast evaluation
* Robust decision-making
* Uncertainty-aware analytics
* Data-driven operations

⸻

Dependencies

The project uses standard Python data science libraries:

pip install pandas numpy scikit-learn matplotlib

Optional dependency for the LSTM volatility proxy:

pip install tensorflow

If TensorFlow is not installed, the project automatically uses an EWMA volatility proxy instead.


⸻

Author

Jie Chen

