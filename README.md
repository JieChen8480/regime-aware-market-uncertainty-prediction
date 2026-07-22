Regime-Dependent Predictive Modeling under Market Uncertainty

Overview

This repository presents an empirical research project on predictive model reliability under changing market conditions. The project examines whether supervised learning models for short-horizon directional prediction remain stable when financial markets shift across different uncertainty regimes.

Rather than focusing only on which classifier achieves the highest static accuracy, this study asks a more research-oriented question: how robust are predictive models when the data-generating environment is non-stationary?

The repository is structured as a reproducible research artifact. It includes feature engineering workflows, regime construction, benchmark evaluation, rolling-window validation, overfitting diagnostics, and generated outputs for comparing model behavior under different volatility conditions.

⸻

Research Question

How reliable are directional prediction models when financial market conditions shift across regimes?

This question is motivated by a core challenge in financial machine learning: model performance that appears acceptable in aggregate may deteriorate substantially once the market transitions into a different volatility or uncertainty state. In this setting, static train-test evaluation alone may not be sufficient to assess predictive reliability.

⸻

Motivation

Financial time-series prediction is difficult because the underlying environment is rarely stable. Return dynamics, volatility patterns, temporal dependence, and market microstructure can change over time, which makes predictive relationships fragile.

This project was designed to study that instability directly. Instead of treating all observations as if they came from a single homogeneous process, the analysis evaluates model performance conditional on market regime and across rolling historical windows.

The broader goal is not to claim a production-ready trading signal, but to investigate how model reliability changes under uncertainty and how evaluation design can better reflect non-stationary conditions.

⸻

Data

The project uses financial market data to study:

* short-horizon return direction prediction
* volatility variation over time
* temporal dependence in returns
* regime-dependent model behavior

The workflow is designed to run on local CSV data when available. For reproducibility, the repository also includes a synthetic market-data fallback that allows the full pipeline to execute without proprietary inputs. The synthetic data generation process is intended only to demonstrate the workflow and should not be interpreted as a substitute for real market data.

Typical fields supported by the pipeline include:

* timestamp or datetime index
* open, high, low, close
* volume
* return series
* optional cross-asset proxy variables

⸻

Methodological Design

The empirical pipeline is organized around five components:

1. Feature engineering
2. Uncertainty representation
3. Regime construction
4. Predictive modeling
5. Robustness evaluation

1. Feature Engineering

The modeling dataset is constructed using lagged and rolling features derived from market data. These include:

* lagged returns
* rolling means and standard deviations of returns
* rolling skewness and absolute-return statistics
* momentum and moving-average ratios from price series
* range-based features
* volume transformations and rolling z-scores

The target variable is defined as the next-period return direction, turning the problem into a binary classification task.

2. Uncertainty Representation

To characterize market uncertainty, the project includes a sequence-based volatility proxy:

* primary approach: an LSTM trained on recent return sequences to predict next-period absolute return
* fallback approach: exponentially weighted moving volatility when TensorFlow is unavailable or the sample is too small

This volatility proxy is used as an input into the regime-analysis workflow. It is intended as an uncertainty representation rather than as a standalone volatility-forecasting contribution.

3. Regime Construction

The project defines market regimes in two complementary ways:

* transparent volatility regimes based on quantile grouping of the volatility proxy
* unsupervised Gaussian mixture regimes based on returns and volatility-related features

This allows the analysis to compare a more interpretable grouping rule with a more flexible clustering-based representation.

4. Predictive Models

The main supervised models evaluated in the current workflow are:

* Logistic Regression
* Decision Tree
* Random Forest

These models were selected to provide a comparison across:

* a linear baseline
* a low-complexity nonlinear model
* an ensemble tree method

5. Robustness Evaluation

Model assessment goes beyond a single train-test split. The workflow includes:

* chronological train-test evaluation
* benchmark comparison
* overfitting diagnostics
* regime-specific performance analysis
* rolling-window validation

This structure is intended to evaluate not only predictive performance, but also stability across time and market environments.

⸻

Benchmarks

In weak-signal financial prediction tasks, benchmark selection is critical. This project compares machine learning models against simple but meaningful reference rules:

* Majority Class Benchmark
    Predict the most common direction observed in the training set.
* Random-Walk Persistence Benchmark
    Predict that the next direction will match the current observed return direction.

These benchmarks help contextualize whether apparent model gains are economically or statistically meaningful, rather than simply artifacts of class imbalance or noisy data.

⸻

Evaluation Metrics

The project reports standard classification metrics, including:

* accuracy
* balanced accuracy
* F1 score
* ROC-AUC when applicable

Balanced accuracy is especially important in this setting because directional classes may not be perfectly balanced.

To assess model reliability more carefully, the workflow also reports:

* train accuracy
* test accuracy
* train-test performance gap as a simple overfitting diagnostic
* regime-conditional metrics
* rolling-window summary statistics

⸻

Main Analytical Components

Chronological Train-Test Split

The project uses chronological splitting rather than random shuffling to preserve temporal ordering and reduce unrealistic information leakage.

Regime-Specific Evaluation

After fitting models on the training set, performance is evaluated separately within different volatility regimes on the test set. This helps identify whether a model that appears acceptable overall becomes unstable in high-volatility or high-uncertainty environments.

Rolling-Window Validation

A rolling-window framework is used to measure temporal generalization. Instead of relying on a single historical split, the project repeatedly trains and tests models across multiple sequential windows, allowing the analysis to observe variation in performance over time.

Overfitting Diagnostics

The project compares in-sample and out-of-sample accuracy to identify large train-test gaps that may indicate model overfitting. This is particularly useful for tree-based methods in noisy financial prediction settings.

Feature Importance

For models that expose feature importance, the workflow extracts and visualizes the most influential predictors. This supports interpretability and helps identify which engineered signals contribute most strongly to model behavior.

⸻

Main Findings

The central takeaway of this project is methodological rather than purely predictive.

Key findings include:

* Aggregate model performance can hide substantial instability.
    A model that appears acceptable on overall test metrics may perform very differently across volatility regimes.
* Simple benchmarks remain difficult to beat consistently.
    In directional market prediction, naive reference rules are important because the signal-to-noise ratio is often low.
* Temporal robustness matters as much as point performance.
    Rolling-window validation often reveals meaningful variation in model quality across historical subperiods.
* Higher model flexibility does not guarantee better generalization.
    More complex tree-based models may fit the training data more closely while showing larger train-test gaps.
* Uncertainty-aware evaluation adds interpretive value.
    Studying performance conditional on market regime provides a richer picture than static accuracy alone.

Overall, the project suggests that evaluating predictive systems under changing regimes is more informative than reporting a single out-of-sample score.

⸻
Repository structure

* README.md
    Project overview, research framing, and repository guide.
* methodology.md
    Detailed description of empirical design, feature construction, regime analysis, and validation logic.
* research_summary.md
    Concise summary of the research question, methods, and findings.
* project_limitations.md
    Discussion of methodological and practical limitations.
* outputs/
    Generated result tables for model comparison, regime-specific analysis, and rolling-window validation.
* figures/
    Saved visualizations of volatility dynamics, model comparison, overfitting gaps, regime-specific performance, and rolling-window behavior.

⸻

Reproducibility

The repository is intended to be reproducible in two modes:

1. With local market data

If a supported CSV file is available locally, the pipeline will load and process it for analysis.

2. With synthetic fallback data

If no local dataset is found and synthetic mode is enabled, the pipeline generates market-like intraday data so the workflow can still be executed end to end.

This design allows the repository to remain runnable for demonstration and review purposes while preserving flexibility for proprietary or restricted datasets.

⸻

Typical Workflow

The standard workflow is:

1. load market data
2. infer or construct return series
3. engineer lagged, rolling, price-based, and volume-based features
4. construct a volatility proxy
5. define market regimes
6. build the modeling dataset
7. perform chronological train-test splitting
8. evaluate benchmarks and supervised models
9. compare regime-specific performance
10. run rolling-window validation
11. inspect feature importance
12. save outputs and figures

⸻

Why This Project Matters

This project is relevant to research on:

* financial machine learning
* forecasting under non-stationarity
* robustness and model stability
* empirical evaluation under uncertainty
* regime-aware prediction and decision systems

Although the application domain is financial markets, the broader methodological lesson extends beyond finance: predictive systems should be evaluated not only by aggregate performance, but also by their stability under changing environments.

This perspective is relevant to many areas of data-driven decision-making where the underlying process may shift over time.

⸻

Limitations

This repository should be interpreted as a research-oriented empirical project, not as a deployable trading system.

Important limitations include:

* the target is a simplified binary direction task rather than a direct economic optimization objective
* evaluation focuses on classification metrics rather than trading profitability, transaction costs, or portfolio utility
* regime definitions are useful but still simplified relative to richer state-space or structural-break approaches
* the LSTM-based volatility proxy is included as an uncertainty feature, not as a fully optimized deep-learning volatility model
* the project emphasizes reproducibility and interpretability over exhaustive hyperparameter tuning

These limitations do not invalidate the project’s conclusions, but they define the scope of the claims that can reasonably be made from it.

For additional detail, see project_limitations.md.

⸻

Future Extensions

Natural directions for future work include:

* hidden Markov or Markov-switching regime models
* train-only regime fitting to tighten leakage controls
* additional benchmark models and calibration analysis
* economic evaluation with returns, turnover, and transaction costs
* probability-threshold optimization instead of fixed classification rules
* more extensive hyperparameter tuning and nested validation
* interpretation of regime transitions using macro or cross-asset variables

⸻

Summary

This repository documents a study of regime-dependent predictive modeling under market uncertainty. Its primary contribution is not the claim of a universally superior classifier, but the development of a regime-aware empirical evaluation framework for studying predictive reliability under non-stationary conditions.

The project shows that, in financial prediction, static performance metrics alone may be misleading. To understand whether a model is genuinely reliable, it is necessary to evaluate how it behaves across time, across uncertainty regimes, and against simple reference rules.

