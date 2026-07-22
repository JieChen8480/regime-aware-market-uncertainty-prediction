Research Summary

Project Title

Regime-Dependent Predictive Modeling under Market Uncertainty

Institutional Context

Columbia University
Project period: January 2023 – May 2023
Supervision: Prof. Dr. Jörg Osterrieder; Prof. Dr. Ali Hirsa

Research Question

How reliable are predictive models when financial market conditions shift across regimes?

Motivation

A central difficulty in financial prediction is that the data-generating environment is not stable over time. Volatility, dependence structure, and market behavior can change across periods, making predictive models vulnerable to degradation outside the conditions under which they were fit.

This project studies that problem directly by evaluating whether directional prediction models remain reliable when the market moves across different uncertainty regimes.

Objective

The goal of the project is not simply to maximize static predictive accuracy, but to build and assess a regime-aware evaluation framework for short-horizon market prediction under non-stationarity.

Data Context

The analysis uses financial market time-series data to examine:

* short-horizon directional prediction
* volatility variation
* temporal dependence
* regime-dependent model stability

To support reproducibility, the implementation also includes a synthetic-data fallback so that the full workflow can be executed without proprietary data access.

Methods

The empirical workflow includes the following components:

Feature engineering

Engineered predictors include:

* lagged returns
* rolling return statistics
* momentum measures
* moving-average ratios
* price-range features
* volume-based transformations
* volatility-related inputs

Uncertainty representation

A sequence-based volatility proxy is constructed using:

* an LSTM-based next-period absolute return approximation
* an EWMA volatility fallback when deep-learning dependencies are unavailable or the sample is insufficient

Regime construction

Market conditions are represented using:

* volatility-based regime grouping
* Gaussian mixture model regime classification

Predictive models

The main supervised models in the core pipeline are:

* Logistic Regression
* Decision Tree
* Random Forest

Evaluation design

Model reliability is evaluated through:

* chronological train-test splitting
* majority-class and random-walk benchmark comparison
* regime-specific performance analysis
* rolling-window validation
* overfitting diagnostics
* feature-importance inspection for tree-based models

Main Findings

The main result of the project is that predictive reliability is regime-dependent and temporally unstable.

More specifically:

* models that appear acceptable in aggregate can weaken substantially in specific volatility regimes
* naive benchmarks remain important reference points in weak-signal prediction tasks
* rolling-window evaluation reveals that model quality varies across historical subperiods
* higher-complexity models may achieve stronger in-sample fit without delivering stable out-of-sample gains
* regime-aware evaluation provides more informative evidence than a single static performance metric

Contribution

The primary contribution of this project is a reproducible empirical framework for evaluating predictive models under changing market conditions.

Its value lies less in claiming a universally superior classifier and more in showing that:

1. model assessment under non-stationarity requires more than one train-test split
2. aggregate accuracy can obscure meaningful instability
3. regime-aware evaluation improves interpretability of predictive performance

Limitations

The project has several important scope limitations:

* it studies binary direction prediction rather than direct economic optimization
* it focuses on classification metrics rather than portfolio-level utility
* regime definitions are simplified relative to richer state-space approaches
* the volatility proxy is used as an uncertainty feature, not as a definitive volatility model
* hyperparameter tuning is limited relative to a full production modeling pipeline

Relevance for Doctoral Research

This project reflects research interests in:

* forecasting under non-stationarity
* model robustness under uncertainty
* empirical evaluation of predictive systems
* time-series learning in changing environments
* interpretable assessment of model behavior across regimes

It is best understood as a prior empirical research artifact demonstrating problem formulation, methodological structuring, reproducible workflow design, and critical evaluation of model reliability under unstable conditions.
