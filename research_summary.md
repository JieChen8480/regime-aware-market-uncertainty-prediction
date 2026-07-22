
Research Summary

Project Title: Regime-Dependent Predictive Modeling under Market Uncertainty
Institution: Columbia University
Period: Jan. 2023 – May 2023
Supervision: Prof. Dr. Jörg Osterrieder; Prof. Dr. Ali Hirsa

Research Question

How reliable are predictive models when financial market conditions shift across regimes?

Motivation

Financial prediction is difficult because market behavior is not stable over time. Models that perform reasonably well in one period may deteriorate when volatility and dependence patterns change. This project studies that instability directly.

Objective

The goal is to evaluate predictive model reliability under non-stationarity, with emphasis on regime-dependent performance rather than static accuracy alone.

Methods

The workflow includes:

* feature engineering from financial time-series data
* LSTM-based volatility proxy with EWMA fallback
* volatility-based and Gaussian-mixture regime classification
* Logistic Regression, Decision Tree, and Random Forest
* benchmark comparison against majority-class and persistence rules
* chronological train-test split
* rolling-window validation
* overfitting diagnostics
* regime-specific performance analysis

Main Findings

The project finds that predictive performance is not stable across time or market conditions. Aggregate test results can conceal substantial variation across volatility regimes, and simple benchmarks remain difficult to outperform consistently. Rolling-window validation further shows that model quality changes across historical subperiods.

Contribution

The main contribution is a reproducible regime-aware evaluation framework for studying predictive reliability under changing market conditions. The project focuses on robustness and empirical evaluation rather than claiming a production-ready trading signal.

Limitations

The study uses a simplified binary prediction target, relies mainly on classification metrics rather than economic utility, and employs regime definitions that can be extended further. The volatility proxy is used as an uncertainty feature, not as a definitive volatility model.

Relevance

This project reflects broader interests in forecasting under non-stationarity, model robustness, and empirical analysis of predictive systems in changing environments.
