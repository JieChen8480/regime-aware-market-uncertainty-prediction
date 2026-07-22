Regime-Dependent Predictive Modeling under Market Uncertainty

Overview

This repository presents an empirical research project on predictive model reliability under changing market conditions. Instead of focusing only on static predictive accuracy, the project studies whether short-horizon forecasting models remain stable when markets shift across different uncertainty regimes.

The core question is:

How reliable are predictive models when financial market conditions change across regimes?

⸻

Motivation

Financial time series are non-stationary. Return dynamics, volatility, and dependence structure can change over time, which makes predictive performance fragile. A model that appears acceptable in aggregate may behave very differently across volatility states or historical subperiods.

This project addresses that problem through a regime-aware evaluation framework.

⸻

Methods

The workflow includes:

* feature engineering from financial market data
* LSTM-based volatility proxy with EWMA fallback
* volatility-based and Gaussian-mixture regime classification
* Logistic Regression, Decision Tree, and Random Forest
* benchmark comparison against majority-class and persistence rules
* chronological train-test splitting
* rolling-window validation
* overfitting diagnostics
* regime-specific performance analysis

The target is next-period return direction.

⸻

Main Findings

The main result is that predictive reliability is regime-dependent and temporally unstable.

In particular:

* aggregate test metrics can hide substantial instability across market regimes
* simple benchmarks remain difficult to outperform consistently
* rolling-window validation reveals meaningful variation across time
* more flexible models do not necessarily generalize better

The project therefore emphasizes robustness under changing conditions rather than isolated headline accuracy.

⸻

Contribution

The main contribution of this repository is a reproducible regime-aware empirical framework for evaluating predictive models under market uncertainty.

Its goal is not to claim a production-ready trading system, but to show that model assessment under non-stationarity should go beyond a single train-test result.

⸻

Repository Structure

.
├── README.md
├── methodology.md
├── research_summary.md
├── project_limitations.md
├── outputs/
├── figures/
└── notebooks/ or src/

Key files

* README.md — project overview
* methodology.md — empirical design and workflow
* research_summary.md — concise research summary
* project_limitations.md — scope and limitations
* outputs/ — result tables
* figures/ — generated visualizations

⸻

Reproducibility

The pipeline is designed to run with local CSV market data when available. For reproducibility, it also includes a synthetic-data fallback so the workflow can execute without proprietary data.

⸻

Limitations

This is a research-oriented empirical project, not a deployable trading framework.

Main limitations:

* simplified binary prediction target
* evaluation based mainly on classification metrics rather than economic utility
* regime definitions that can be extended further
* volatility proxy used as an uncertainty feature rather than a definitive volatility model

See project_limitations.md for more detail.

⸻

Project Context

Columbia University
Jan. 2023 – May 2023
Supervised by Prof. Dr. Jörg Osterrieder and Prof. Dr. Ali Hirsa

This repository is included as a reproducible research artifact representing prior work on predictive modeling under non-stationarity, uncertainty-aware evaluation, and regime-dependent empirical analysis.
