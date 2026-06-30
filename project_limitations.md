# Project Limitations

This project is designed as a research-oriented machine learning demonstration rather than a production trading system.

Main limitations include:

- The empirical setting is financial market data, which may not directly generalize to all operational systems.
- Predictive signals in high-frequency financial data are noisy and unstable.
- The LSTM volatility proxy is used as an uncertainty representation, not as a fully optimized deep learning forecasting system.
- Regime definitions are based on volatility quantiles and could be extended using hidden Markov models or structural break detection.
- Future work could apply the same framework to operations management problems such as staffing, demand forecasting, inventory planning, or queueing systems.
