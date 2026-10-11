"""Independent Cryptocurrency Backtester (ICBT).

A self-contained, dependency-free spot-trading backtesting engine with a FastAPI
dashboard, CLI, and support for custom strategies including a port of
NostalgiaForInfinityX8 (see ``icbt.strategy.nfi_x8``).

Public API:
  - Data loading / validation:     :mod:`icbt.data`
  - Indicators:                    :mod:`icbt.strategy.indicators`
  - Strategy interface:            :class:`icbt.strategy.base.Strategy`
  - Backtesting engine:            :class:`icbt.engine.engine.BacktestEngine`
  - Analytics / metrics:           :mod:`icbt.analytics`
  - Persistence / run registry:    :mod:`icbt.engine.persistence`
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
