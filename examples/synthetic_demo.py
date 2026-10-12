from __future__ import annotations

import numpy as np
import pandas as pd

from wr_nitrate.temporal import exact_change_targets, persistence_skill


def main() -> None:
    rng = np.random.default_rng(2026)
    timestamps = pd.date_range("2024-04-01", periods=600, freq="min")
    values = 8 + np.cumsum(rng.normal(0, 0.02, len(timestamps)))
    series = pd.Series(values, index=timestamps, name="bt_effluent_no3")
    labeled = exact_change_targets(series, "60min")
    y = labeled["delta"].to_numpy()
    split = int(len(y) * 0.8)
    print("Synthetic Persistence reference:", persistence_skill(y[split:], np.zeros_like(y[split:])))


if __name__ == "__main__":
    main()
