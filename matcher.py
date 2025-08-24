"""Logic to compare new orders against historical data.

The OrderMatcher class uses historical order quantities to compute
expected behaviour for each customer+item combination. When matching
new orders it identifies unknown items and quantity anomalies based on
z‑scores. You can tune the threshold by supplying it at instantiation
time.
"""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


class OrderMatcher:
    """Compare new orders with historical quantity patterns."""

    def __init__(self, history_df: pd.DataFrame, qty_zscore_threshold: float = 3.0) -> None:
        """Construct an OrderMatcher.

        Parameters
        ----------
        history_df: pandas.DataFrame
            Historical orders with at least the columns ``customer_code``,
            ``item_code`` and ``qty_ordered``.
        qty_zscore_threshold: float, optional
            Absolute z‑score above which a quantity is considered anomalous.
            Default is 3.0 (roughly 3 standard deviations).
        """
        self.history = history_df.copy()
        self.qty_zscore_threshold = qty_zscore_threshold
        # Compute mean and std for quantity ordered per customer+item
        key = ["customer_code", "item_code"]
        # Filter rows with numeric quantities
        qty_series = pd.to_numeric(self.history.get("qty_ordered"), errors="coerce")
        self.history["qty_ordered_num"] = qty_series
        grouped = self.history.groupby(key)["qty_ordered_num"].agg(["mean", "std"]).reset_index()
        grouped.rename(columns={"mean": "qty_mean", "std": "qty_std"}, inplace=True)
        self.stats = grouped

    def match(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Compare new orders against historical stats.

        Parameters
        ----------
        orders_df: pandas.DataFrame
            New orders with at least ``customer_code``, ``item_code`` and
            ``qty_ordered`` columns.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the original orders plus extra columns:

            - qty_mean: mean quantity for this customer+item in history
            - qty_std: standard deviation of quantity
            - qty_zscore: z‑score of the current order quantity
            - flags: comma‑separated list of flags such as "UNKNOWN_ITEM" or "QTY_ANOM"
        """
        df = orders_df.copy()
        # Ensure numeric
        df["qty_ordered"] = pd.to_numeric(df["qty_ordered"], errors="coerce")
        key = ["customer_code", "item_code"]
        merged = pd.merge(df, self.stats, on=key, how="left")
        # Compute z‑score where stats are available
        merged["qty_zscore"] = (merged["qty_ordered"] - merged["qty_mean"]) / merged["qty_std"]
        # Determine flags
        def flag_row(row) -> str:
            flags: List[str] = []
            # Unknown item means no history entry
            if pd.isna(row["qty_mean"]):
                flags.append("UNKNOWN_ITEM")
            else:
                # Avoid division by zero: std could be zero if all historical quantities
                # are the same; in that case any difference counts as anomaly
                z = row["qty_zscore"]
                if pd.isna(z):
                    pass
                else:
                    # treat infinite z as anomaly
                    if not np.isfinite(z) or abs(z) > self.qty_zscore_threshold:
                        flags.append("QTY_ANOM")
            return ", ".join(flags)

        merged["flags"] = merged.apply(flag_row, axis=1)
        return merged