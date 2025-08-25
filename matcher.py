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
import difflib
from typing import List, Dict, Optional


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
        # Make a copy of history and normalise key types
        self.history = history_df.copy()
        self.qty_zscore_threshold = qty_zscore_threshold

        # Ensure merge keys are consistently typed as strings. Without this
        # normalisation, merging new orders with historical stats can raise
        # ValueError when one side is numeric and the other is object (e.g.
        # int64 vs object). Cast both customer and item codes to string
        # before computing statistics.
        key = ["customer_code", "item_code"]
        for k in key:
            if k in self.history.columns:
                # Cast to string; preserve NaNs (astype(str) would convert
                # NaN to 'nan'), so use pandas string type where possible
                self.history[k] = self.history[k].astype(str)

        # Filter rows with numeric quantities and compute stats per customer+item
        qty_series = pd.to_numeric(self.history.get("qty_ordered"), errors="coerce")
        self.history["qty_ordered_num"] = qty_series
        grouped = self.history.groupby(key)["qty_ordered_num"].agg(["mean", "std"]).reset_index()
        grouped.rename(columns={"mean": "qty_mean", "std": "qty_std"}, inplace=True)
        self.stats = grouped
        # Compute description-level statistics. If item_code is missing,
        # statistics based on the item description will be used instead.
        desc_grouped = (
            self.history.groupby("item_description")["qty_ordered_num"]
            .agg(["mean", "std"])
            .reset_index()
        )
        # Remove entries where mean or std is NaN
        desc_grouped = desc_grouped.dropna(subset=["mean", "std"])
        # Build a dict keyed by lower-case description for quick lookup
        self.desc_stats: Dict[str, Dict[str, float]] = {}
        for _, row in desc_grouped.iterrows():
            desc_key = str(row["item_description"]).lower()
            self.desc_stats[desc_key] = {"mean": row["mean"], "std": row["std"]}

        # Precompute mappings for description‑based lookups.
        # Global mapping from description to the most frequent item_code across all customers.
        global_desc_to_code: Dict[str, str] = {}
        # Mapping from (customer_code, description) to the most frequent item_code for that customer.
        cust_desc_to_code: Dict[tuple[str, str], str] = {}
        # Also compute purchase frequency for weighting: total quantity ordered per (customer, item)
        qty_totals = (
            self.history
            .groupby(["customer_code", "item_code"])["qty_ordered_num"]
            .sum()
            .reset_index(name="qty_total")
        )
        # Aggregate total quantity ordered for each (description, code) pair globally.
        # We prefer codes with the highest total quantity over simple frequency
        global_totals = (
            self.history
            .groupby(["item_description", "item_code"])["qty_ordered_num"]
            .sum()
            .reset_index(name="qty_total")
        )
        for desc, group in global_totals.groupby("item_description"):
            # Select the code with the maximum total quantity ordered for this description
            max_idx = group["qty_total"].idxmax()
            code_str = str(group.loc[max_idx, "item_code"])
            desc_str = str(desc).lower()
            global_desc_to_code[desc_str] = code_str
        # Build per‑customer description mapping. For each customer we find the most
        # commonly purchased code for each description. If multiple codes share
        # the same description for the same customer, pick the one with the highest
        # total quantity ordered.
        # First merge quantity totals into the history for easy lookup
        hist_with_totals = pd.merge(
            self.history,
            qty_totals,
            on=["customer_code", "item_code"],
            how="left",
        )
        # Group by customer and description
        for (cust, desc), group in hist_with_totals.groupby(["customer_code", "item_description"]):
            # Select the row with maximum qty_total; if tie, take first occurrence
            idx_max = group["qty_total"].idxmax()
            code_str = str(group.loc[idx_max, "item_code"])
            cust_desc_to_code[(str(cust), str(desc).lower())] = code_str
        self.global_desc_to_code = global_desc_to_code
        self.cust_desc_to_code = cust_desc_to_code

        # Precompute total quantity ordered per item_code across all customers. This will be
        # used to weight description‑similarity matches when a code is not found via
        # customer or global description mappings. Casting codes to string ensures
        # consistent keys when codes are numeric.
        self.code_qty_totals: Dict[str, float] = (
            self.history.groupby("item_code")["qty_ordered_num"].sum().astype(float).to_dict()
        )
        # Prepare a list of (description_lower, item_code) tuples for similarity search.
        # We keep duplicates because descriptions may map to multiple codes; weighting
        # by total quantity will favour more frequently ordered codes during matching.
        self._desc_code_pairs: List[tuple[str, str]] = [
            (str(row["item_description"]).lower(), str(row["item_code"]))
            for _, row in self.history.iterrows()
        ]

        # Build a set of known (customer_code, item_code) pairs to quickly identify
        # whether a new order item exists in the history. Casting both parts to
        # string ensures consistent comparisons when codes are numeric.
        self.known_pairs = set(
            zip(
                self.history["customer_code"].astype(str),
                self.history["item_code"].astype(str),
            )
        )

    def _find_similar_code(self, description: str) -> Optional[str]:
        """Find the best matching item_code for a given description using fuzzy similarity.

        This fallback is used when no match is found via exact or fuzzy description mapping.
        It computes a similarity ratio between the target description and every description
        in the history. Scores are weighted by the total quantity ordered for each code
        to bias the selection towards frequently purchased items. The code with the
        highest weighted similarity score is returned. If no descriptions are present
        in the history, ``None`` is returned.

        Parameters
        ----------
        description: str
            The item description from a new order for which we need to guess a code.

        Returns
        -------
        Optional[str]
            The most plausible item code based on description similarity, or ``None``
            if the history is empty.
        """
        if not self._desc_code_pairs:
            return None
        from difflib import SequenceMatcher  # imported lazily to avoid overhead
        target = str(description).lower().strip()
        # Precompute the maximum total quantity to normalise weights.
        max_qty = max(self.code_qty_totals.values()) if self.code_qty_totals else 1.0
        best_code: Optional[str] = None
        best_score: float = 0.0
        # Iterate through all known description/code pairs. We do not deduplicate
        # descriptions here because the quantity weight can differ per code even if
        # descriptions are identical across different item codes.
        for desc, code in self._desc_code_pairs:
            if not desc:
                continue
            # Compute similarity ratio between descriptions. SequenceMatcher returns
            # a float in [0, 1] representing the proportion of matching blocks.
            sim = SequenceMatcher(None, target, desc).ratio()
            # Weight by the relative quantity ordered for this code. This biases
            # towards codes with larger historical order volumes.
            qty_weight = self.code_qty_totals.get(code, 0.0) / max_qty
            # Combine similarity and weight. A small epsilon ensures that even
            # codes with zero quantity can be selected if no better match exists.
            score = sim * (1.0 + qty_weight)
            if score > best_score:
                best_score = score
                best_code = code
        return best_code

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
        # Make a copy of the orders and prepare for processing
        df = orders_df.copy()
        # Track whether the item code was mapped from the description
        df["desc_mapped"] = False
        # Cast merge keys to string early to ensure comparisons and replacements work reliably
        # We do not cast qty here to allow numeric comparison later
        for k in ["customer_code", "item_code"]:
            if k in df.columns:
                df[k] = df[k].astype(str)
        # Attempt to map unknown item codes using the description. For each row
        # where the (customer_code, item_code) pair is not present in the historical data,
        # find a known code based on the description. We prioritise mappings from
        # this customer's purchase history; if none exist we fall back to a global
        # description mapping. Exact matches are preferred over fuzzy matches.
        for idx, row in df.iterrows():
            cust = str(row.get("customer_code"))
            code = str(row.get("item_code")) if row.get("item_code") is not None else None
            # Skip if we already know this customer and code combination
            if code and (cust, code) in self.known_pairs:
                continue
            desc_lower = str(row.get("item_description")).lower()
            mapped_code: Optional[str] = None
            # 1) Exact match on customer‑specific description
            key_cust_exact = (cust, desc_lower)
            if key_cust_exact in self.cust_desc_to_code:
                mapped_code = self.cust_desc_to_code[key_cust_exact]
            else:
                # 2) Fuzzy match on customer‑specific descriptions
                # Build candidate descriptions for this customer
                cust_desc_keys = [k[1] for k in self.cust_desc_to_code.keys() if k[0] == cust]
                if cust_desc_keys:
                    matches = difflib.get_close_matches(desc_lower, cust_desc_keys, n=1, cutoff=0.7)
                    if matches:
                        # Map back to full key to retrieve code
                        matched_desc = matches[0]
                        mapped_code = self.cust_desc_to_code[(cust, matched_desc)]
            # 3) Fallback to global exact match
            if mapped_code is None and desc_lower in self.global_desc_to_code:
                mapped_code = self.global_desc_to_code[desc_lower]
            # 4) Fallback to global fuzzy match
            if mapped_code is None:
                global_keys = list(self.global_desc_to_code.keys())
                matches = difflib.get_close_matches(desc_lower, global_keys, n=1, cutoff=0.8)
                if matches:
                    mapped_code = self.global_desc_to_code[matches[0]]
            if mapped_code:
                df.at[idx, "item_code"] = mapped_code
                df.at[idx, "desc_mapped"] = True

        # Additional fallback: for any remaining unknown codes (i.e. None or 'None'),
        # attempt to infer the most plausible code by computing similarity between
        # the current description and all historical descriptions, weighted by the
        # total quantity ordered for each candidate code. This ensures that even
        # previously unseen descriptions are mapped to the closest available item.
        for idx, row in df.iterrows():
            code = row.get("item_code")
            # Treat string 'None' or empty strings as missing codes
            if not code or str(code).lower() in {"none", "nan", ""}:
                desc = row.get("item_description")
                best = self._find_similar_code(desc)
                if best:
                    df.at[idx, "item_code"] = best
                    df.at[idx, "desc_mapped"] = True
        # Cast merge keys to string again in case mapping introduced new codes
        for k in ["customer_code", "item_code"]:
            if k in df.columns:
                df[k] = df[k].astype(str)
        # Ensure quantity is numeric
        df["qty_ordered"] = pd.to_numeric(df["qty_ordered"], errors="coerce")
        # Merge with precomputed statistics on customer+item
        key = ["customer_code", "item_code"]
        merged = pd.merge(df, self.stats, on=key, how="left")
        # Fill missing stats using description-level statistics
        merged["desc_used"] = False
        na_idx = merged["qty_mean"].isna()
        for idx in merged[na_idx].index:
            desc = str(merged.at[idx, "item_description"]).lower()
            stats = self.desc_stats.get(desc)
            if stats:
                merged.at[idx, "qty_mean"] = stats["mean"]
                merged.at[idx, "qty_std"] = stats["std"]
                merged.at[idx, "desc_used"] = True
        # Compute z‑score where stats are available (including desc stats)
        merged["qty_zscore"] = (merged["qty_ordered"] - merged["qty_mean"]) / merged["qty_std"]

        # Determine flags. A row is unknown if no stats were found and we did not map the code or use desc stats.
        def flag_row(row) -> str:
            flags: List[str] = []
            # Unknown item means no historical stats and no description-based match
            if pd.isna(row["qty_mean"]) and not row.get("desc_used") and not row.get("desc_mapped"):
                flags.append("UNKNOWN_ITEM")
            else:
                # If a description mapping or stats were used, flag accordingly
                if row.get("desc_used") or row.get("desc_mapped"):
                    flags.append("DESC_MATCH")
                # Avoid division by zero: std could be zero if all historical quantities are the same;
                # in that case any difference counts as anomaly.
                z = row["qty_zscore"]
                if pd.isna(z):
                    pass
                else:
                    if not np.isfinite(z) or abs(z) > self.qty_zscore_threshold:
                        flags.append("QTY_ANOM")
            return ", ".join(flags)

        merged["flags"] = merged.apply(flag_row, axis=1)
        return merged