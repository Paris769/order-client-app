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

        # Build a mapping of codes purchased by each customer.  This will be used
        # to restrict description-based matching to items that the customer has
        # actually ordered before.  Without this restriction the matcher can
        # incorrectly assign a new order to a product purchased by another
        # customer simply because the description is vaguely similar.  Keys and
        # codes are cast to strings to ensure consistent comparisons.
        self.customer_codes: Dict[str, List[str]] = (
            self.history.groupby("customer_code")
            ["item_code"]
            .apply(lambda series: [str(c) for c in series])
            .to_dict()
        )
        # Compute total quantity ordered for each (customer_code, item_code) pair.
        # This is used to weight description-similarity matches: codes with
        # higher historical quantities are preferred.  Casting keys to strings
        # ensures consistent lookup keys when codes are numeric.
        cust_totals = (
            self.history
            .groupby(["customer_code", "item_code"])["qty_ordered_num"]
            .sum()
            .astype(float)
        )
        self.cust_code_qty_totals: Dict[tuple[str, str], float] = {}
        for (cust, code), qty in cust_totals.items():
            self.cust_code_qty_totals[(str(cust), str(code))] = qty

        # Build a canonical description mapping for each item code.  For codes
        # associated with multiple descriptions across the history, choose the
        # description with the highest total quantity ordered as the canonical
        # representation. This ensures that when we map a new row's item_code
        # based on description we can also update its description to match the
        # historical record. Without this mapping, newly matched rows could
        # retain their original description (e.g. from a PDF) which may differ
        # from the description stored in the sales history. Casting codes to
        # strings ensures consistent lookup keys.
        code_desc_totals = (
            self.history.groupby(["item_code", "item_description"])  # type: ignore[list-item]
            ["qty_ordered_num"]
            .sum()
            .reset_index(name="qty_total")
        )
        self.code_to_desc: Dict[str, str] = {}
        for code, group in code_desc_totals.groupby("item_code"):
            # Select the description with the maximum total ordered quantity
            idx = group["qty_total"].idxmax()
            desc = str(group.loc[idx, "item_description"])
            self.code_to_desc[str(code)] = desc

        # Build a set of known (customer_code, item_code) pairs to quickly identify
        # whether a new order item exists in the history. Casting both parts to
        # string ensures consistent comparisons when codes are numeric.
        self.known_pairs = set(
            zip(
                self.history["customer_code"].astype(str),
                self.history["item_code"].astype(str),
            )
        )

    def _normalise(self, text: str) -> List[str]:
        """Normalise a description into a list of lower‑case tokens.

        This helper removes punctuation and splits on whitespace. It is used
        to compute Jaccard similarity between descriptions. Numbers and
        very short tokens (length < 2) are ignored as they tend not to be
        discriminative for product matching.

        Parameters
        ----------
        text: str
            The text to normalise.

        Returns
        -------
        List[str]
            A list of normalised tokens.
        """
        import re
        # Replace non-word characters with spaces and split on 'x' to retain
        # numerical dimensions (e.g. '50x60' -> '50 60').  We deliberately
        # keep numeric tokens because dimensional differences (e.g. 72x110 vs
        # 50x60) are important when matching products such as bags or sacs.
        cleaned = re.sub(r"[^\w]+", " ", str(text).lower())
        cleaned = cleaned.replace("x", " ")  # separate dimension tokens
        raw_tokens = cleaned.split()
        # Keep tokens longer than 1 character.  Do not discard numeric tokens,
        # but ignore very short or single-character tokens which tend not to be
        # discriminative.  We do not filter out digits anymore because they
        # convey important size information.
        tokens = [t for t in raw_tokens if len(t) > 1]
        return tokens

    def _find_similar_code(self, description: str, used_codes: Optional[set[str]] = None) -> Optional[str]:
        """Find the best matching item_code for a given description using token/Jaccard similarity.

        When no match is found via the customer or global description mappings, this
        fallback searches all historical descriptions and selects the code with
        the highest Jaccard similarity (on normalised token sets) to the target
        description. Similarity scores are weighted by the relative quantity
        purchased for each code to prefer frequently ordered items. Codes
        already used for other unknown items can be excluded by passing
        ``used_codes``.

        Parameters
        ----------
        description: str
            The item description from a new order for which we need to guess a code.
        used_codes: set[str], optional
            A set of item codes that have already been assigned to other rows. If
            provided, these codes will not be considered for matching.

        Returns
        -------
        Optional[str]
            The most plausible item code based on description similarity, or ``None``
            if no suitable match exists.
        """
        if not self._desc_code_pairs:
            return None
        target_tokens = set(self._normalise(description))
        if not target_tokens:
            return None
        max_qty = max(self.code_qty_totals.values()) if self.code_qty_totals else 1.0
        best_code: Optional[str] = None
        best_score: float = 0.0
        used_codes = used_codes or set()
        # Compute similarity for each historical description
        for desc, code in self._desc_code_pairs:
            if code in used_codes:
                continue
            desc_tokens = set(self._normalise(desc))
            if not desc_tokens:
                continue
            # Jaccard similarity: intersection over union
            inter = target_tokens & desc_tokens
            if not inter:
                continue  # skip completely disjoint descriptions
            union = target_tokens | desc_tokens
            jac = len(inter) / len(union)
            if jac <= 0:
                continue
            qty_weight = self.code_qty_totals.get(code, 0.0) / max_qty
            score = jac * (1.0 + qty_weight)
            if score > best_score:
                best_score = score
                best_code = code
        # If no candidate found using Jaccard, fall back to SequenceMatcher as a last resort
        if best_code is None:
            from difflib import SequenceMatcher
            target = " ".join(target_tokens)
            for desc, code in self._desc_code_pairs:
                if code in used_codes:
                    continue
                sim = SequenceMatcher(None, target, desc).ratio()
                qty_weight = self.code_qty_totals.get(code, 0.0) / max_qty
                score = sim * (1.0 + qty_weight)
                if score > best_score:
                    best_score = score
                    best_code = code
        return best_code

    def _find_similar_code_for_customer(
        self,
        cust: str,
        description: str,
        used_codes: Optional[set[str]] = None,
        threshold: float = 0.3,
    ) -> Optional[str]:
        """Find the most similar item_code for a given description based on a specific customer's history.

        This method restricts matching to the set of item codes previously
        purchased by the given customer.  Similarity is computed using
        Jaccard similarity between token sets (including numeric tokens) and
        weighted by the proportion of quantity ordered for each code.  Codes
        that have been assigned to other rows can be excluded via ``used_codes``.
        Only return a code if the best weighted similarity score exceeds
        ``threshold``; otherwise return None to indicate that no suitable
        match exists.

        Parameters
        ----------
        cust: str
            Customer code for which to find a matching item code.
        description: str
            The item description from a new order.
        used_codes: set[str], optional
            Codes to exclude from consideration (already used for other rows).
        threshold: float, optional
            Minimum weighted similarity score required to consider a match.

        Returns
        -------
        Optional[str]
            The best matching item code or None if no match exceeds the
            threshold.
        """
        # Retrieve the list of codes this customer has ordered
        codes = self.customer_codes.get(str(cust), [])
        if not codes:
            return None
        used_codes = used_codes or set()
        target_tokens = set(self._normalise(description))
        if not target_tokens:
            return None
        # Determine maximum quantity ordered by this customer for normalisation
        max_qty = 1.0
        # Compute max per-customer qty
        for code in codes:
            qty = self.cust_code_qty_totals.get((str(cust), code), 0.0)
            if qty > max_qty:
                max_qty = qty
        best_code: Optional[str] = None
        best_score: float = 0.0
        # Compute similarity for each candidate code
        for code in codes:
            if code in used_codes:
                continue
            canon_desc = self.code_to_desc.get(code)
            if not canon_desc:
                continue
            desc_tokens = set(self._normalise(canon_desc))
            if not desc_tokens:
                continue
            inter = target_tokens & desc_tokens
            if not inter:
                continue
            union = target_tokens | desc_tokens
            jac = len(inter) / len(union)
            if jac <= 0:
                continue
            # Add a numeric similarity bonus: match numeric tokens (e.g. dimensions)
            target_nums = [int(t) for t in target_tokens if t.isdigit()]
            cand_nums = [int(t) for t in desc_tokens if t.isdigit()]
            numeric_bonus = 0.0
            if target_nums:
                for tn in target_nums:
                    if tn in cand_nums:
                        numeric_bonus += 1.0
                    else:
                        # find nearest candidate number and award partial bonus if close (<=2 units)
                        diffs = [abs(tn - cn) for cn in cand_nums] if cand_nums else []
                        if diffs:
                            min_diff = min(diffs)
                            if min_diff <= 2:
                                numeric_bonus += 0.5
                numeric_score = numeric_bonus / len(target_nums)
            else:
                numeric_score = 0.0
            # Combine Jaccard similarity with numeric similarity
            combined_sim = jac + numeric_score
            qty_weight = self.cust_code_qty_totals.get((str(cust), code), 0.0) / max_qty
            score = combined_sim * (1.0 + qty_weight)
            if score > best_score:
                best_score = score
                best_code = code
        # Only accept a match if the score exceeds the threshold
        if best_score >= threshold:
            return best_code
        return None

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
        # Track codes that have been assigned via description matching to avoid
        # mapping multiple unknown items to the same code when using similarity.
        used_codes_in_loop: set[str] = set()
        for idx, row in df.iterrows():
            cust = str(row.get("customer_code"))
            # item_code may be None or 'nan'
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
                # 2) Fuzzy match on customer‑specific descriptions using close_matches
                cust_desc_keys = [k[1] for k in self.cust_desc_to_code.keys() if k[0] == cust]
                if cust_desc_keys:
                    # Use a high cutoff to avoid loosely matching descriptions (e.g. 50x60 vs 72x110)
                    matches = difflib.get_close_matches(desc_lower, cust_desc_keys, n=1, cutoff=0.8)
                    if matches:
                        matched_desc = matches[0]
                        mapped_code = self.cust_desc_to_code[(cust, matched_desc)]
            # 2.5) Token/Jaccard-based match restricted to the customer's purchase history
            if mapped_code is None:
                best_for_cust = self._find_similar_code_for_customer(
                    cust,
                    row.get("item_description"),
                    used_codes=used_codes_in_loop,
                    threshold=0.4,
                )
                if best_for_cust:
                    mapped_code = best_for_cust
            # 3) Fallback to global exact match
            if mapped_code is None and desc_lower in self.global_desc_to_code:
                mapped_code = self.global_desc_to_code[desc_lower]
            # 4) Fallback to global fuzzy match
            if mapped_code is None:
                global_keys = list(self.global_desc_to_code.keys())
                # Use a high cutoff to avoid loosely matching descriptions across all customers
                matches = difflib.get_close_matches(desc_lower, global_keys, n=1, cutoff=0.9)
                if matches:
                    mapped_code = self.global_desc_to_code[matches[0]]
            if mapped_code:
                # Assign the mapped code and mark that it came from a description-based match
                df.at[idx, "item_code"] = mapped_code
                df.at[idx, "desc_mapped"] = True
                # Record that this code was used to avoid duplicate assignment
                used_codes_in_loop.add(mapped_code)
                # Also update the description to the canonical one from the history if available.
                canon_desc = self.code_to_desc.get(mapped_code)
                if canon_desc is not None:
                    df.at[idx, "item_description"] = canon_desc

        # Additional fallback: for any remaining unknown codes (i.e. None or blank),
        # attempt to infer the most plausible code using per-customer similarity.
        # If no suitable code is found for the customer (score below threshold), leave
        # the code as missing (it will be flagged as UNKNOWN_ITEM).
        used_codes: set[str] = set()
        for idx, row in df.iterrows():
            code = row.get("item_code")
            # Treat missing codes (None, NaN, empty string) as unknown
            if not code or str(code).lower() in {"none", "nan", ""}:
                desc = row.get("item_description")
                cust = str(row.get("customer_code"))
                best = self._find_similar_code_for_customer(
                    cust,
                    desc,
                    used_codes=used_codes,
                    threshold=0.4,
                )
                if best:
                    df.at[idx, "item_code"] = best
                    df.at[idx, "desc_mapped"] = True
                    canon_desc = self.code_to_desc.get(best)
                    if canon_desc is not None:
                        df.at[idx, "item_description"] = canon_desc
                    used_codes.add(best)
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