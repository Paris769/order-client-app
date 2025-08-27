"""
Logic to compare new orders against historical data, with hierarchical matching and fallback.

This module defines an ``OrderMatcher`` class that takes historical order data and
matches new order lines against it.  It uses a hierarchy of matching strategies:

1.  **Exact match** on the combination of customer code and item description.
    If a customer has already purchased an item with the same description, that
    item code is reused.

2.  **Fuzzy match** on the customer’s past descriptions using ``difflib``.  This
    catches minor typos or variations (e.g. missing accents, pluralisation).

3.  **Token‑based similarity** using Jaccard on normalised tokens and a simple
    numeric signature (dimensions such as ``10x20`` → ``[10, 20]``).  This stage
    scores candidate item codes purchased by the same customer and selects the
    best match above a configurable threshold.

4.  **Fallback global match** across all historical items.  First an exact
    match on description, then a fuzzy match, and finally a token‑based match
    across all items.  This ensures that a plausible code is always suggested
    even for completely new customers or descriptions.

When a description match is used to assign an item code, the official
historical description is substituted and a boolean flag ``desc_mapped`` is
set.  Quantity anomalies are flagged via a z‑score comparison against the
customer’s historical purchasing pattern; if no customer history is available
for a given code, global statistics per description are used.
"""

from __future__ import annotations

import difflib
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


class OrderMatcher:
    """Compare new orders with historical quantity patterns and assign item codes."""

    def __init__(self, history_df: pd.DataFrame, qty_zscore_threshold: float = 3.0) -> None:
        """
        Parameters
        ----------
        history_df: pandas.DataFrame
            Historical orders with at least the columns ``customer_code``, ``item_code``,
            ``item_description`` and ``qty_ordered``.
        qty_zscore_threshold: float, optional
            Absolute z‑score above which a quantity is considered anomalous.
            Default is 3.0 (roughly 3 standard deviations).
        """
        # Make a copy of history and normalise key types
        self.history = history_df.copy()
        for col in ["customer_code", "item_code"]:
            if col in self.history.columns:
                self.history[col] = self.history[col].astype(str)

        # Quantity numeric
        self.history["qty_ordered_num"] = pd.to_numeric(
            self.history.get("qty_ordered"), errors="coerce"
        )

        # Stats per customer + item (mean and std)
        key = ["customer_code", "item_code"]
        grouped = (
            self.history.groupby(key)["qty_ordered_num"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "qty_mean", "std": "qty_std"})
        )
        self.stats = grouped

        # Stats per description (fallback when no customer stats exist)
        desc_grouped = (
            self.history.groupby("item_description")["qty_ordered_num"]
            .agg(["mean", "std"])
            .dropna(subset=["mean", "std"])
            .reset_index()
        )
        # Map normalised description → {"mean": float, "std": float}
        self.desc_stats: Dict[str, Dict[str, float]] = {}
        for _, row in desc_grouped.iterrows():
            self.desc_stats[str(row["item_description"]).lower()] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }

        # Canonical description per code: pick the most frequent description for each item code
        # Determine a canonical description per item_code: choose the most frequent
        code_to_desc: Dict[str, str] = {}
        for code, hgrp in self.history.groupby("item_code"):
            # count occurrences of each description for this code
            counts = hgrp["item_description"].value_counts()
            # pick the description with the highest count (idxmax returns the index label)
            canonical_desc = str(counts.idxmax()) if not counts.empty else ""
            code_to_desc[str(code)] = canonical_desc
        self.code_to_desc = code_to_desc

        # Quantity totals per item_code (used to weight similarity)
        qty_totals = (
            self.history.groupby("item_code")["qty_ordered_num"].sum().reset_index(name="qty_total")
        )
        self.code_qty_totals: Dict[str, float] = {
            str(row["item_code"]): float(row["qty_total"])
            for _, row in qty_totals.iterrows()
        }
        # Avoid zero max
        max_total = max(self.code_qty_totals.values()) if self.code_qty_totals else 1.0
        if max_total <= 0:
            max_total = 1.0
        self._max_total_qty = max_total

        # Global description → predominant code (by quantity)
        # Compute total quantities per (description, code)
        desc_code_totals = (
            self.history
            .groupby(["item_description", "item_code"])["qty_ordered_num"]
            .sum()
            .reset_index(name="qty_total")
        )
        global_desc_to_code: Dict[str, str] = {}
        for desc, grp in desc_code_totals.groupby("item_description"):
            # find the code with the highest total quantity for this description
            idx_max = grp["qty_total"].idxmax()
            row = grp.loc[idx_max]
            global_desc_to_code[str(desc).lower()] = str(row["item_code"])
        self.global_desc_to_code = global_desc_to_code

        # Per‑customer mapping (customer, description) → predominant code
        qty_totals_cust = (
            self.history.groupby(["customer_code", "item_code"])["qty_ordered_num"]
            .sum()
            .reset_index(name="qty_total")
        )
        hist_with_totals = pd.merge(
            self.history,
            qty_totals_cust,
            on=["customer_code", "item_code"],
            how="left",
        )
        cust_desc_to_code: Dict[Tuple[str, str], str] = {}
        for (cust, desc), grp in hist_with_totals.groupby(["customer_code", "item_description"]):
            # pick the item_code with highest qty_total for this customer/description
            idx_max = grp["qty_total"].idxmax()
            cust_desc_to_code[(str(cust), str(desc).lower())] = str(
                hist_with_totals.loc[idx_max, "item_code"]
            )
        self.cust_desc_to_code = cust_desc_to_code

        # Reverse mapping: customer → list of codes purchased (used to restrict search domain).
        # We store codes as a list for each customer; converting to a set when checking
        # membership allows us to quickly determine whether a given code has been
        # purchased by a customer.  This mapping is central to the initial pass
        # of order matching, as we prioritise codes that a customer has previously
        # ordered over descriptions.
        self.customer_codes: Dict[str, List[str]] = (
            self.history.groupby("customer_code")["item_code"]
            .apply(lambda s: [str(c) for c in s])
            .to_dict()
        )

        # Prepare a list of (lowercase description, code) pairs for global token‑based similarity
        self.desc_code_pairs: List[Tuple[str, str]] = [
            (str(r["item_description"]).lower(), str(r["item_code"]))
            for _, r in self.history.iterrows()
        ]

        # Threshold for anomaly detection
        self.qty_zscore_threshold = float(qty_zscore_threshold)

    # ----------------------------------------------------------------------
    # Normalisation utilities

    @staticmethod
    def _normalise(text: str) -> List[str]:
        """Normalise a description into a list of tokens.

        Lowercase, keep numbers, split on punctuation and treat dimension
        patterns like ``50x60`` as separate numbers.
        """
        s = str(text).lower()
        # separate 'x' (dimensions) by spaces to split numbers
        s = s.replace("x", " x ").replace(" × ", " x ")
        for ch in [
            ",",
            ".",
            "?",
            "!",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "/",
            "\\",
            "-",
            "_",
            "=",
            "+",
            "*",
            "\"",
        ]:
            s = s.replace(ch, " ")
        tokens: List[str] = []
        for tok in s.split():
            # if 'x' inside token, split by 'x'
            if "x" in tok:
                parts = [p for p in tok.split("x") if p]
                tokens.extend(parts)
            else:
                tokens.append(tok)
        return tokens

    @staticmethod
    def _numeric_signature(tokens: Iterable[str]) -> List[int]:
        """Extract numeric dimensions from tokens.

        Returns a sorted list of integers found in tokens.  Non‑numeric tokens
        are ignored.
        """
        nums: List[int] = []
        for t in tokens:
            if t.isdigit():
                try:
                    nums.append(int(t))
                except Exception:
                    pass
        return sorted(nums)

    def _score_similarity(self, a: str, b: str) -> float:
        """Compute a similarity score in [0, 1] between two descriptions.

        Combines Jaccard similarity on normalised tokens with a numeric
        signature similarity (for dimensions).  A simple heuristic is used:
        Jaccard on token sets plus 0.5 times a dimension similarity
        (comparing numeric lists element‑wise).
        """
        ta = self._normalise(a)
        tb = self._normalise(b)
        sa, sb = set(ta), set(tb)
        inter = sa & sb
        union = sa | sb
        # Compute Jaccard on all tokens as before
        jacc = len(inter) / len(union) if union else 0.0

        # If there are no alphabetical tokens in common, penalise the score.
        # This avoids matching descriptions that only share numbers (e.g. "300") but are otherwise unrelated.
        # Determine tokens containing at least one alphabetic character.
        alpha_a = {t for t in sa if any(c.isalpha() for c in t)}
        alpha_b = {t for t in sb if any(c.isalpha() for c in t)}
        if alpha_a and alpha_b and not (alpha_a & alpha_b):
            # Reset Jaccard and numeric similarity to zero if no alphabetic overlap.
            # Without any shared words, matching purely on numbers (e.g. "300") can lead to spurious
            # matches such as equating a rotolo pellicola to a mocho simply because both contain "300".
            jacc = 0.0
            # Flag to zero out numeric similarity later
            no_alpha_overlap = True
        else:
            no_alpha_overlap = False

        na = self._numeric_signature(ta)
        nb = self._numeric_signature(tb)
        if not na or not nb or no_alpha_overlap:
            # Either no numeric signature or intentionally disable numeric similarity when there
            # is no alphabetical token overlap.
            num_sim = 0.0
        else:
            m = min(len(na), len(nb))
            if m == 0:
                num_sim = 0.0
            else:
                # align lists by truncating longer one
                num_sim = 1.0 - (sum(abs(na[i] - nb[i]) for i in range(m)) / sum(max(na[i], nb[i]) for i in range(m)))
                num_sim = max(0.0, num_sim)
        # combine: weight numeric similarity half as much as Jaccard overlap
        score = jacc + 0.5 * num_sim
        # clip to [0, 1]
        return max(0.0, min(1.0, score))

    # ----------------------------------------------------------------------
    # Finder utilities (customer and global)

    def _find_similar_code_for_customer(
        self,
        cust: str,
        description: str,
        used_codes: Optional[set[str]] = None,
        threshold: float = 0.25,
    ) -> Optional[str]:
        """Return the best matching item code for a customer using token similarity.

        Only item codes already purchased by the customer are considered.
        A weighting based on total quantity emphasises frequently ordered
        items.  If no candidate exceeds the threshold, ``None`` is returned.
        """
        candidates = set(self.customer_codes.get(str(cust), []))
        if used_codes:
            candidates -= set(used_codes)
        if not candidates:
            return None

        desc_lower = str(description or "").lower()
        best_code = None
        best_score = 0.0
        for code in candidates:
            canon_desc = self.code_to_desc.get(code, "")
            sim = self._score_similarity(desc_lower, canon_desc.lower())
            # weight by total quantity (avoid zero division).  Use a square root to reduce the influence of very
            # popular items so that similarity drives the match more than sales volume.
            qty = self.code_qty_totals.get(code, 0.0)
            weight = 1.0 + (qty / self._max_total_qty) ** 0.5
            score = sim * weight
            if score > best_score:
                best_score = score
                best_code = code
        # return only if above threshold
        if best_code and best_score > threshold:
            return best_code
        return None

    def _find_similar_code(
        self,
        description: str,
        used_codes: Optional[set[str]] = None,
        threshold: float = 0.30,
    ) -> Optional[str]:
        """Global fallback: search all item codes by token similarity.

        A weighting based on total quantity emphasises popular items.  If no
        candidate exceeds the threshold, ``None`` is returned.
        """
        desc_lower = str(description or "").lower()
        best_code = None
        best_score = 0.0
        for desc, code in self.desc_code_pairs:
            if used_codes and code in used_codes:
                continue
            sim = self._score_similarity(desc_lower, desc)
            qty = self.code_qty_totals.get(code, 0.0)
            # As above, dampen the weighting effect of very large quantities to prioritise descriptive similarity.
            weight = 1.0 + (qty / self._max_total_qty) ** 0.5
            score = sim * weight
            if score > best_score:
                best_score = score
                best_code = code
        if best_code and best_score > threshold:
            return best_code
        return None

    # ----------------------------------------------------------------------
    # Main matching routine

    def match(
        self,
        orders_df: pd.DataFrame,
        cust_desc_threshold: float = 0.25,
        global_desc_threshold: float = 0.30,
    ) -> pd.DataFrame:
        """Assign item codes and anomaly flags to a new orders DataFrame.

        Returns a DataFrame with additional columns:

        - ``qty_mean``, ``qty_std``, ``qty_zscore``: statistics for the matched
          item/customer combination (or description fallback).
        - ``flags``: comma‑separated string of flags:
            - ``UNKNOWN_ITEM`` if the item could not be mapped at all.
            - ``DESC_MATCH`` if the item code was inferred from the description (rather than provided).
            - ``QTY_ANOM`` if the quantity deviates from the historical pattern beyond the z‑score threshold.
        - ``desc_mapped``: bool indicating whether the description was replaced by the canonical one.
        """
        df = orders_df.copy()
        # normalise key columns to string
        for col in ["customer_code", "item_code"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # First pass: map unknown or missing item codes using description
        used_codes_in_pass: set[str] = set()
        # keep track of which rows used description mapping
        df["desc_mapped"] = False

        for idx, row in df.iterrows():
            cust = str(row.get("customer_code"))
            # normalise provided code; treat empty strings and various 'None' representations as missing
            raw_code = row.get("item_code")
            code = None
            if raw_code not in [None, "None", "nan", "NaN", ""]:
                code = str(raw_code)
            desc = str(row.get("item_description") or "")

            # If a code was provided, accept it only if it exists among the codes previously
            # purchased by this customer.  Otherwise, treat it as missing and fall back to
            # description-based matching.  This prevents mis-assigned codes from being blindly
            # accepted and enforces the "code first" matching policy per customer.
            cust_codes = set(self.customer_codes.get(cust, []))
            if code and code in cust_codes:
                # Update the description to the canonical one for the code
                canon_desc = self.code_to_desc.get(code)
                if canon_desc:
                    df.at[idx, "item_description"] = canon_desc
                # No mapping needed; proceed to next row
                continue
            else:
                # Provided code is either missing or not previously purchased by this customer.
                # Remove it to trigger the description-based matching logic below.
                df.at[idx, "item_code"] = None

            desc_lower = desc.lower()
            mapped_code: Optional[str] = None

            # (1) Exact per-customer match on description
            key_cust_exact = (cust, desc_lower)
            if key_cust_exact in self.cust_desc_to_code:
                mapped_code = self.cust_desc_to_code[key_cust_exact]
            else:
                # (2) Fuzzy per-customer match with difflib
                cust_desc_keys = [k for k in self.cust_desc_to_code.keys() if k[0] == cust]
                if cust_desc_keys:
                    matches = difflib.get_close_matches(
                        desc_lower,
                        [k[1] for k in cust_desc_keys],
                        n=1,
                        cutoff=0.8,
                    )
                    if matches:
                        mapped_code = self.cust_desc_to_code[(cust, matches[0])]
                # (3) Token/numeric similarity per customer
                if mapped_code is None:
                    mapped_code = self._find_similar_code_for_customer(
                        cust=cust,
                        description=desc_lower,
                        used_codes=used_codes_in_pass,
                        threshold=cust_desc_threshold,
                    )
            # (4) Exact global match on description
            if mapped_code is None and desc_lower in self.global_desc_to_code:
                mapped_code = self.global_desc_to_code[desc_lower]
            # (5) Fuzzy global match
            if mapped_code is None:
                global_keys = list(self.global_desc_to_code.keys())
                matches = difflib.get_close_matches(
                    desc_lower,
                    global_keys,
                    n=1,
                    cutoff=0.9,
                )
                if matches:
                    mapped_code = self.global_desc_to_code[matches[0]]
            # (6) Token/numeric global similarity
            if mapped_code is None:
                mapped_code = self._find_similar_code(
                    description=desc_lower,
                    used_codes=used_codes_in_pass,
                    threshold=global_desc_threshold,
                )
            # At this point, mapped_code may still be None; if so, try again without threshold
            if mapped_code is None:
                mapped_code = self._find_similar_code(
                    description=desc_lower,
                    used_codes=used_codes_in_pass,
                    threshold=0.0,
                )

            # apply mapping if found
            if mapped_code:
                df.at[idx, "item_code"] = mapped_code
                df.at[idx, "desc_mapped"] = True
                used_codes_in_pass.add(mapped_code)
                # replace description with canonical description
                canon_desc = self.code_to_desc.get(mapped_code)
                if canon_desc:
                    df.at[idx, "item_description"] = canon_desc

        # Second pass: ensure any remaining missing codes are filled with a plausible suggestion
        used_codes_overall: set[str] = set()
        for idx, row in df.iterrows():
            code = row.get("item_code")
            if not code or code in ["None", "nan", "NaN", ""]:
                cust = str(row.get("customer_code"))
                desc = str(row.get("item_description") or "")
                best = self._find_similar_code_for_customer(
                    cust=cust,
                    description=desc.lower(),
                    used_codes=used_codes_overall,
                    threshold=cust_desc_threshold,
                )
                if not best:
                    best = self._find_similar_code(
                        description=desc.lower(),
                        used_codes=used_codes_overall,
                        threshold=global_desc_threshold,
                    )
                if not best:
                    # no threshold: always suggest a code
                    best = self._find_similar_code(
                        description=desc.lower(),
                        used_codes=used_codes_overall,
                        threshold=0.0,
                    )
                if best:
                    df.at[idx, "item_code"] = best
                    used_codes_overall.add(best)
                    df.at[idx, "desc_mapped"] = True
                    canon_desc = self.code_to_desc.get(best)
                    if canon_desc:
                        df.at[idx, "item_description"] = canon_desc

        # Cast keys to string again (safety)
        for col in ["customer_code", "item_code"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Quantity numeric conversion
        if "qty_ordered" in df.columns:
            df["qty_ordered"] = pd.to_numeric(df["qty_ordered"], errors="coerce")

        # Join with stats (mean and std) on customer_code and item_code
        key = ["customer_code", "item_code"]
        merged = pd.merge(df, self.stats, on=key, how="left")
        merged["desc_used"] = False

        # Fill missing stats using description when available
        for i, row in merged.iterrows():
            if pd.isna(row["qty_mean"]):
                d = str(row["item_description"]).lower()
                s = self.desc_stats.get(d)
                if s:
                    merged.at[i, "qty_mean"] = s["mean"]
                    merged.at[i, "qty_std"] = s["std"]
                    merged.at[i, "desc_used"] = True

        # Compute z‑score; beware division by zero / NaN
        merged["qty_zscore"] = (merged["qty_ordered"] - merged["qty_mean"]) / merged["qty_std"]

        # Flag rows
        def flag_row(row: pd.Series) -> str:
            flags: List[str] = []
            # unknown if no stats and no description mapping occurred
            if pd.isna(row["qty_mean"]) and not row.get("desc_used") and not row.get("desc_mapped"):
                flags.append("UNKNOWN_ITEM")
            else:
                # description mapping used
                if row.get("desc_used") or row.get("desc_mapped"):
                    flags.append("DESC_MATCH")
                z = row.get("qty_zscore")
                if pd.isna(z):
                    pass
                else:
                    if not np.isfinite(z) or abs(z) > self.qty_zscore_threshold:
                        flags.append("QTY_ANOM")
            return ", ".join(flags)

        merged["flags"] = merged.apply(flag_row, axis=1)
        return merged
