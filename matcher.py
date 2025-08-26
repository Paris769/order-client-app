"""
Logic to compare new orders against historical data.
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


class OrderMatcher:
    """Compare new orders with historical quantity patterns."""

    def __init__(self, history_df: pd.DataFrame, qty_zscore_threshold: float = 3.0) -> None:
        """
        Parameters
        ----------
        history_df: pandas.DataFrame
            Historical orders with at least the columns ``customer_code``,
            ``item_code``, ``item_description`` and ``qty_ordered``.
        qty_zscore_threshold: float, optional
            Absolute z-score above which a quantity is considered anomalous.
            Default is 3.0 (roughly 3 standard deviations).
        """
        # Make a copy of history and normalise key types
        self.history = history_df.copy()
        self.qty_zscore_threshold = qty_zscore_threshold

        # Ensure merge keys are consistently typed as strings (preservando NaN)
        for k in ["customer_code", "item_code"]:
            if k in self.history.columns:
                self.history[k] = self.history[k].astype(str)

        # qty numerica
        qty_series = pd.to_numeric(self.history.get("qty_ordered"), errors="coerce")
        self.history["qty_ordered_num"] = qty_series

        # Stats per customer+item
        key = ["customer_code", "item_code"]
        grouped = (
            self.history.groupby(key)["qty_ordered_num"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "qty_mean", "std": "qty_std"})
        )
        self.stats = grouped

        # Stats per descrizione (fallback)
        desc_grouped = (
            self.history.groupby("item_description")["qty_ordered_num"]
            .agg(["mean", "std"])
            .dropna(subset=["mean", "std"])
            .reset_index()
        )
        # Dizionario {"desc_lower": {"mean":..., "std":...}}
        self.desc_stats: Dict[str, Dict[str, float]] = {}
        for _, row in desc_grouped.iterrows():
            self.desc_stats[str(row["item_description"]).lower()] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }

        # Descrizione canonica per codice (la più frequente)
        code_desc_counts = (
            self.history.groupby(["item_code", "item_description"])["qty_ordered_num"]
            .count()
            .reset_index(name="cnt")
        )
        code_to_desc: Dict[str, str] = {}
        for code, grp in code_desc_counts.groupby("item_code"):
            idx = grp["cnt"].idxmax()
            code_to_desc[str(code)] = str(code_desc_counts.loc[idx, "item_description"])
        self.code_to_desc = code_to_desc

        # Mapping globale: per descrizione → codice con quantità totale massima
        global_totals = (
            self.history.groupby(["item_description", "item_code"])["qty_ordered_num"]
            .sum()
            .reset_index(name="qty_total")
        )
        global_desc_to_code: Dict[str, str] = {}
        for desc, grp in global_totals.groupby("item_description"):
            max_idx = grp["qty_total"].idxmax()
            global_desc_to_code[str(desc).lower()] = str(grp.loc[max_idx, "item_code"])

        # Mapping per cliente: (customer, descr_lower) → codice con qty_total massima
        qty_totals = (
            self.history.groupby(["customer_code", "item_code"])["qty_ordered_num"]
            .sum()
            .reset_index(name="qty_total")
        )
        hist_with_totals = pd.merge(
            self.history, qty_totals, on=["customer_code", "item_code"], how="left"
        )
        cust_desc_to_code: Dict[Tuple[str, str], str] = {}
        for (cust, desc), grp in hist_with_totals.groupby(["customer_code", "item_description"]):
            idx_max = grp["qty_total"].idxmax()
            cust_desc_to_code[(str(cust), str(desc).lower())] = str(grp.loc[idx_max, "item_code"])

        self.global_desc_to_code = global_desc_to_code
        self.cust_desc_to_code = cust_desc_to_code

        # Quantità totali per codice (peso per similarità)
        self.code_qty_totals: Dict[str, float] = (
            self.history.groupby("item_code")["qty_ordered_num"].sum().astype(float).to_dict()
        )

        # Elenco (desc_lower, code) per eventuali strategie globali
        self._desc_code_pairs: List[Tuple[str, str]] = [
            (str(r["item_description"]).lower(), str(r["item_code"]))
            for _, r in self.history.iterrows()
        ]

        # Codici acquistati per cliente (per restringere il dominio nel matching cliente)
        self.customer_codes: Dict[str, List[str]] = (
            self.history.groupby("customer_code")["item_code"]
            .apply(lambda s: [str(c) for c in s])
            .to_dict()
        )

        # Dizionario rapido code -> desc canonica (se non già costruito)
        # (già self.code_to_desc)
        # fine __init__

    # ---------------------------- utilità di similarità ----------------------------

    @staticmethod
    def _normalise(text: str) -> List[str]:
        """
        Normalizza la descrizione in token:
        - minuscole
        - mantiene numeri
        - spezza pattern '72x110' in ['72','110']
        - rimuove punteggiatura
        """
        s = str(text).lower()
        # separa 'x' con spazi
        s = s.replace("×", "x").replace(" x ", " x ")
        for ch in [",", ";", "(", ")", "[", "]", "{", "}", ":", ".", "/", "\\", "-", "_", "+", "'", '"']:
            s = s.replace(ch, " ")
        tokens: List[str] = []
        for tok in s.split():
            if "x" in tok:
                parts = [p for p in tok.split("x") if p]
                tokens.extend(parts)
            else:
                tokens.append(tok)
        # rimuove token troppo corti (eccetto numeri)
        out: List[str] = []
        for t in tokens:
            if len(t) > 1 or t.isdigit():
                out.append(t)
        return out

    @staticmethod
    def _numeric_signature(tokens: List[str]) -> List[int]:
        """estrae tutti i numeri interi dai token, ordinati (per confronto di dimensioni)"""
        nums: List[int] = []
        for t in tokens:
            if t.isdigit():
                try:
                    nums.append(int(t))
                except Exception:
                    pass
        return sorted(nums)

    def _score_similarity(self, a: str, b: str) -> float:
        """
        Combina Jaccard token + vicinanza numerica (dimensioni).
        Ritorna uno score in [0,1].
        """
        ta = self._normalise(a)
        tb = self._normalise(b)
        sa, sb = set(ta), set(tb)
        inter = sa & sb
        union = sa | sb
        jacc = len(inter) / len(union) if union else 0.0

        na, nb = self._numeric_signature(ta), self._numeric_signature(tb)
        if not na or not nb:
            num_sim = 0.0
        else:
            # allinea per lunghezza
            m = min(len(na), len(nb))
            if m == 0:
                num_sim = 0.0
            else:
                diffs = [abs(na[i] - nb[i]) for i in range(m)]
                max_base = max([max(na), max(nb), 1])
                num_sim = max(0.0, 1.0 - (sum(diffs) / (m * max_base)))
        # pesi: 0.7 jaccard, 0.3 numerico
        return 0.7 * jacc + 0.3 * num_sim

    # ---------------------------- finder per cliente / globale ----------------------------

    def _find_similar_code_for_customer(
        self,
        cust: str,
        description: str,
        used_codes: Optional[set[str]] = None,
        threshold: float = 0.25,
    ) -> Optional[str]:
        """
        Cerca il codice più simile tra quelli già acquistati dal cliente.
        Usa punteggio di similarità e peso per quantità totale acquistata.
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
            # pesatura per importanza del codice
            weight = 1.0 + (self.code_qty_totals.get(code, 0.0) / max(1.0, max(self.code_qty_totals.values(), default=1.0)))
            score = sim * weight
            if score > best_score:
                best_score = score
                best_code = code

        if best_code and best_score >= threshold:
            return best_code
        return None

    def _find_similar_code(
        self,
        description: str,
        used_codes: Optional[set[str]] = None,
        threshold: float = 0.30,
    ) -> Optional[str]:
        """
        Fallback globale: cerca il codice più plausibile su tutto lo storico.
        """
        desc_lower = str(description or "").lower()
        best_code = None
        best_score = 0.0
        for desc, code in self._desc_code_pairs:
            if used_codes and code in used_codes:
                continue
            sim = self._score_similarity(desc_lower, desc)
            weight = 1.0 + (self.code_qty_totals.get(code, 0.0) / max(1.0, max(self.code_qty_totals.values(), default=1.0)))
            score = sim * weight
            if score > best_score:
                best_score = score
                best_code = code
        if best_code and best_score >= threshold:
            return best_code
        return None

    # ---------------------------- matching principale ----------------------------

    def match(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ritorna un DataFrame con colonne aggiuntive:
          - qty_mean, qty_std, qty_zscore
          - flags (UNKNOWN_ITEM, DESC_MATCH, QTY_ANOM)
          - desc_mapped (bool)
        """
        df = orders_df.copy()

        # Traccia se il codice è stato mappato da descrizione
        df["desc_mapped"] = False

        # Cast chiavi a stringa
        for k in ["customer_code", "item_code"]:
            if k in df.columns:
                df[k] = df[k].astype(str)

        # -------------- Primo pass: prova a mappare i codici sconosciuti --------------

        used_codes_in_loop: set[str] = set()

        for idx, row in df.iterrows():
            cust = str(row.get("customer_code"))
            code = str(row.get("item_code")) if row.get("item_code") not in [None, "None", "nan", "NaN"] else None
            if code and (cust, code) in set(zip(self.history["customer_code"], self.history["item_code"])):
                # già visto in storico
                continue

            desc_lower = str(row.get("item_description") or "").lower()
            mapped_code: Optional[str] = None

            # 1) exact match cliente su descrizione
            key_cust_exact = (cust, desc_lower)
            if key_cust_exact in self.cust_desc_to_code:
                mapped_code = self.cust_desc_to_code[key_cust_exact]
            else:
                # 2) fuzzy cliente (difflib) cutoff alto
                cust_desc_keys = [k[1] for k in self.cust_desc_to_code.keys() if k[0] == cust]
                if cust_desc_keys:
                    matches = difflib.get_close_matches(desc_lower, cust_desc_keys, n=1, cutoff=0.8)
                    if matches:
                        mapped_code = self.cust_desc_to_code[(cust, matches[0])]

                # 2.5) token/numbers cliente (Jaccard+numeri)
                if mapped_code is None:
                    best_for_cust = self._find_similar_code_for_customer(
                        cust=cust,
                        description=desc_lower,
                        used_codes=used_codes_in_loop,
                        threshold=0.25,  # più permissiva per il cliente
                    )
                    if best_for_cust:
                        mapped_code = best_for_cust

            # 3) exact globale su descrizione
            if mapped_code is None and desc_lower in self.global_desc_to_code:
                mapped_code = self.global_desc_to_code[desc_lower]

            # 4) fuzzy globale (difflib) cutoff molto alto
            if mapped_code is None:
                global_keys = list(self.global_desc_to_code.keys())
                matches = difflib.get_close_matches(desc_lower, global_keys, n=1, cutoff=0.9)
                if matches:
                    mapped_code = self.global_desc_to_code[matches[0]]

            # applica mappatura
            if mapped_code:
                df.at[idx, "item_code"] = mapped_code
                df.at[idx, "desc_mapped"] = True
                used_codes_in_loop.add(mapped_code)
                # aggiorna descrizione con quella canonica di storico
                canon_desc = self.code_to_desc.get(mapped_code)
                if canon_desc is not None:
                    df.at[idx, "item_description"] = canon_desc

        # -------------- Secondo pass: fallback finale per i None rimasti --------------

        used_codes: set[str] = set()
        for idx, row in df.iterrows():
            code = row.get("item_code")
            if not code or str(code).lower() in {"none", "nan", ""}:
                desc = row.get("item_description")
                cust = str(row.get("customer_code"))

                best = self._find_similar_code_for_customer(
                    cust=cust,
                    description=desc,
                    used_codes=used_codes,
                    threshold=0.25,
                )
                if not best:
                    best = self._find_similar_code(
                        description=desc,
                        used_codes=used_codes,
                        threshold=0.30,
                    )

                if best:
                    df.at[idx, "item_code"] = best
                    df.at[idx, "desc_mapped"] = True
                    canon_desc = self.code_to_desc.get(best)
                    if canon_desc is not None:
                        df.at[idx, "item_description"] = canon_desc
                    used_codes.add(best)

        # Cast di sicurezza
        for k in ["customer_code", "item_code"]:
            if k in df.columns:
                df[k] = df[k].astype(str)

        # qty numerica
        if "qty_ordered" in df.columns:
            df["qty_ordered"] = pd.to_numeric(df["qty_ordered"], errors="coerce")

        # Join con stats
        key = ["customer_code", "item_code"]
        merged = pd.merge(df, self.stats, on=key, how="left")

        # Riempi con stats per descrizione quando mancano quelle per codice
        merged["desc_used"] = False
        na_idx = merged["qty_mean"].isna()
        for j in merged[na_idx].index:
            d = str(merged.at[j, "item_description"]).lower()
            s = self.desc_stats.get(d)
            if s:
                merged.at[j, "qty_mean"] = s["mean"]
                merged.at[j, "qty_std"] = s["std"]
                merged.at[j, "desc_used"] = True

        # calcolo z-score (attenzione a std=0 / NaN)
        merged["qty_zscore"] = (merged["qty_ordered"] - merged["qty_mean"]) / merged["qty_std"]

        # ----------------------- flags -----------------------
        def flag_row(row) -> str:
            flags: List[str] = []

            # sconosciuto: nessuna stat e nessuna mappatura descrizione
            if pd.isna(row["qty_mean"]) and not row.get("desc_used") and not row.get("desc_mapped"):
                flags.append("UNKNOWN_ITEM")
            else:
                # se mappato da descrizione o usate stats descrizione
                if row.get("desc_used") or row.get("desc_mapped"):
                    flags.append("DESC_MATCH")

                # evita div/0: std potrebbe essere 0
                z = row.get("qty_zscore")
                if pd.isna(z):
                    pass
                else:
                    if not np.isfinite(z) or abs(z) > self.qty_zscore_threshold:
                        flags.append("QTY_ANOM")

            return ", ".join(flags)

        merged["flags"] = merged.apply(flag_row, axis=1)
        return merged
