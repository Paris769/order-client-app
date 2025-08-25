"""Utility functions to parse different order file formats.

This module provides functions to read orders from Excel, PDF and text files.
The goal is to normalise disparate order formats into a unified DataFrame
with the following columns:

    - customer_code
    - item_code
    - item_description
    - qty_shipped
    - qty_ordered

Only the relevant columns are extracted; any additional columns from the
source files are ignored. Users may need to adjust the heuristics below
to handle bespoke layouts (e.g. customer‑specific PDF templates). The
parsers attempt to make sensible guesses based on column headers and text
patterns.
"""

from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import pandas as pd  # type: ignore
try:
    # pdfplumber may not be installed in all environments; import lazily
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # type: ignore


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to map input column names to canonical names.

    The function looks for keywords in the column names of the provided
    DataFrame and renames them to canonical names used by the rest of the
    application. If a required column cannot be located the function will
    leave it missing and callers should handle missing columns appropriately.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame whose columns should be renamed.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with columns renamed where possible.
    """
    colmap: Dict[str, str] = {}
    for col in df.columns:
        c = str(col).lower().strip()
        # customer code
        if any(k in c for k in ["cliente", "customer", "fornitore"]):
            colmap[col] = "customer_code"
        # item / product code
        elif any(k in c for k in ["articolo", "item", "product", "codice"]):
            # avoid remapping customer column again
            if col not in colmap:
                colmap[col] = "item_code"
        # description
        elif any(k in c for k in ["descr", "description"]):
            colmap[col] = "item_description"
        # quantity shipped
        elif any(k in c for k in ["sped", "shipped"]):
            colmap[col] = "qty_shipped"
        # quantity ordered
        elif any(k in c for k in ["qtaord", "ordinata", "ordered"]):
            colmap[col] = "qty_ordered"
        # quantity generic catch‑all
        elif "qty" in c and "ordered" not in c:
            colmap[col] = "qty_ordered"
    return df.rename(columns=colmap)


def parse_excel(uploaded_file) -> pd.DataFrame:
    """Parse an Excel file into a standardised DataFrame.

    This parser attempts to locate the most relevant columns based on
    common keywords in the header names. It picks the first match for
    each canonical field to avoid duplicates when multiple columns
    contain similar keywords (e.g. ``Codice cliente/fornitore`` and
    ``Fornitore`` both include "fornitore").

    Parameters
    ----------
    uploaded_file: file-like
        An uploaded file object from Streamlit or a file path. pandas
        supports file‑like objects directly.

    Returns
    -------
    pandas.DataFrame
        DataFrame with canonical columns where detected. Missing
        columns are omitted.
    """
    df = pd.read_excel(uploaded_file)
    used_cols: set[str] = set()

    def find_col(keywords: List[str]) -> Optional[str]:
        """Find the first column containing any of the given keywords."""
        for col in df.columns:
            if col in used_cols:
                continue
            c = str(col).lower()
            for kw in keywords:
                if kw in c:
                    used_cols.add(col)
                    return col
        return None

    # Define keyword lists for each canonical field
    customer_col = find_col(["cliente/fornitore", "cliente", "customer"])
    item_col = find_col(["codice articolo", "articolo", "item", "product"])
    description_col = find_col(["descr", "descrizione", "description"])
    qty_shipped_col = find_col(["qtasped", "sped", "spedita", "shipped"])
    qty_ordered_col = find_col(["qtaord", "ordinata", "ordered", "quantità ordinata", "qta ord"])

    data: Dict[str, pd.Series] = {}
    if customer_col:
        data["customer_code"] = df[customer_col]
    if item_col:
        data["item_code"] = df[item_col]
    if description_col:
        data["item_description"] = df[description_col]
    if qty_shipped_col:
        data["qty_shipped"] = df[qty_shipped_col]
    if qty_ordered_col:
        data["qty_ordered"] = df[qty_ordered_col]
    # Build DataFrame
    if not data:
        # fallback: use normalise to attempt a rescue
        fallback = _normalise_column_names(df)
        cols = [c for c in [
            "customer_code",
            "item_code",
            "item_description",
            "qty_shipped",
            "qty_ordered",
        ] if c in fallback.columns]
        return fallback[cols].copy()
    return pd.DataFrame(data)


def parse_text(uploaded_file) -> pd.DataFrame:
    """Parse a plain text file into a DataFrame.

    The parser uses a simple regular expression to capture rows of the
    following approximate format (whitespace separated):

        <item_code> <item_description> <qty_ordered>

    If a line cannot be parsed it will be ignored.

    Parameters
    ----------
    uploaded_file: file-like
        An uploaded file object from Streamlit or a file path.

    Returns
    -------
    pandas.DataFrame
        DataFrame with canonical columns. Customer code must be
        injected elsewhere as text files rarely include it.
    """
    content: str
    # If a plain string is provided, treat it as the content itself
    if isinstance(uploaded_file, str):
        content = uploaded_file
    elif hasattr(uploaded_file, "read"):
        # streamlit returns a BytesIO; decode bytes
        raw = uploaded_file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("latin-1")
    else:
        with open(uploaded_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    rows: List[Dict[str, Optional[str]]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split the line into tokens. We expect the last token to be the quantity.
        tokens = line.split()
        if not tokens:
            continue
        # Attempt to parse the last token as an integer quantity
        qty_token = tokens[-1]
        try:
            qty_val = float(qty_token.replace(",", "."))
        except ValueError:
            # not a quantity; skip this line
            continue
        # Determine whether the first token is an item code. If it contains any digits,
        # treat it as a code; otherwise assume no code is provided and the description
        # encompasses all tokens except the last one.
        first = tokens[0]
        has_digit = bool(re.search(r"\d", first))
        if has_digit and len(tokens) >= 3:
            code = first
            desc_tokens = tokens[1:-1]
        else:
            code = None
            desc_tokens = tokens[:-1]
        item_description = " ".join(desc_tokens).strip()
        rows.append({
            "item_code": code,
            "item_description": item_description,
            "qty_ordered": qty_val,
        })
    return pd.DataFrame(rows)


def parse_pdf(uploaded_file) -> pd.DataFrame:
    """Parse a PDF file into a DataFrame.

    This parser attempts to extract order lines from a variety of PDF
    layouts. It uses ``pdfplumber`` to extract free‑form text from each
    page, then applies heuristics and regular expressions to identify
    product descriptions, quantities and prices. Many customer order
    confirmations follow a pattern similar to:

    ``
    Item No. Vendor No. Item                             Qty Unit        Price    Total
    12345    6789       SOME PRODUCT DESCRIPTION         4,00 Each        5,00    20,00
    HSN Code 1234
    ``

    On other documents the ``Vendor No.`` column may be blank and a
    separate ``HSN Code`` row appears below each item line. In such
    cases only the description and quantity can be extracted and the
    application will attempt to map the description to a known item.

    Parameters
    ----------
    uploaded_file: file-like
        An uploaded file object from Streamlit or a file path.

    Returns
    -------
    pandas.DataFrame
        DataFrame with at most the columns ``item_code``,
        ``item_description``, ``qty_ordered`` and optionally ``price``.
        The caller is responsible for injecting a customer code before
        further processing.
    """
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber non è installato nell'ambiente. Impossibile analizzare i file PDF."
        )
    rows: List[Dict[str, Optional[str]]] = []
    # Prepare file‑like object for pdfplumber
    if hasattr(uploaded_file, "read"):
        pdf_reader = pdfplumber.open(io.BytesIO(uploaded_file.read()))
    else:
        pdf_reader = pdfplumber.open(uploaded_file)
    # Regular expression for item lines. We allow an optional numeric
    # vendor code at the start of the line. The description may
    # contain any characters, followed by a quantity expressed as
    # digits with a comma decimal separator (e.g. ``4,00``). After
    # the quantity we skip over the unit text and capture price and
    # total if present. ``HSN Code`` lines are ignored later.
    item_pattern = re.compile(
        r"^(?:(?P<item_code>\d+)\s+)?"
        r"(?P<item_description>.+?)\s+"
        r"(?P<qty>[\d]+,[\d]+)\s+"
        r".*?"  # unit and other text
        r"(?P<price>[\d]+,[\d]+)\s+"
        r"(?P<total>[\d]+,[\d]+)"
        r"$"
    )
    with pdf_reader as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                # Skip empty lines and HSN Code rows
                if not line or line.lower().startswith("hsn code"):
                    continue
                m = item_pattern.match(line)
                if not m:
                    continue
                groups = m.groupdict()
                desc = groups.get("item_description", "").strip()
                qty_str = groups.get("qty") or ""
                price_str = groups.get("price") or ""
                code = groups.get("item_code")
                # Normalise decimal separators: replace thousands separators (.) and
                # decimal comma with a dot for numeric conversion
                def to_float(s: str) -> Optional[float]:
                    try:
                        return float(s.replace(".", "").replace(",", "."))
                    except Exception:
                        return None
                qty_val = to_float(qty_str)
                price_val = to_float(price_str)
                row: Dict[str, Optional[str]] = {
                    "item_code": code if code else None,
                    "item_description": desc,
                    "qty_ordered": qty_val,
                }
                # Optionally include price for downstream analytics
                if price_val is not None:
                    row["price"] = price_val
                rows.append(row)
    return pd.DataFrame(rows)