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
    if hasattr(uploaded_file, "read"):
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
        # Pattern: code description quantity
        m = re.match(r"(\S+)\s+(.+?)\s+(\d+)(?:\s+|$)", line)
        if m:
            item_code, item_description, qty_ordered = m.groups()
            rows.append({
                "item_code": item_code.strip(),
                "item_description": item_description.strip(),
                "qty_ordered": int(qty_ordered),
            })
    return pd.DataFrame(rows)


def parse_pdf(uploaded_file) -> pd.DataFrame:
    """Parse a PDF file into a DataFrame.

    This function uses pdfplumber to extract text from each page and applies
    a simple regular expression similar to the text parser. PDF layouts vary
    widely, so you may need to customise the patterns to suit your suppliers.

    Parameters
    ----------
    uploaded_file: file-like
        An uploaded file object from Streamlit or a file path.

    Returns
    -------
    pandas.DataFrame
        DataFrame with canonical columns. Customer code must be
        injected elsewhere as PDFs often do not include it on each row.
    """
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber non è installato nell'ambiente. Impossibile analizzare i file PDF."
        )
    rows: List[Dict[str, Optional[str]]] = []
    # pdfplumber requires a file-like object with a read method
    if hasattr(uploaded_file, "read"):
        pdf_reader = pdfplumber.open(io.BytesIO(uploaded_file.read()))
    else:
        pdf_reader = pdfplumber.open(uploaded_file)
    with pdf_reader as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                m = re.match(r"(\S+)\s+(.+?)\s+(\d+)(?:\s+|$)", line)
                if m:
                    item_code, item_description, qty_ordered = m.groups()
                    rows.append({
                        "item_code": item_code.strip(),
                        "item_description": item_description.strip(),
                        "qty_ordered": int(qty_ordered),
                    })
    return pd.DataFrame(rows)