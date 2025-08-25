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
    """Parse a PDF order confirmation into a DataFrame.

    This parser extracts product lines from PDF order confirmations with
    minimal assumptions about the layout. It supports two extraction
    strategies:

    1. When the ``pdfplumber`` library is available, it uses
       ``page.extract_text()`` to obtain the raw text for each page.
    2. If ``pdfplumber`` is not installed or fails, it falls back to
       calling the ``pdftotext`` command‐line tool with the ``-layout``
       option to preserve the relative spacing of columns. ``pdftotext``
       must be available in the system for this fallback to work.

    Each line of extracted text is scanned for a decimal quantity of the
    form ``\d+,\d+`` (comma as the decimal separator). The first such
    occurrence on a line is interpreted as the ordered quantity. The
    portion of the line preceding the quantity is treated as the product
    description. If that prefix begins with an alphanumeric token that
    contains at least one digit, that token is interpreted as a vendor
    item code and removed from the description. Any lines containing
    headers (e.g. ``HSN Code``) or totals are ignored. Price and total
    columns are not captured because they are not required for the
    downstream matching logic.

    Parameters
    ----------
    uploaded_file: file‑like
        Either a file path or a file object provided by Streamlit.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``item_code``, ``item_description`` and
        ``qty_ordered``. The caller is responsible for adding the
        ``customer_code`` column before matching.
    """
    # Read the PDF into a list of lines using pdfplumber or pdftotext
    lines: List[str] = []
    # Determine whether uploaded_file is a streamlit UploadedFile or a path
    if hasattr(uploaded_file, "read"):
        # We need to read the bytes into memory once. We'll keep a copy
        # because pdfplumber and pdftotext both require a file path or bytes.
        data = uploaded_file.read()
        pdf_bytes = io.BytesIO(data)
        file_path: Optional[str] = None
    else:
        pdf_bytes = None
        file_path = str(uploaded_file)
    # Try using pdfplumber first if available
    tried_plumber = False
    if pdfplumber is not None:
        try:
            tried_plumber = True
            if pdf_bytes is not None:
                pdf_obj = pdfplumber.open(io.BytesIO(data))
            else:
                pdf_obj = pdfplumber.open(file_path)  # type: ignore[arg-type]
            with pdf_obj as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    lines.extend(text.split("\n"))
        except Exception:
            # Fall back to pdftotext
            lines = []
    # If lines is still empty, try using pdftotext
    if not lines:
        try:
            import subprocess
            # Write bytes to a temp file if necessary
            if pdf_bytes is not None and file_path is None:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                file_path = tmp_path
            # Use -layout to preserve columns
            output = subprocess.check_output(["pdftotext", "-layout", file_path or "", "-"], text=True)
            lines = output.split("\n")
        except Exception:
            raise RuntimeError(
                "Impossibile estrarre testo dal PDF: assicurati che 'pdfplumber' o 'pdftotext' siano disponibili."
            )
    rows: List[Dict[str, Optional[str]]] = []
    # Pattern to find the first decimal quantity with a comma
    qty_regex = re.compile(r"(\d+,\d+)")
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # Skip common non‑item lines
        lowered = line.lower()
        if any(key in lowered for key in ["hsn code", "delivery date", "net total", "grand net total", "item no."]):
            continue
        m = qty_regex.search(line)
        if not m:
            continue
        qty_str = m.group(1)
        # Everything before the quantity is considered the prefix
        prefix = line[: m.start()].strip()
        # Attempt to split the prefix into an item code and description. If the
        # first token contains any digit it is treated as the code.
        tokens = prefix.split()
        code: Optional[str] = None
        desc_tokens: List[str] = []
        if tokens:
            first_tok = tokens[0]
            if re.search(r"\d", first_tok):
                code = first_tok
                desc_tokens = tokens[1:]
            else:
                desc_tokens = tokens
        description = " ".join(desc_tokens).strip()
        # Convert quantity: replace comma decimal with dot; ignore thousands separators
        try:
            qty_val = float(qty_str.replace(".", "").replace(",", "."))
        except Exception:
            qty_val = None
        row: Dict[str, Optional[str]] = {
            "item_code": code,
            "item_description": description or prefix,  # fallback to prefix if no tokens
            "qty_ordered": qty_val,
        }
        rows.append(row)
    return pd.DataFrame(rows)