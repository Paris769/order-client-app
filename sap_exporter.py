"""Export matched orders to an Excel format suitable for SAP import.

This module defines a simple exporter that takes the matched orders DataFrame
and selects columns appropriate for import into SAP (or similar ERP). The
exporter can be customised to include additional fields or formats by
adjusting the mapping in ``export_to_sap``. The default export includes
customer code, item code, item description and ordered quantity.
"""

from __future__ import annotations

import pandas as pd  # type: ignore
from typing import Dict, List


def export_to_sap(df: pd.DataFrame, path: str = "sap_export.xlsx") -> str:
    """Write the given DataFrame to an Excel file for SAP.

    Parameters
    ----------
    df: pandas.DataFrame
        The matched orders DataFrame containing at least the columns
        ``customer_code``, ``item_code``, ``item_description`` and
        ``qty_ordered``.
    path: str, optional
        Path of the Excel file to create. Defaults to ``sap_export.xlsx`` in
        the current working directory.

    Returns
    -------
    str
        The path to the written file.
    """
    # Map internal columns to SAP import columns; adjust as required.
    export_df = pd.DataFrame({
        "CustomerCode": df.get("customer_code"),
        "ItemCode": df.get("item_code"),
        "ItemDescription": df.get("item_description"),
        "Quantity": df.get("qty_ordered"),
    })
    export_df.to_excel(path, index=False)
    return path