"""Streamlit interface for auditing customer orders.

This application allows users to upload a historical order file and one or
more new order documents (Excel, PDF, text). It then analyses the new
orders against historical quantities to identify unknown items and
quantity anomalies. Users can download an Excel file containing the
results in a format suitable for importing into SAP or other ERP systems.

Example usage:

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
from typing import List, Optional, Tuple, Dict

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore

from parsers import parse_excel, parse_pdf, parse_text
import subprocess
import re
import tempfile

# ---------------------------------------------------------------------
# Helper to parse PDFs with Italian headers
def parse_pdf_flexible(uploaded_file) -> pd.DataFrame:
    """Parse a PDF with flexible header detection.

    This function is a fallback parser for PDF order confirmations where
    the header columns are labelled in Italian (e.g. ``Articolo`` and
    ``Qta``) rather than the English ``Item``/``Qty`` that the default
    ``parse_pdf`` looks for. It uses the ``pdftotext`` command with
    ``-layout`` to extract text and then applies heuristics similar to
    ``parse_pdf`` but with extended header detection.

    Parameters
    ----------
    uploaded_file: file‑like
        A Streamlit uploaded file or a filesystem path.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``item_code``, ``item_description`` and
        ``qty_ordered``.
    """
    try:
        # Read bytes from the uploaded file or treat as path
        if hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
            # Write to a temporary file for pdftotext
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                file_path = tmp.name
        else:
            file_path = str(uploaded_file)
        # Extract layout‑preserved text
        output = subprocess.check_output(["pdftotext", "-layout", file_path, "-"], text=True)
        lines = output.split("\n")
        rows: List[Dict[str, Optional[str]]] = []
        qty_regex = re.compile(r"(\d+,\d+)")
        capture = False
        for raw in lines:
            line = raw.rstrip()
            if not line:
                continue
            lower = line.lower()
            # Start capturing when encountering a header containing item/artic and qty/qta
            if not capture:
                has_item = ("item" in lower) or ("artic" in lower)
                has_qty = ("qty" in lower) or ("qta" in lower) or ("quant" in lower)
                has_vendor = ("vendor" in lower) or ("fornitore" in lower)
                if (has_item and has_qty) or (has_vendor and has_qty):
                    capture = True
                continue
            # Stop capturing at totals or footer lines (English or Italian)
            if any(key in lower for key in ["net total", "grand net total", "delivery date", "totale"]):
                break
            # Skip non‑product lines
            if "hsn code" in lower or "reference" in lower:
                continue
            m = qty_regex.search(line)
            if not m:
                continue
            qty_str = m.group(1)
            prefix = line[: m.start()].rstrip()
            collapsed = re.sub(r"\s{2,}", " ", prefix.strip())
            tokens = collapsed.split() if collapsed else []
            code: Optional[str] = None
            description: str = collapsed
            if tokens:
                first_tok = tokens[0]
                # If the first token contains a digit treat it as a vendor code
                if re.search(r"\d", first_tok):
                    code = first_tok
                    description = " ".join(tokens[1:]).strip()
                else:
                    description = collapsed
            try:
                qty_val = float(qty_str.replace(".", "").replace(",", "."))
            except Exception:
                qty_val = None
            rows.append(
                {
                    "item_code": code,
                    "item_description": description,
                    "qty_ordered": qty_val,
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        # On any error, return empty DataFrame
        return pd.DataFrame()
from matcher import OrderMatcher
from sap_exporter import export_to_sap


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Order Audit App", layout="wide")
    st.title("Order Audit App")
    st.write(
        """
        Carica un file Excel con lo storico degli ordini (cliente, articolo,
        descrizione, quantità spedita e quantità ordinata) e una o più nuove
        conferme d'ordine in formato Excel, PDF o testo. L'applicazione
        confronterà i nuovi ordini con lo storico per evidenziare articoli
        sconosciuti o quantità anomale. Alla fine potrai scaricare un file
        pronto per l'import in SAP.
        """
    )

    # Sidebar inputs
    st.sidebar.header("Dati di ingresso")
    hist_file = st.sidebar.file_uploader(
        "Carica lo storico degli ordini (Excel)",
        type=["xls", "xlsx"],
    )
    # Input for default customer code (for files that don't contain it)
    default_customer_code = st.sidebar.text_input(
        "Codice cliente predefinito per file senza colonna cliente",
        value="",
    )

    new_files = st.sidebar.file_uploader(
        "Carica nuovi ordini (PDF, testo o Excel)",
        type=["pdf", "txt", "xls", "xlsx", "csv"],
        accept_multiple_files=True,
    )

    # Optional text area for manual order entry. Users can paste their order
    # details here when no file is available. Expected format per line:
    # <codice> <descrizione> <quantità>
    manual_text = st.sidebar.text_area(
        "Oppure inserisci l'ordine in forma testuale (una riga per ordine: codice descrizione quantità)",
        value="",
        height=150,
    )

    if hist_file is None:
        st.info("Per favore carica un file Excel con lo storico degli ordini.")
        return

    # Parse historical data
    try:
        hist_df = parse_excel(hist_file)
    except Exception as e:
        st.error(f"Errore nel caricamento dello storico: {e}")
        return

    if hist_df.empty:
        st.warning(
            "Lo storico degli ordini sembra vuoto o non contiene le colonne attese."
        )
        return

    st.sidebar.success(f"Storico caricato: {len(hist_df)} righe")

    # Show a preview of the historical data
    with st.expander("Anteprima dello storico", expanded=False):
        st.dataframe(hist_df.head())

    # Instantiate matcher
    matcher = OrderMatcher(hist_df)

    # Slider per coefficiente di similitudine descrizione
    sim_threshold = st.sidebar.slider(
        "Coefficiente di similitudine descrizione",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Modifica questa soglia per affinare il matching sulla descrizione",
    )
    # Slightly lower threshold for customer-specific matching
    cust_threshold = sim_threshold * 0.8

    # Collect and parse all new orders from uploaded files and manual text
    parsed_orders: List[pd.DataFrame] = []

    # Parse uploaded files if any
    if new_files:
        for uploaded in new_files:
            try:
                suffix = Path(uploaded.name).suffix.lower()
                df: Optional[pd.DataFrame] = None
                if suffix in [".xls", ".xlsx", ".csv"]:
                    df = parse_excel(uploaded)
                elif suffix == ".pdf":
                    # First try the default PDF parser.  Some PDFs may not have
                    # the English headers ("Item", "Qty"), so the default
                    # parser can return an empty DataFrame or raise an error.
                    df = None
                    try:
                        df = parse_pdf(uploaded)
                    except Exception:
                        # Ignore errors from the standard parser and fall back
                        df = None
                    # If no rows were extracted (empty or None), fall back to
                    # the flexible parser that recognises Italian headers (e.g.
                    # "Articolo", "Qta", "Fornitore").  This ensures PDFs
                    # like the Optima order confirmations are parsed correctly.
                    if df is None or df.empty:
                        df = parse_pdf_flexible(uploaded)
                elif suffix in [".txt", ".text"]:
                    df = parse_text(uploaded)
                else:
                    st.warning(f"Formato non supportato per {uploaded.name}")
                if df is not None and not df.empty:
                    # If missing customer_code, inject default
                    if "customer_code" not in df.columns or df["customer_code"].isna().all():
                        if default_customer_code:
                            df["customer_code"] = default_customer_code
                        else:
                            st.warning(
                                f"Il file {uploaded.name} non contiene il codice cliente e non è stato specificato un valore predefinito."
                            )
                    parsed_orders.append(df)
            except Exception as e:
                st.warning(f"Errore nel parsing di {uploaded.name}: {e}")

    # Parse manual text input if provided
    if manual_text and manual_text.strip():
        try:
            text_df = parse_text(manual_text)
            if text_df is not None and not text_df.empty:
                # Inject default customer code if missing
                if "customer_code" not in text_df.columns or text_df["customer_code"].isna().all():
                    if default_customer_code:
                        text_df["customer_code"] = default_customer_code
                    else:
                        st.warning(
                            "L'ordine testuale non contiene il codice cliente e non è stato specificato un valore predefinito."
                        )
                parsed_orders.append(text_df)
        except Exception as e:
            st.warning(f"Errore nel parsing del testo inserito manualmente: {e}")

    if not parsed_orders:
        st.info("Nessun nuovo ordine valido da analizzare.")
        return

    # Concatenate all parsed orders
    new_df = pd.concat(parsed_orders, ignore_index=True)

    # Display new orders
    st.subheader("Nuovi ordini caricati")
    st.dataframe(new_df)

    # Match against history
    result_df = matcher.match(
        new_df,
        cust_desc_threshold=cust_threshold,
        global_desc_threshold=sim_threshold,
    )

    # Display results
    st.subheader("Risultati dell'analisi")
    st.dataframe(result_df)

    # Offer the user a chance to flag incorrect matches.
    # For each row in the results, present a checkbox labelled with the
    # description and the suggested code.  A True value indicates that the
    # user considers the suggested mapping to be wrong.  The selections are
    # stored in a new column ``user_flagged`` on ``result_df``.
    st.markdown("**Spunta le righe con articoli associati erroneamente:**")
    user_flags: List[bool] = []
    for idx, row in result_df.iterrows():
        label = f"{row['item_description']} (codice suggerito: {row['item_code']})"
        # Use a unique key per row to preserve checkbox state across reruns
        is_wrong = st.checkbox(label, key=f"user_flag_{idx}")
        user_flags.append(is_wrong)
    # Append the user flags to the DataFrame.  This creates a new boolean column
    # ``user_flagged`` indicating which rows were manually marked as incorrect.
    result_df["user_flagged"] = user_flags

    # Show flagged rows
    # Combine system flags (non‑empty ``flags`` column) with user feedback.
    flagged = result_df[(result_df["flags"] != "") | (result_df["user_flagged"] == True)]
    if not flagged.empty:
        st.subheader("Righe con avvisi o segnalate dall'utente")
        st.dataframe(flagged)

        # Offer the user a choice to correct each flagged row.  For usability,
        # propose only codes whose descriptions are similar to the flagged
        # description, rather than all codes purchased by the customer.
        st.markdown("**Seleziona il codice corretto per le righe segnalate:**")
        # Iterate over flagged rows and build a selectbox for each one.
        for idx, row in flagged.iterrows():
            cust = str(row['customer_code'])
            desc = str(row['item_description'])
            # Gather candidate codes from the customer's purchase history
            candidate_codes = list(matcher.customer_codes.get(cust, []))
            options: List[Tuple[float, str, str]] = []
            # Compute similarity for each candidate; use the matcher's private
            # method to score similarity.
            for code in candidate_codes:
                canon_desc = matcher.code_to_desc.get(code, "")
                sim = matcher._score_similarity(desc.lower(), canon_desc.lower())
                options.append((sim, code, canon_desc))
            # Filter out candidates with very low similarity and sort descending
            options = [t for t in options if t[0] > 0.1]
            options_sorted = sorted(options, key=lambda x: -x[0])[:5] if options else []
            # If no candidate meets the threshold, fall back to top 5 by similarity across all codes
            if not options_sorted:
                # Compute similarity across all codes in the history
                all_options: List[Tuple[float, str, str]] = []
                for code2, canon_desc2 in matcher.code_to_desc.items():
                    sim2 = matcher._score_similarity(desc.lower(), canon_desc2.lower())
                    all_options.append((sim2, code2, canon_desc2))
                options_sorted = sorted(all_options, key=lambda x: -x[0])[:5]
            # Build display labels for the selectbox
            select_options = [f"{code} – {canon_desc}" for _, code, canon_desc in options_sorted]
            # Default index 0 if available
            default_index = 0
            chosen_label = st.selectbox(
                f"Codice corretto per '{desc}'", select_options, index=default_index, key=f"correct_{idx}"
            )
            # Extract the selected code (string before the dash)
            selected_code = chosen_label.split(" – ")[0] if " – " in chosen_label else chosen_label
            # Update the result data frame with the user‑selected code
            result_df.at[idx, 'item_code'] = selected_code
            # Mark as description mapped (manual correction)
            result_df.at[idx, 'desc_mapped'] = True
            # Append a note in the flags column indicating manual correction
            note = result_df.at[idx, 'flags']
            if note:
                note = f"{note}; user corrected"
            else:
                note = "user corrected"
            result_df.at[idx, 'flags'] = note
    else:
        st.success("Nessuna anomalia rilevata nei nuovi ordini.")

    # Offer download
    buffer = BytesIO()
    import tempfile  # use tempfile to determine a writable temporary directory
    temp_dir = tempfile.gettempdir()
    temp_path = str(Path(temp_dir) / "sap_export.xlsx")
    export_path = export_to_sap(result_df, path=temp_path)
    # Read the file into the BytesIO buffer
    with open(export_path, "rb") as f:
        buffer.write(f.read())
    # Present a download button with the binary content
    st.download_button(
        label="Scarica file per SAP",
        data=buffer.getvalue(),
        file_name="sap_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()