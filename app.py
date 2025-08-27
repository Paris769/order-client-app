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
from typing import List, Optional

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore

from parsers import parse_excel, parse_pdf, parse_text
from matcher import OrderMatcher
from sap_exporter import export_to_sap


def main() -> None:
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
        "Codice cliente predefinito per file senza colonna cliente", value=""
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
        st.warning("Lo storico degli ordini sembra vuoto o non contiene le colonne attese.")
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
        help="Modifica questa soglia per affinare il matching sulla descrizione"
    )
    cust_threshold = sim_threshold * 0.8

    # Collect and parse all new orders from uploaded files and manual text
    parsed_orders: List[pd.DataFrame] = []
    # Parse uploaded files if any
    if new_files:
        for uploaded in new_files:
            try:
                # Use pathlib to determine the file suffix instead of os.path, to avoid
                # relying on the os module which can be sandboxed on some platforms.
                suffix = Path(uploaded.name).suffix.lower()
                df: Optional[pd.DataFrame] = None
                if suffix in [".xls", ".xlsx", ".csv"]:
                    df = parse_excel(uploaded)
                elif suffix == ".pdf":
                    df = parse_pdf(uploaded)
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
  #.match(, cust_desc_threshold=cust_threshold, global_desc_threshold=sim_thresholdnew_df)
    st.subheader("Risultati dell'analisi")
      result_df = matcher.match(new_df, cust_desc_threshold=cust_threshold, global_desc_threshold=sim_threshold)
    # Show flagged rows
    flagged = result_df[result_df["flags"] != ""]
    if not flagged.empty:
        st.subheader("Righe con avvisi")
        st.dataframe(flagged)
    else:
        st.success("Nessuna anomalia rilevata nei nuovi ordini.")

    # Offer download
    # Export the SAP-ready DataFrame to a temporary file in a writable
    # directory. Writing to /home/oai/share is not allowed on Streamlit Cloud.
    buffer = BytesIO()
    import tempfile  # use tempfile to determine a writable temporary directory
    # Determine a temporary path for the export file.
    temp_dir = tempfile.gettempdir()
    # Build the temporary file path using pathlib instead of os.path.join
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
