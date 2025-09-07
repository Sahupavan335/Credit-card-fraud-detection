import streamlit as st
import pandas as pd
import pickle
import numpy as np
from io import BytesIO

# --- Load saved models and scalers ---
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# --- Page Config ---
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# --- Header Section ---
st.title("💳 Credit Card Fraud Detection App")
st.markdown(
    """
    This application allows you to upload a large CSV/Excel file (up to ~400MB) and
    detect fraudulent transactions using different machine learning models.
    """
)

st.markdown("---")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Select Prediction Model", list(models.keys()))
model = models[model_name]
chunk_size = st.sidebar.slider("Chunk Size for Processing", 20000, 100000, 50000, 10000)
st.sidebar.info("Increase chunk size if you have more memory available.")

# --- File Upload ---
st.subheader("📂 Upload Your File")
uploaded_file = st.file_uploader("Upload Transaction Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    st.info("⏳ Processing file in chunks. This may take a while...")
    progress_bar = st.progress(0)

    results_list = []

    if file_type == "csv":
        # Count rows in CSV
        uploaded_file.seek(0)
        total_rows = sum(1 for _ in uploaded_file) - 1  # excluding header
        uploaded_file.seek(0)  # reset pointer

        reader = pd.read_csv(uploaded_file, chunksize=chunk_size)
    else:
        df = pd.read_excel(uploaded_file)
        total_rows = len(df)
        reader = [df[i:i+chunk_size] for i in range(0, total_rows, chunk_size)]

    processed_rows = 0

    for chunk_idx, chunk in enumerate(reader):
        df_chunk = chunk.copy()

        # Apply scalers safely
        for col, scaler in scalers.items():
            if col in df_chunk.columns:
                df_chunk[[col]] = scaler.transform(df_chunk[[col]])

        # Predictions
        predictions = model.predict(df_chunk)
        pred_probs = model.predict_proba(df_chunk)

        df_chunk["Predicted_Class"] = predictions
        df_chunk["Probability_Legitimate"] = pred_probs[:, 0]
        df_chunk["Probability_Fraudulent"] = pred_probs[:, 1]

        results_list.append(df_chunk)

        # Update progress
        processed_rows += len(chunk)
        progress = int(processed_rows / total_rows * 100)
        progress_bar.progress(min(progress, 100))

    # Combine all chunks
    df_results = pd.concat(results_list, ignore_index=True)

    st.success("✅ File processed successfully!")

    # --- Show Results Preview ---
    st.subheader("📊 Predictions Preview")
    st.dataframe(df_results.head(500), use_container_width=True)

    # --- Download Section ---
    csv = df_results.to_csv(index=False)
    st.download_button(
        "⬇️ Download Full Predictions as CSV",
        data=csv,
        file_name="predictions_large_file.csv",
        mime="text/csv",
    )

    # Excel download
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Predictions")
    excel_data = output.getvalue()

    st.download_button(
        "⬇️ Download Full Predictions as Excel",
        data=excel_data,
        file_name="predictions_large_file.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # --- Summary ---
    st.markdown("---")
    st.subheader("📈 Fraud Detection Summary")
    total_fraud = (df_results["Predicted_Class"] == 1).sum()
    total_legit = (df_results["Predicted_Class"] == 0).sum()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fraudulent Transactions Detected", total_fraud)
    with col2:
        st.metric("Legitimate Transactions Detected", total_legit)

else:
    st.warning("⚠️ Please upload a CSV or Excel file to begin.")
