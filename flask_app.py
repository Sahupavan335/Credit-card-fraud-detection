import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from io import BytesIO

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages

# --- Load saved models and scalers ---
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# Store results globally (for downloads)
results_cache = {}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_name = request.form.get("model")
        chunk_size = int(request.form.get("chunk_size"))
        file = request.files.get("file")

        if not file:
            flash("⚠️ Please upload a CSV or Excel file.", "warning")
            return redirect(url_for("index"))

        file_type = file.filename.split(".")[-1]
        model = models[model_name]

        results_list = []

        if file_type == "csv":
            file.stream.seek(0)
            df_iter = pd.read_csv(file, chunksize=chunk_size)
        else:
            df = pd.read_excel(file)
            df_iter = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        for chunk in df_iter:
            df_chunk = chunk.copy()

            # Apply scalers
            for col, scaler in scalers.items():
                if col in df_chunk.columns:
                    df_chunk[[col]] = scaler.transform(df_chunk[[col]])

            predictions = model.predict(df_chunk)
            pred_probs = model.predict_proba(df_chunk)

            df_chunk["Predicted_Class"] = predictions
            df_chunk["Probability_Legitimate"] = pred_probs[:, 0]
            df_chunk["Probability_Fraudulent"] = pred_probs[:, 1]

            results_list.append(df_chunk)

        df_results = pd.concat(results_list, ignore_index=True)

        # Save in cache for downloads
        results_cache["results"] = df_results

        total_fraud = (df_results["Predicted_Class"] == 1).sum()
        total_legit = (df_results["Predicted_Class"] == 0).sum()

        return render_template(
            "results.html",
            preview=df_results.head(100).to_html(classes="table table-striped table-hover", index=False),
            total_fraud=total_fraud,
            total_legit=total_legit
        )

    return render_template("index.html", models=list(models.keys()))


@app.route("/download/<filetype>")
def download(filetype):
    df_results = results_cache.get("results")
    if df_results is None:
        flash("⚠️ No results to download. Please process a file first.", "warning")
        return redirect(url_for("index"))

    if filetype == "csv":
        output = BytesIO()
        df_results.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="predictions.csv", mimetype="text/csv")

    elif filetype == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_results.to_excel(writer, index=False, sheet_name="Predictions")
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name="predictions.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        flash("❌ Invalid download type.", "danger")
        return redirect(url_for("index"))
    
# --- Error handler for large files ---
@app.errorhandler(413)
def request_entity_too_large(error):
    flash("🚨 File too large! Please upload a file smaller than 500 MB.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
