import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
# Allow your frontend friend's Vercel/Cloudflare domains to fetch this API.
# Swap "*" for the real domain(s) once you have them, e.g.
# CORS(app, resources={r"/*": {"origins": ["https://your-app.vercel.app"]}})
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqi_model.keras")
FEATURE_SCALER_PATH = os.path.join(os.path.dirname(__file__), "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(os.path.dirname(__file__), "target_scaler.pkl")

LOOKBACK = 48
FORECAST = 24
TARGET_INDEX = 1  # index of 'Target_Pollutant' in the 13-column feature order below

# Must match the exact column order used when training (df.columns after cleaning)
FEATURE_ORDER = [
    "CO(GT)", "Target_Pollutant", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH",
]

print("Loading model and scalers...")
model = load_model(MODEL_PATH)
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)
print("Model and scalers loaded.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
      "sequence": [
        {"CO(GT)": 2.1, "Target_Pollutant": 1150, "NMHC(GT)": 120, "C6H6(GT)": 9.2,
         "PT08.S2(NMHC)": 950, "NOx(GT)": 180, "PT08.S3(NOx)": 900, "NO2(GT)": 110,
         "PT08.S4(NO2)": 1500, "PT08.S5(O3)": 1000, "T": 18.2, "RH": 45.0, "AH": 0.9},
        ... exactly 48 of these objects, oldest hour first, most recent hour last ...
      ]
    }

    Returns:
    {
      "predictions": [1180.4, 1172.1, ..., 24 values],
      "unit": "Target_Pollutant scale (same units as training data)"
    }
    """
    data = request.get_json(force=True)
    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request body"}), 400

    sequence = data["sequence"]
    if len(sequence) != LOOKBACK:
        return jsonify({
            "error": f"'sequence' must contain exactly {LOOKBACK} hourly records, got {len(sequence)}"
        }), 400

    try:
        raw = np.array([[row[col] for col in FEATURE_ORDER] for row in sequence])
    except KeyError as e:
        return jsonify({"error": f"Missing feature in one of the records: {e}"}), 400

    scaled = feature_scaler.transform(raw)
    model_input = scaled.reshape(1, LOOKBACK, len(FEATURE_ORDER))

    pred_scaled = model.predict(model_input)[0]  # shape (24,)

    pred_real = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    return jsonify({
        "predictions": pred_real.tolist(),
        "unit": "Target_Pollutant scale (same units as training data)"
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)