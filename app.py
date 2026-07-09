import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# --- Page Config ---
st.set_page_config(page_title="AeroGuard AQI", page_icon="🌍", layout="centered")

# --- Header ---
st.title("🌍 AeroGuard: Smart City AQI Predictor")
st.write("Adjust the environmental factors below to predict the Air Quality Index using your original Keras model.")

# --- Load Assets safely ---
@st.cache_resource
def load_assets():
    # Load the native Keras model (compile=False skips training setup)
    model = tf.keras.models.load_model("aeroguard_lstm_engine.keras", compile=False)
    scaler = joblib.load("aqi_pipeline_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading Keras model: {e}")
    st.stop()

# --- UI Sliders ---
st.markdown("### 📊 Environmental Parameters")
traffic = st.slider("🚗 Traffic Density", 10.0, 100.0, 50.0)
factory = st.slider("🏭 Factory Emissions", 20.0, 150.0, 60.0)
wind = st.slider("💨 Wind Speed", 2.0, 35.0, 10.0)
temp = st.slider("🌡️ Temperature (°C)", 15.0, 45.0, 25.0)

# --- Prediction Logic ---
if st.button("🔮 Predict AQI", use_container_width=True):
    with st.spinner("Running TensorFlow Inference..."):
        # 1. Format and scale features
        features = np.array([[traffic, factory, wind, temp]])
        scaled_features = scaler.transform(features)
        
        # 2. Reshape for LSTM: (batch_size, sequence_length, features)
        input_data = np.reshape(scaled_features, (1, 1, 4)).astype(np.float32)
        
        # 3. Run prediction
        raw_prediction = model.predict(input_data)
        prediction = float(raw_prediction[0][0])
        
        # --- Display Results ---
        st.markdown("---")
        st.subheader("Predicted Air Quality Index")
        
        if prediction <= 50:
            st.success(f"## 🟢 {prediction:.1f} (Good)")
        elif prediction <= 100:
            st.warning(f"## 🟡 {prediction:.1f} (Moderate)")
        else:
            st.error(f"## 🔴 {prediction:.1f} (Poor/Hazardous)")