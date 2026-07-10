import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="AeroGuard 24h Forecast", page_icon="🌍", layout="wide")

st.title("🌍 AeroGuard: 24-Hour Smart City AQI Forecast")
st.markdown("Powered by BiLSTM Deep Learning and real-time UCI sensor data.")

# --- Load Assets Safely ---
# --- Load Assets Safely ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("aqi_model.keras", compile=False)
    # CHANGE THIS LINE TO MATCH THE NEW FILE:
    feature_scaler = joblib.load("feature_scaler_v2.pkl")
    target_scaler = joblib.load("target_scaler.pkl")
    return model, feature_scaler, target_scaler

try:
    model, feature_scaler, target_scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading assets. Make sure .keras and .pkl files are uploaded! Error: {e}")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛️ Live City Sensors")
    st.write("Adjust current conditions to forecast the next 24 hours.")
    
    # The most understandable features from the 13-column UCI dataset
    current_aqi = st.slider("🌫️ Current Target Pollutant", 500.0, 2000.0, 1000.0)
    temp = st.slider("🌡️ Temperature (°C)", -5.0, 45.0, 25.0)
    rh = st.slider("💧 Relative Humidity (%)", 10.0, 90.0, 50.0)
    co = st.slider("🚗 CO (Carbon Monoxide)", 0.5, 8.0, 2.5)
    nox = st.slider("🏭 NOx Emissions", 20.0, 500.0, 150.0)
    no2 = st.slider("⛽ NO2 Emissions", 20.0, 300.0, 100.0)

# --- Real-Time Prediction Logic ---
try:
    # 1. Reconstruct the 13-feature array exactly as it appeared in your notebook's DataFrame
    # Column Order: ['CO(GT)', 'Target_Pollutant', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
    #                'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    # We use median default values for the sensors not exposed in the UI
    feature_row = np.array([[
        co, current_aqi, 200.0, 10.0, 1000.0, 
        nox, 800.0, no2, 1500.0, 1000.0, 
        temp, rh, 1.0
    ]])
    
    # 2. Scale the 13 features using your feature_scaler
    scaled_row = feature_scaler.transform(feature_row)
    
    # 3. Create a 48-hour historical window (Shape: [1, 48, 13])
    # For a real-time dashboard without 48 hours of live DB data, we simulate the 
    # recent baseline by repeating the current condition across the 48-hour lookback window.
    input_sequence = np.repeat(scaled_row, 48, axis=0)
    input_data = np.reshape(input_sequence, (1, 48, 13)).astype(np.float32)
    
    # 4. Predict the next 24 hours (Shape: [1, 24])
    raw_prediction = model.predict(input_data, verbose=0)
    
    # 5. Inverse Transform using your dedicated target_scaler
    # The target scaler was fit on a 2D column, so we reshape the 24 predictions to (24, 1), 
    # inverse transform them, and flatten back to a 1D list.
    preds_24h = raw_prediction[0].reshape(-1, 1)
    real_world_forecast = target_scaler.inverse_transform(preds_24h).flatten()

    # --- Dashboard Visuals ---
    st.markdown("---")
    
    # Metrics Header
    avg_aqi = np.mean(real_world_forecast)
    max_aqi = np.max(real_world_forecast)
    col1, col2 = st.columns(2)
    col1.metric("24-Hour Average Prediction", f"{avg_aqi:.0f}")
    col2.metric("Expected Peak (Worst Air Quality)", f"{max_aqi:.0f}")

    # Generate Timestamps for the X-Axis
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [current_time + timedelta(hours=i) for i in range(1, 25)]
    
    # Create DataFrame for Plotly
    chart_data = pd.DataFrame({
        "Time": timestamps,
        "Predicted Pollutant Level": real_world_forecast
    })
    
    # Interactive Plotly Chart
    st.subheader("📈 24-Hour Forecasting Trend")
    fig = px.area(chart_data, x="Time", y="Predicted Pollutant Level", 
                  markers=True, color_discrete_sequence=["#ff4b4b"])
    
    # Add a visual baseline/danger threshold (adjust 1200 based on your target's danger level)
    fig.add_hline(y=1200, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
    fig.add_hline(y=1500, line_dash="dash", line_color="red", annotation_text="High Risk")
    
    fig.update_layout(hovermode="x unified", height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw Data Table
    with st.expander("📂 View Raw 24-Hour Forecast Data"):
        st.dataframe(chart_data.set_index("Time").style.background_gradient(cmap='Reds'), use_container_width=True)

except Exception as e:
    st.error(f"Prediction Error: {e}")
