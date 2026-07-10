import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="AeroGuard 24h Forecast (v3.0)", page_icon="🌍", layout="wide")

st.title("🌍 AeroGuard: 24-Hour Smart City AQI Forecast")
st.markdown("Powered by an 88% Accuracy CNN-BiLSTM Hybrid Engine.")

if 'active_chart' not in st.session_state:
    st.session_state.active_chart = 'Area'

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("aqi_model.keras", compile=False)
    # MUST MATCH YOUR NEWEST SCALER NAME EXACTLY:
    feature_scaler = joblib.load("feature_scaler_v3.pkl") 
    target_scaler = joblib.load("target_scaler.pkl")
    return model, feature_scaler, target_scaler

try:
    model, feature_scaler, target_scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading assets: {e}")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛️ Live City Sensors")
    current_aqi = st.slider("🌫️ Current AQI", 10.0, 400.0, 100.0)
    temp = st.slider("🌡️ Temperature (°C)", -5.0, 45.0, 25.0)
    rh = st.slider("💧 Humidity (%)", 10.0, 90.0, 50.0)
    co = st.slider("🚗 CO Emissions", 0.5, 8.0, 2.5)
    nox = st.slider("🏭 NOx Emissions", 20.0, 500.0, 150.0)
    no2 = st.slider("⛽ NO2 Emissions", 20.0, 300.0, 100.0)

# --- Logic ---
try:
    base_sensors = np.array([co, current_aqi, 200.0, 10.0, 1000.0, nox, 800.0, no2, 1500.0, 1000.0, temp, rh, 1.0])
    
    current_time = datetime.now()
    historical_window = np.zeros((48, 17))
    
    for i in range(48):
        past_time = current_time - timedelta(hours=(47 - i))
        h, d = past_time.hour, past_time.weekday()
        
        hour_sin, hour_cos = np.sin(h * (np.pi / 12)), np.cos(h * (np.pi / 12))
        day_sin, day_cos = np.sin(d * (2 * np.pi / 7)), np.cos(d * (2 * np.pi / 7))
        
        full_row = np.append(base_sensors, [hour_sin, hour_cos, day_sin, day_cos])
        full_row[:13] *= (1 + (hour_sin * 0.15))
        historical_window[i, :] = full_row
    
    input_data = np.reshape(feature_scaler.transform(historical_window), (1, 48, 17)).astype(np.float32)
    
    raw_sensor_forecast = target_scaler.inverse_transform(model.predict(input_data, verbose=0)[0].reshape(-1, 1)).flatten()
    
    # Mathematical AQI Conversion & Hard Ceiling
    standardized_aqi = ((raw_sensor_forecast - 700.0) / 1300.0) * 500.0
    real_world_forecast = np.clip(standardized_aqi, a_min=0, a_max=400) # THIS GUARANTEES MAX 400

    # --- Visuals ---
    st.markdown("---")
    avg_aqi, max_aqi = np.mean(real_world_forecast), np.max(real_world_forecast)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 24-Hour Average", f"{avg_aqi:.0f}")
    col2.metric("⚠️ Expected Peak", f"{max_aqi:.0f}")
    with col3:
        if max_aqi > 300: st.error("🚨 HIGH RISK")
        elif max_aqi > 150: st.warning("⚠️ MODERATE RISK")
        else: st.success("✅ LOW RISK")

    timestamps = [current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i) for i in range(1, 25)]
    chart_data = pd.DataFrame({"Time": timestamps, "AQI": real_world_forecast})
    
    btn1, btn2, btn3, btn4, btn5 = st.columns(5)
    if btn1.button("🌊 Area", use_container_width=True): st.session_state.active_chart = 'Area'
    if btn2.button("📊 Bar", use_container_width=True): st.session_state.active_chart = 'Bar'
    if btn3.button("🕸️ Radar", use_container_width=True): st.session_state.active_chart = 'Radar'
    if btn4.button("⏱️ Gauge", use_container_width=True): st.session_state.active_chart = 'Gauge'
    if btn5.button("📅 Heatmap", use_container_width=True): st.session_state.active_chart = 'Heatmap'

    if st.session_state.active_chart == 'Area':
        fig = px.area(chart_data, x="Time", y="AQI", markers=True)
        fig.update_layout(yaxis_range=[0, 420])
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.active_chart == 'Bar':
        colors = ['#ef4444' if v > 300 else '#f59e0b' if v > 150 else '#10b981' for v in real_world_forecast]
        fig = go.Figure(data=[go.Bar(x=chart_data['Time'], y=chart_data['AQI'], marker_color=colors)])
        fig.update_layout(yaxis_range=[0, 420])
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.active_chart == 'Radar':
        fig = go.Figure(data=go.Scatterpolar(r=[co/8, nox/500, no2/300, (temp+5)/50, rh/100], theta=['CO', 'NOx', 'NO2', 'Temp', 'Hum'], fill='toself'))
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.active_chart == 'Gauge':
        fig = go.Figure(go.Indicator(mode="gauge+number", value=real_world_forecast[0], gauge={'axis': {'range': [None, 400]}}))
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.active_chart == 'Heatmap':
        fig = px.imshow(real_world_forecast.reshape(4, 6), color_continuous_scale="RdYlGn_r", zmin=0, zmax=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📂 View Raw Data"):
        st.dataframe(chart_data.set_index("Time"), use_container_width=True) # REMOVED MATPLOTLIB REQUIREMENT

except Exception as e:
    st.error(f"Prediction Error: {e}")
