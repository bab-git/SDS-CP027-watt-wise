# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("🔋 Watt Wise: SARIMA Energy Forecasting")

@st.cache_data
def load_data():
    df = pd.read_csv("Energy_consumption.csv", parse_dates=['Timestamp'], index_col='Timestamp')
    df = df.asfreq('H')  # ensure hourly frequency
    df = df.ffill()
    return df

df = load_data()
st.write("### Raw Data", df.tail())

horizon = st.selectbox("Select forecast window", ["Next 24 Hours", "Next 7 Days"])
steps = 24 if horizon == "Next 24 Hours" else 168

# Fit SARIMA model
st.write("Training SARIMA model...")
model = SARIMAX(df['EnergyConsumption'], order=(1,1,1), seasonal_order=(1,1,1,24))
results = model.fit(disp=False)

# Forecast
forecast = results.forecast(steps=steps)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=steps, freq='H')
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot
st.write("### Forecast")
fig, ax = plt.subplots(figsize=(12, 5))
df['EnergyConsumption'].iloc[-7*24:].plot(ax=ax, label='Historical')
forecast_series.plot(ax=ax, label='Forecast', linestyle='--', color='orange')
plt.legend()
plt.title(f"Energy Forecast - {horizon}")
st.pyplot(fig)