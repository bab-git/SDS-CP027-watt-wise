import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('submissions/team/Patrick-Edosoma/Sarimax_model.pkl')

st.markdown("### Welcome to the Energy Consumption Forecast App")
st.sidebar.title("About the App")
st.sidebar.info(
    "This app forecasts energy consumption based on exogenous variables using a SARIMAX model incorporating temperature and occupancy as exogenous variables. Users can upload their own exogenous data, and the app generates precise consumption forecasts over the specified period. Leveraging time series analysis and external factors, the model captures seasonal patterns and trends to provide reliable energy consumption predictions, supporting better planning and decision-making."
)

# Load the sample CSV from local path
sample_path = "submissions/team/Patrick-Edosoma/forecast_data.csv"
sample_data = pd.read_csv(sample_path)
st.image("submissions/team/Patrick-Edosoma/POWER.png", caption="Power Demand Prediction")

# Convert to CSV bytes for download
csv = sample_data.to_csv(index=False).encode('utf-8')
st.sidebar.title("Features")
st.sidebar.write("""


  - **Temperature:**  
    Ambient temperature readings, which affect heating/cooling energy demands.

  - **Occupancy:**  
    Number of occupants present, directly impacting energy usage through lighting, appliance use, and HVAC load.

These features are critical drivers of energy consumption in buildings. For instance, high occupancy during hot weather typically increases air conditioning use.
""")


# Sidebar section
with st.sidebar:
    st.header("📊 Sample Exogenous Data")
    st.dataframe(sample_data, use_container_width=True)

    st.download_button(
        label="⬇️ Download Sample CSV",
        data=csv,
        file_name="sample_exogenous_data.csv",
        mime="text/csv"
    )

# Upload exogenous variables CSV
uploaded_file = st.file_uploader("Upload Exogenous Variables (CSV)", type="csv")

if uploaded_file is not None:
    X_future = pd.read_csv(uploaded_file)

    # Forecast for the number of rows in exogenous data
    forecast = model.predict(start=len(model.data.endog),
                             end=len(model.data.endog) + len(X_future) - 1,
                             exog=X_future)

    # Convert to DataFrame for display
    forecast_df = pd.DataFrame({
        'Forecasted Energy Consumption': forecast
    })

    st.subheader("Forecasted Energy Consumption (Table View):")
    st.dataframe(forecast_df)

    st.subheader("Forecasted Energy Consumption (Line Chart):")
    st.line_chart(forecast_df)
