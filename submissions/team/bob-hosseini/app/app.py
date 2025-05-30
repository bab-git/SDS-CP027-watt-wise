import streamlit as st
import json
import pickle
import os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys
import os
import pandas as pd

# Add the parent directory to the Python path to import from src/
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)
from utils import seasonal_decompose_stl, evaluate_model

# This must be FIRST Streamlit command
st.set_page_config(
    page_title="WattWise",
    # layout="wide",
    initial_sidebar_state="auto"
)

BASE_DIR = os.path.dirname(__file__)  # wherever app.py lives

# -----------------------
# Load cleaned data
# -----------------------
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data_cleaned.pkl")

@st.cache_data
def load_data(path):
    with open(path, 'rb') as f:
        data_cleaned = pickle.load(f)
    df_data = data_cleaned['df_data']
    df_data_history = df_data.iloc[-(48+24):-48]
    df_24h_complete = data_cleaned['df_data_24h_complete']
    return df_data, df_24h_complete, df_data_history
df_data, df_24h_complete, df_data_history = load_data(DATA_PATH)

# -----------------------
# Load modeling data splits
# -----------------------
DATA_SPLIT_PATH = os.path.join(BASE_DIR, "..", "data", "data_split.pkl")
@st.cache_data
def load_model_data(path):
    with open(path, 'rb') as f:
        data_splits = pickle.load(f)
        # unpack the data splits
    train_hr = data_splits['train_hr']
    exog_train = data_splits['exog_train']
    exog_test = data_splits['exog_test']    
    return train_hr, exog_train, exog_test

train_hr, exog_train, exog_test = load_model_data(DATA_SPLIT_PATH)

# load model checkpoint
MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "models", "sarimax_checkpoint.json")
@st.cache_data
def load_model_checkpoint(path):
    with open(path, 'r') as f:
        checkpoint = json.load(f)
    # unpack the model checkpoint
    return (tuple(
        checkpoint["order"]), 
        tuple(checkpoint["seasonal_order"]), 
        checkpoint["params"]
    )
order, seasonal_order, params = load_model_checkpoint(MODEL_CHECKPOINT_PATH)

# -----------------------
# Rebuild & cache the model
# -----------------------
@st.cache_resource
def load_filtered_model(train_data, exog_data, order, seasonal_order, params):
    model = SARIMAX(train_data, exog=exog_data, order=order, seasonal_order=seasonal_order)
    return model.filter(params)

fit_model = load_filtered_model(train_hr, exog_train, order, seasonal_order, params)

# -----------------------
# Streamlit App Layout
# -----------------------
# App title
st.title("🔌 Watt Wise Forecasting App")
st.markdown(
    """
    This interactive app is part of the **Watt Wise** project, a collaborative, open-source initiative hosted by the **SuperDataScience** community.  
    The project focuses on forecasting building energy consumption using historical usage patterns and contextual factors such as weather and occupancy.  
    This app features a pre-trained **SARIMAX model** that uses simulated future weather inputs to generate short-term energy forecasts.
    The model is trained on historical data and evaluated on a test set.
    """,
    unsafe_allow_html=True
)
st.markdown(
    "[📂 View Source Code on GitHub](https://github.com/bab-git/SDS-CP027-watt-wise/tree/dev_bob/submissions/team/bob-hosseini)",
    unsafe_allow_html=True
)

# Tabs
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "📈 Forecasting Energy Consumption"])

# -----------------------
# EDA TAB
# -----------------------
with tab1:
    st.header("📊 Exploratory Data Analysis")
    st.subheader("📊 Preprocessed Feature Data")
    st.dataframe(df_data.head(100))  # display first 100 rows


    # ====================== Descriptive Statistics ======================
    st.subheader("Descriptive Statistics")
    st.dataframe(df_data.describe())

    # ====================== Rolling Mean of Hourly Energy Consumption ======================
    st.subheader("📈 Rolling Mean of Hourly Energy Consumption")
    st.caption("Use the slider to adjust the size of the rolling window in hours. This smooths short-term fluctuations and helps highlight trends.")

    window_size = st.slider("Rolling window size (hours)", 6, 72, 24)
    
    df_rolling = df_data.copy()
    df_rolling['RollingMean'] = df_rolling['EnergyConsumption'].rolling(window=window_size).mean()

    fig = px.line(
        df_rolling,
        x=df_rolling.index,
        y='RollingMean',
        title=f"{window_size}-Hour Rolling Mean of Energy Consumption",
        labels={'RollingMean': 'Energy Consumption', 'index': 'Time'},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ====================== Histogram ======================
    # Create the histogram with KDE using Plotly Express
    fig = px.histogram(
        df_data,
        x="EnergyConsumption",
        nbins=40,
        title="Distribution of Energy Consumption",
        labels={'EnergyConsumption': 'Energy Consumption'},
        marginal="violin",  # Adds a violin plot on the side for KDE
        opacity=0.7
    )

    # Update layout for better readability
    fig.update_layout(
        bargap=0.1  # Gap between bars
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    - Energy consumption is moderately concentrated, with most values falling between 71 kWh (Q1) and 82 kWh (Q3).
    """)

    # ====================== Energy Consumption by Time of Day ======================
    st.subheader("Energy Consumption by Time of Day")
    fig = px.box(df_data,
                x='Time',                # assumes 'Time' is in HH:MM or similar format
                y='EnergyConsumption',
                # title="Energy Consumption by Hour",
                labels={'EnergyConsumption': 'Energy Consumption', 'Time': 'Hour of Day'})

    fig.update_layout(xaxis_tickangle=90)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    - The Energy Consumption has almost a uniform distribution across the hours of the day.
    """)

    # ====================== Energy Consumption by Day of Week ======================
    st.subheader("Energy Consumption by Day of Week")
    # Boxplot of energy consumption by day of week
    df_data_daily = (
        df_24h_complete['EnergyConsumption']
        .resample('D')
        .sum()
        .rename('EnergyConsumption')
        .to_frame()
    )
    df_data_daily['DayOfWeek'] = df_data_daily.index.day_name()
    # st.dataframe(df_data_daily.head(100))
    fig = px.box(df_data_daily,
                x='DayOfWeek',
                y='EnergyConsumption',
                # title="Energy Consumption by Day of Week",
                labels={'EnergyConsumption': 'Energy Consumption', 'DayOfWeek': 'Day of Week'})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    - Here we observer larger variation on Sundays.
    - Saturday seems to have most peeks in consumption.
    """)

    # Correlation matrix using seaborn
    st.subheader("Correlation Matrix")
    st.caption("The correlation between the Energy Consumption and the other features.")
    num_cols = df_data.select_dtypes(include=['int64', 'float64']).columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_data[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax, center=0)
    st.pyplot(fig, use_container_width=True)
    st.markdown("""
    - The correlation matrix of numeric features show that Energy consumption has a high correlation with the Tempreture.
    - However, it's important to consider that we won't have the tempreture value for timestamps in the future!
    """)

    # ====================== STL decomposition of energy consumption ======================
    st.subheader("STL Decomposition of Energy Consumption")
    st.markdown("Select a time window to perform STL decomposition on the energy consumption series.")
    # User selects week
    selected_range = st.selectbox("Select week:", options=[
        ('2022-01-03', '2022-01-09'),
        ('2022-01-10', '2022-01-16'),
        ('2022-01-17', '2022-01-23')
    ])
        # Subset data and decompose
    start_date, end_date = selected_range
    df_window = df_data.loc[start_date:end_date]

    fig, ratio_std, ratio_range, explained_var = seasonal_decompose_stl(
        df_window, col='EnergyConsumption', period=24, figsize=(12, 7))

    st.pyplot(fig)
    st.markdown(f"""
    **Explained Variance:** {explained_var:.2%}  
    **Residual Std / Signal Std:** {ratio_std:.2%}  
    **Residual Range / Seasonal Range:** {ratio_range:.2%}
    """)

    # ====================== ACF and PACF using subplot ======================
    st.subheader("ACF and PACF")
    st.caption("The ACF and PACF of the energy consumption series.")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(df_window['EnergyConsumption'], ax=axs[0])
    plot_pacf(df_window['EnergyConsumption'], ax=axs[1])
    plt.tight_layout()
    for ax in axs:
        ax.grid(True)
    st.pyplot(fig, use_container_width=True)

    st.markdown("""
    - ACF and PACF drops quickly and then oscillates slightly around 0
    - No dominant lags stand out strongly, which might suggest no clear MA structure
    """)

# -------------------------------
# Forecasting TAB
# -------------------------------
with tab2:
    st.header("📈 Forecasting Energy Consumption")
    st.markdown(f"""
    **SARIMAX Model Configuration:**

    - **Order (p, d, q):** `{order}`
    - **Seasonal Order (P, D, Q, s):** `{seasonal_order}`
    """)
    st.markdown("Select how many hours into the future you want to forecast (up to 47).")

    horizon = st.slider("Forecast Horizon (hours)", min_value=1, max_value=47, value=24)

    # Subset the exog for the selected forecast horizon
    exog_input = exog_test.iloc[:horizon]

    # Perform forecasting
    st.markdown("Generating forecast...")
    forecast_result = fit_model.get_forecast(steps=len(exog_input), exog=exog_input)
    forecast_df = forecast_result.summary_frame()

    # Rename for consistency
    forecast_df.rename(columns={
        "mean": "Forecast",
        "mean_ci_lower": "Lower CI",
        "mean_ci_upper": "Upper CI"
    }, inplace=True)


    df_data_test = df_data.iloc[-48:-(48-horizon)]

    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot mean forecast
    ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green', marker='x', linestyle='--')

    # Plot confidence interval
    ax.fill_between(forecast_df.index,
                    forecast_df['Lower CI'],
                    forecast_df['Upper CI'],
                    color='orange', alpha=0.3, label='Confidence Interval')

    # plot the actual values
    ax.plot(df_data_test.index, df_data_test['EnergyConsumption'], label='Test Data', color='orange', marker='o')

    # plot the training data
    ax.plot(df_data_history.index, df_data_history['EnergyConsumption'], label='Training Data', color='blue', marker='o')

    ax.set_title(f"{horizon}-Hour Forecast with Confidence Interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Consumption")
    # rotate the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("""
    - The confidence interval remains nearly constant across the forecast horizon. This suggests that the SARIMA model does not capture any increasing uncertainty over time, likely because it lacks strong trend or seasonal structure in the target series.
    - Instead, the forecast relies heavily on the exogenous variables provided at prediction time, which are treated by SARIMA model as fixed and fully known, leading to narrow, stable confidence bounds regardless of forecast distance.        
    """)

    df_predictions_sarimax = pd.DataFrame(index=exog_input.index)
    df_predictions_sarimax['prediction'] = forecast_df['Forecast']
    df_predictions_sarimax['truth'] = df_data_test['EnergyConsumption'].values
    metrics =  evaluate_model(df_predictions_sarimax, verbose=True)
    st.markdown(f"""
    **Model Evaluation Metrics:**
    - **MAE:** {metrics['mae']:.2f} kWh
    - **MAPE:** {metrics['mape']:.2f}%
    - **RMSE:** {metrics['rmse']:.2f} kWh
    - **R2:** {metrics['r2']:.2f}
    """)

    # # show df_predictions_sarimax table
    # st.dataframe(df_predictions_sarimax)
    # from sklearn.metrics import r2_score
    # st.write(r2_score(df_predictions_sarimax['truth'], df_predictions_sarimax['prediction']))
