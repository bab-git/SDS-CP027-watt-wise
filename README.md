# Welcome to the SuperDataScience Community Project!
Welcome to the **Watt Wise: Intelligent Time Series Energy Consumption Forecasting** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:

## Project Overview
**Watt Wise** is a time series forecasting project aimed at predicting and analyzing building energy usage using data on temperature, humidity, occupancy, and operational factors such as HVAC and lighting usage. By leveraging temporal modeling techniques, this project will address seasonality, trends, and other time-dependent patterns in energy consumption.

Dataset: https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction

## Project Objectives
- Timeseries analysis of historical data and analysis of trends
- Develop advanced timeseries models which fit the current data
- Deploy the model to production via a cloud service

## Workflow

### **Phase 1: Setup (1 Week)**
- Setup GitHub repo and project folders
- Setup virtual environment and respective libraries

### **Phase 2: EDA (1 Week)**
- **Data Integrity Check**
    - Load dataset; ensure timestamps are properly formatted and consistent.
    - Identify any missing or duplicated timestamps.

- **Time Series Profiling**
    - Assess data continuity and frequency of timestamps.
    - Explore seasonality, trends, and anomalies in the energy consumption over time.

- **Feature Exploration & Engineering**
    - Investigate temperature, humidity, occupancy, HVAC, lighting usage, and renewable energy over time.
    - Create time-lagged features (e.g., last-hour consumption), rolling means, or day-of-week indicators to capture temporal patterns.

- **Time Series Visualization**
    - Plot consumption trends (daily, weekly, monthly).
    - Identify peaks or dips in consumption relative to HVAC, lighting, holiday, and occupancy changes.

- **Data Preprocessing & Feature Engineering**
    - Fill or interpolate missing time periods where appropriate.
    - Create lagged features, rolling statistics, day-of-week or holiday encodings.
    - Examine potential transformations (log or Box-Cox) if needed to stabilize variance.

### **Phase 3: Model Development (2 Weeks)**
1. **Baseline & Classical Models**
    - Implement naive or moving average baseline forecasts.
    - Build ARIMA/SARIMA or similar statistical models, incorporating exogenous variables (HVAC usage, occupancy, etc.).

2. **Machine Learning/Deep Learning Approaches** (optional)
    - Evaluate hybrid methods using scikit-learn or XGBoost with time-based splitting.
    - Test LSTM/GRU neural networks if you have enough data and want advanced modeling.

3. **Hyperparameter Tuning & Evaluation**
    - Leverage time series cross-validation.
    - Compare performance via RMSE, MAE, MAPE, and R².
    - Select the best-performing model based on accuracy, forecast stability, and interpretability.

### **Phase 4: Deployment (1 Week)**
- **Streamlit App Creation**
    - **Forecast Feature**: Users can select a timeframe to predict (next 24 hours, next 7 days, etc.).
    - **Visualization Dashboards**: Provide interactive charts of historical trends and forecast intervals.

- **Hosting & Documentation**
    - Deploy the app on Streamlit Community Cloud or equivalent.
    - Include README or user guide in the GitHub repository for easy onboarding.

## Timeline

| Phase                          | Task                                                                     | Duration    |
| ------------------------------ | ------------------------------------------------------------------------ | ----------- |
| **Phase 1: Setup**             | Repository setup, environment installation, data integrity check         | Week 1      |
| **Phase 2: Time Series EDA**   | EDA, feature engineering, examining correlations and time-based patterns | Week 2      |
| **Phase 3: Model Development** | Baseline & ARIMA-class models, machine learning / deep learning, tuning  | Weeks 3 & 4 |
| **Phase 4: Deployment**        | Streamlit app build, hosting, documentation                              | Week 5      |
