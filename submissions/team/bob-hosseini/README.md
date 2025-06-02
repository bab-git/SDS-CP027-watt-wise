# 🧠 Watt Wise Submission – SuperDataScience Collaborative Project
[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?logo=streamlit)](https://watt-wise-bob-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
<!-- ![GitHub stars](https://img.shields.io/github/stars/bab-git/SDS-CP027-watt-wise?style=social) -->
<!-- ![GitHub forks](https://img.shields.io/github/forks/bab-git/SDS-CP027-watt-wise?style=social) -->

Welcome! This folder contains my individual contribution to the **Watt Wise: Intelligent Time Series Energy Consumption Forecasting** project, a collaborative, open-source initiative hosted by the SuperDataScience community.

---

## 📌 Project Overview

**Watt Wise** is a community-driven time series forecasting project focused on analyzing and predicting building energy usage based on historical consumption patterns and contextual factors like weather, occupancy, HVAC, and lighting.

🔗 Main Project Repository: [Watt Wise on GitHub](https://github.com/SuperDataScience-Community-Projects/SDS-CP027-watt-wise)  
🗓️ Timeline: 5-week collaborative sprint  
🧠 Hosted by: SuperDataScience Community  

---
    
## 📈 Dataset Information

This project uses a **synthetic dataset** designed for educational and experimental purposes. While it includes realistic patterns and contextual variables, it may not fully reflect real-world building energy usage behavior or distributional properties.

* 📁 **Dataset Source**: [Energy Consumption Prediction on Kaggle](https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction)
* 📌 **License**: Publicly shared for non-commercial use (verify on Kaggle)
* ⚠️ **Disclaimer**: Forecasting results and feature influences observed in this project may not generalize to operational energy management systems due to the synthetic nature of the data.

### 🧾 Feature Overview

| Feature Name           | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `Temperature`          | Hourly ambient temperature                       |
| `Humidity`             | Hourly humidity percentage                       |
| `SquareFootage`        | Size of the building (static contextual feature) |
| `Occupancy`            | Simulated number of occupants                    |
| `HVACUsage`            | Binary flag indicating HVAC system usage         |
| `LightingUsage`        | Binary flag indicating lighting system activity  |
| `RenewableEnergy`      | Amount of energy produced by on-site renewables  |
| `DayOfWeek`, `Holiday` | Categorical calendar context                     |
| `EnergyConsumption`    | Target variable: hourly energy usage (kWh)       |

Additional time-based and lagged features were derived during preprocessing for modeling purposes (e.g., `Temperature_lag1`, `HVACUsage_rol`, etc.).

---


## 👤 My Role & Contribution

As a participant in this collaborative project, I contributed to the following areas:

---

### ✅ **Exploratory Data Analysis (EDA)**

* Parsed and processed timestamp data for both hourly and daily resolutions.
* Identified non-uniform energy usage patterns, including higher weekend consumption.
* Conducted early correlation analysis, which initially showed strong contemporaneous correlation between temperature and energy consumption.

---

### ✅ **Stationarity & Time Series Profiling**

* Performed ADF tests confirming stationarity in both hourly and daily series.
* Examined autocorrelation and volatility characteristics of the target.

---

### ✅ **Outlier Detection & Revised Correlation Analysis**

* Detected and removed anomalies in energy consumption.
* Re-evaluated feature correlations using **lagged features** to avoid leakage.
* Found that while temperature had a strong **instantaneous** correlation with energy, its **lagged values showed weak predictive power** — confirming the white-noise-like nature of the series.
* This supported the causality test findings (by other collaborators) showing no causal influence from historical exogenous variables.

---

### ✅ **Baseline Modeling, ARIMA, and SARIMAX**

* Built naive and moving average baselines for 24-hour and 7-day forecast horizons.
* Conducted ARIMA grid search using time-series cross-validation.
* Benchmarked SARIMAX against ARIMA using both real and simulated exogenous inputs.
* Developed a framework to simulate exogenous variables (e.g., temperature, humidity) using cumulative random walks, enabling realistic multi-step forecasting.
* Trained SARIMAX using simulated exogenous inputs, achieving R² ≈ 0.33 even with injected uncertainty (σ ≈ 0.04% of target std).
* Extracted SARIMAX regression coefficients for feature relevance analysis — confirming temperature as the most influential regressor, even under noisy conditions.

---

### ✅ **Forecast App Deployment (Streamlit)**

* Developed and deployed a Streamlit web application to interactively forecast building energy usage.
* Enabled users to select forecast horizon (1–48 hours), run SARIMAX predictions, and visualize actual vs. forecasted trends.
* Integrated uncertainty handling by using pre-simulated noisy exogenous inputs for robust prediction.
* Hosted on [Streamlit Community Cloud](https://watt-wise-bob-app.streamlit.app) with preloaded model and data for fast and scalable access.


---

## 📂 Folder Structure

```plaintext
submissions/team/bob-hosseini/
│
├── notebooks/
│   ├── 01_EDA_Notebook.ipynb          # Data loading, preprocessing, and time series profiling
│   ├── 02_Model_Baseline_ARIMA.ipynb  # Baseline forecasts and ARIMA/SARIMAX modeling
│   ├── 03_Model_Export.ipynb          # Exporting the model and data splits
├── data/
│   ├── data.csv                       # Raw energy consumption raw data
│   ├── data_cleaned.pkl               # Cleaned version of the energy consumption dataset
│   ├── data_split.pkl                 # Data splits for training and testing
├── models/
│   ├── sarimax_checkpoint.json        # Final model checkpoint
├── app/
│   ├── app.py                         # Streamlit app
├── src/
│   ├── utils.py                       # Utility functions
├── README.md                          # This file
```

> *Note: Notebooks are self-contained and annotated for educational clarity.*

---

## 📊 Summary & Learnings

* **Exogenous Leakage Is Critical**: Using future values of variables like temperature leads to **data leakage** and overly optimistic performance. These values must be simulated or excluded in real forecasting.

* **Target Series is Largely Unpredictable**: Energy consumption behaves like **white noise**, showing minimal autocorrelation or predictable structure over time.

* **Lagged Exogenous Features Had Little Impact**: When using only past values of exogenous variables, **SARIMAX performance did not improve** over ARIMA — confirming weak predictive influence.

* **Realistic Forecasting with Simulated Inputs**: Injecting **random walk noise** into exogenous inputs allowed SARIMAX to simulate realistic forecast conditions and still achieve **R² ≈ 0.33**.

* **Feature Relevance via SARIMAX Coefficients**: Among all simulated features, **temperature consistently emerged as the most influential**, even under uncertainty.

* **Time-Series Cross-Validation Matters**: Integrating CV into the model selection pipeline ensured **reliable performance estimates** and prevented overfitting to a single test split.

---

## 🙌 Acknowledgments

Thanks to the SuperDataScience community and all collaborators, especially those who contributed to the project's discussions and shared their insights.  

---

## 🛠️ Tools Used

- Python, pandas, statsmodels
- SARIMAX (statsmodels)
- Streamlit (app deployment)
- Custom random walk simulation
- Matplotlib, Seaborn, Plotly


---

## 🚀 Deployment: Streamlit Web App

The forecasting tool is now live on **Streamlit Community Cloud**:  
👉 [Try the WattWise Forecasting App](https://watt-wise-bob-app.streamlit.app)

### App Features
- Select forecast horizon (1–48 hours)
- Visualize forecast results vs. historical trends
- Runs a pre-trained SARIMAX model using simulated exogenous inputs

> Model and data artifacts are preloaded for quick response. Forecast uncertainty is reflected via noise-injected exogenous features.

---

## 📦 How to Run the Streamlit App Locally

```bash
# 1. Clone the repo
git clone https://github.com/bab-git/SDS-CP027-watt-wise.git
cd SDS-CP027-watt-wise/submissions/team/bob-hosseini/

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app/app.py

```


---

## 📧 Contact / Connect

Feel free to reach out or follow me for more data science projects:

* [Bob Hosseini's Github Portfolio](https://github.com/bab-git)
* [LinkedIn](https://www.linkedin.com/in/bhosseini/)