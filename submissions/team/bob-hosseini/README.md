# 🧠 Watt Wise Submission – SuperDataScience Collaborative Project

Welcome! This folder contains my individual contribution to the **Watt Wise: Intelligent Time Series Energy Consumption Forecasting** project, a collaborative, open-source initiative hosted by the SuperDataScience community.

---

## 📌 Project Overview

**Watt Wise** is a community-driven time series forecasting project focused on analyzing and predicting building energy usage based on historical consumption patterns and contextual factors like weather, occupancy, HVAC, and lighting.

🔗 Main Project Repository: [Watt Wise on GitHub](https://github.com/SuperDataScience-Community-Projects/SDS-CP027-watt-wise)  
🗓️ Timeline: 5-week collaborative sprint  
🧠 Hosted by: SuperDataScience Community  

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

### ✅ **Baseline Modeling & ARIMA**

* Built naive and moving average baselines for 24-hour and 7-day horizons.
* Performed grid search with proper time-series cross-validation.
* ARIMA slightly outperformed the baseline but showed **low R²**, indicating high unpredictability in the target.

---

### ✅ **Feature Engineering & SARIMAX with Forecast-Time Simulations**

* Engineered categorical and lag-based features, including rolling stats.
* Addressed **data leakage** by avoiding use of future exogenous values in forecasting.
* Developed a framework to **simulate uncertain future values** (e.g. temperature, humidity) using **cumulative random walks**.
* Fitted SARIMAX using simulated exogenous inputs. Despite noise injection (σ ≈ 0.04% of target std), the model achieved an **R² of 0.33**, showing resilience under uncertainty.
* Extracted SARIMAX regression coefficients, confirming **temperature** as the most influential exogenous feature under realistic conditions.


---

## 📂 Folder Structure

```plaintext
submissions/team/bob-hosseini/
│
├── notebooks/
│   ├── 01_EDA_Notebook.ipynb          # Data loading, preprocessing, and time series profiling
│   ├── 02_Model_Baseline_ARIMA.ipynb  # Baseline forecasts and ARIMA/SARIMAX modeling
├── data/
│   ├── data.csv                       # Cleaned version of the energy consumption dataset
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
- Custom random walk simulation
- Matplotlib/Seaborn for visualizations

---

## 🚀 Next Phase: Deployment (Week 5)

### Streamlit App Creation

* [ ] Forecast feature: allow selection of next 24h or 7-day window
* [ ] Interactive dashboard: show historical trends and forecasts

### Hosting & Documentation

* [ ] Deploy app on Streamlit Community Cloud (or equivalent)
* [ ] Add README/user guide for easy onboarding

---

## 📧 Contact / Connect

Feel free to reach out or follow me for more data science projects:

* [Bob Hosseini's Github Portfolio](https://github.com/bab-git)
* [LinkedIn](https://www.linkedin.com/in/bhosseini/)