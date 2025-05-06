# TODOS:
 - Line plot of energy consumption over time
- Seasonal decomposition (trend, seasonality, residual)
- Correlation matrix of numeric features
- Boxplots by hour/day (for insight into usage patterns)

**ARIMA/SARIMA Readiness Check**
- Check stationarity using Augmented Dickey-Fuller test
- Plot ACF and PACF for the energy consumption series
- If non-stationary, apply first-order differencing and recheck


**Feature Engineering**:
- Encode categorical variables (Holiday, DayOfWeek, HVACUsage, etc.)
Use .map() for binary, get_dummies() for multiclass with drop_first=True
- Add time-based features:
Hour of day
Day of week (as numeric)
Month
Weekend flag
- Create lag or rolling features (at least 1–2)
df['lag_1'] = df['EnergyConsumption'].shift(1)
df['rolling_mean_24'] = df['EnergyConsumption'].rolling(24).mean()

