import calendar
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Constants
FIGSIZE = (16, 5)
FIGSIZE_LONG=(20, 6)

# ===============================================
# Adding weekday labels to the last plot
# ===============================================
def add_weekday_labels(fig):
    ''' 
    """
    Add weekday labels to the last plot in a figure.
    
    This function adds weekday labels to the top x-axis of the last subplot in a figure.
    It converts the numeric x-axis ticks to dates and then to weekday names.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the plot to add weekday labels to.
        
    Returns
    -------
    None
        The function modifies the figure in-place by adding a secondary x-axis with weekday labels.
    """
    '''
    # Add weekday labels to the last plot
    ax = fig.axes[len(fig.axes) - 1]
    locs = ax.get_xticks()
    dates = [mdates.num2date(x) for x in locs]
    day_labels = [calendar.day_name[d.weekday()] for d in dates]
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(locs)
    secax.set_xticklabels(day_labels, rotation=0, fontsize=8)

# ===============================================
# Seasonal Decomposition using STL
# ===============================================
def seasonal_decompose_stl(df_data_window, col, period, figsize = FIGSIZE):
    ''' 
    """
    Perform seasonal decomposition using STL (Seasonal-Trend decomposition using LOESS) and visualize the results.
    
    Parameters
    ----------
    df_data_window : pandas.DataFrame
        DataFrame containing the time series data to decompose
    col : str
        Name of the column containing the time series to decompose
    period : int
        Period of the seasonal component (e.g., 24 for hourly data with daily seasonality)
    figsize : tuple, optional
        Figure size in inches (width, height), by default FIGSIZE
        
    Returns
    -------
    tuple
        A tuple containing:
        - fig : matplotlib.figure.Figure
            The figure containing the decomposition plots
        - ratio_std : float
            Ratio of residual standard deviation to original series standard deviation
        - ratio_range : float
            Ratio of residual range to seasonal amplitude
        - explained_var : float
            Proportion of variance explained by the decomposition
    """
    '''
    # decomposition = seasonal_decompose(df_data_window[col], model='additive', period=24)
    stl = STL(df_data_window[col], period=period, robust=True)
    decomposition = stl.fit()

    resid = decomposition.resid.dropna()           # or STL.resid
    ratio_std = resid.std() / df_data_window[col].std()
    print(f"Residual σ / Original σ = {ratio_std:.2%}")
    explained_var = 1 - (resid.var() / df_data_window[col].var())
    print(f"Explained variance = {explained_var:.2%}")

    season = decomposition.seasonal
    seas_range = season.max() - season.min()
    resid_range = resid.max() - resid.min()
    ratio_range = resid_range/seas_range
    print(f"Residual range / Seasonal amplitude = {ratio_range:.2%}")

    fig = decomposition.plot()
    fig.set_size_inches(figsize)
    add_weekday_labels(fig)
    plt.tight_layout()
    for ax in fig.axes:
        ax.grid(True)
    plt.show()
    return fig, ratio_std, ratio_range, explained_var


# ===============================================
# SARIMAX model
# ===============================================
def sarimax_model(train, test,
                  order, seasonal_order,
                  exog_train=None, exog_test=None):
    """    
    Fit a SARIMAX model and generate forecasts.

    Parameters
    ----------
    train : pd.Series
        Training data series
    test : pd.Series
        Test data series
    order : tuple
        The (p,d,q) order of the model for the ARIMA portion
    seasonal_order : tuple
        The (P,D,Q,s) order of the seasonal component
    exog_train : pd.DataFrame, optional
        Exogenous variables for training, by default None
    exog_test : pd.DataFrame, optional
        Exogenous variables for testing, by default None

    Returns
    -------
    tuple
        A tuple containing:
        - df_pred : pd.DataFrame
            DataFrame with predictions and actual values
        - res : statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
            Fitted SARIMAX model results
    """
    model = SARIMAX(train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
            message=".*divide by zero encountered.*|.*overflow encountered.*|.*invalid value encountered.*|.*Using zeros.*|.*Maximum Likelihood optimization*.|.*Too few observations*.")
        res = model.fit(disp=False)

    # Forecast 
    forecast = res.forecast(steps=len(test),
                            exog=exog_test)

    df_pred = pd.DataFrame(index=test.index)
    df_pred['prediction'] = forecast.values
    df_pred['truth']      = test.values
    return df_pred, res

# ===============================================
# Evaluating the model
# ===============================================
def evaluate_model(df_predictions, verbose=True):
    """
    Evaluate the performance of a forecasting model using multiple metrics.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        DataFrame containing prediction and truth columns
    verbose : bool, optional
        Whether to print the evaluation metrics, by default True

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error 
        - mape: Mean Absolute Percentage Error
        - r2: R-squared score
    """
    y_pred = df_predictions['prediction'].values
    y_true = df_predictions['truth'].values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)

    if verbose:
        print("Rolling-Window Forecast Evaluation:")
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R2:   {r2:.4f}")

    results = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

    return results

# ===============================================
# Visualizing the prediction
# ===============================================
def visualize_prediction(df_predictions, models, df_data, title = None, last_n_samples=48):
    """
    Visualize the predictions of multiple models against actual data.

    Parameters
    ----------
    df_predictions : list of pd.DataFrame
        List of DataFrames containing prediction and truth columns for each model
    models : list of str
        List of model names corresponding to each prediction DataFrame
    df_data : pd.DataFrame
        DataFrame containing the training data with 'EnergyConsumption' column
    title : str, optional
        Custom title for the plot, by default None
    last_n_samples : int, optional
        Number of last samples from training data to plot, by default 48

    Returns
    -------
    None
        Displays the plot
    """

    df_viz = df_data.iloc[-last_n_samples:].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(df_viz.index, df_viz['EnergyConsumption'],  label="Training Data",    marker='o')
    def_pred = df_predictions[0]
    plt.plot(def_pred.index, def_pred['truth'],  label="Test Data",    marker='o')
    for def_pred, model in zip(df_predictions, models):
        # print(def_pred, model)
        plt.plot(def_pred.index, def_pred['prediction'],  label=f"Forecast-{model}",  marker='x', linestyle='--')
    if title == None:
        title = f"Forecasting the 24-Hour window of Energy consumption - Test data"
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===============================================
# Adding random noise to the exogenous variables
# ===============================================
def add_uncertainty(exog_train, exog_test, col, noise_ratio, verbose=True):
    """
    Add uncertainty to exogenous variables by introducing random noise.

    Parameters
    ----------
    exog_train : pd.DataFrame
        Training set exogenous variables
    exog_test : pd.DataFrame or None
        Test set exogenous variables. If None, only training set is modified
    col : str
        Column name to add uncertainty to
    noise_ratio : float
        Ratio of noise standard deviation to feature standard deviation
    verbose : bool, optional
        Whether to plot the results, by default True

    Returns
    -------
    pd.DataFrame or tuple
        If exog_test is None, returns modified exog_train
        Otherwise returns tuple of (modified exog_train, modified exog_test)
    """
    noise_scale = noise_ratio * exog_train[col].std()

    # Uncertainty for training set
    uncertainey_train = pd.Series(np.random.normal(0, noise_scale, size=len(exog_train)), index=exog_train.index)    
    exog_train[col] = exog_train[col] + uncertainey_train
    if exog_test is None:
        return exog_train
    
    # Uncertainty for test set
    N = len(exog_test)
    factors = np.arange(1, N+1)
    stds = noise_scale * factors    
    noise_values_test = [np.random.normal(0, s) for s in stds]    
    uncertainty_test = pd.Series(noise_values_test, index=exog_test.index)
    
    value_test = exog_test[col].copy()
    exog_test[col] = exog_test[col] + uncertainty_test

    if verbose:
        # plot Tempreture and cumulative noise of test set
        plt.figure(figsize=(10, 4))
        plt.plot(exog_test.index, exog_test[col], label=col)
        plt.plot(exog_test.index, value_test, label=f'{col} with Incremental Uncertainty')
        plt.title(f'{col} (test set) with Incremental Uncertainty, noise ratio {noise_ratio:.2f}')
        plt.legend()
        plt.grid(True)
        plt.show()
    return exog_train, exog_test