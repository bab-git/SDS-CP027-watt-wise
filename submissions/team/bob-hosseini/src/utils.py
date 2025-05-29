import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import matplotlib.dates as mdates
import calendar

# Constants
FIGSIZE = (16, 5)
FIGSIZE_LONG=(20, 6)

def add_weekday_labels(fig):
    ''' This function adds weekday labels to the last plot.
    '''
    # Add weekday labels to the last plot
    ax = fig.axes[len(fig.axes) - 1]
    locs = ax.get_xticks()
    dates = [mdates.num2date(x) for x in locs]
    day_labels = [calendar.day_name[d.weekday()] for d in dates]
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(locs)
    secax.set_xticklabels(day_labels, rotation=0, fontsize=8)

def seasonal_decompose_stl(df_data_window, col, period, figsize = FIGSIZE):
    ''' This function decomposes the data into trend, seasonal, and residual components using STL.
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