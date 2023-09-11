import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_OLS_results(dataset, lags, formula = 'delta_midprice ~ OFI', time_int = '30Min', covariation_type = 'HAC'):
    '''Perform Ordinary Least Squares (OLS) regression on a dataset using a specified formula.

    Parameters
    ----------
    dataset : pd.DataFrame
        A DataFrame containing the data for regression analysis.

    lags : int
        The maximum number of lags to consider for autocorrelation.

    formula : str, optional
        The formula for the OLS regression model (default is 'delta_midprice ~ OFI').

    time_int : str, optional
        The time interval for grouping the data (default is '30Min').

    covariation_type : str, optional
        The type of covariance estimator for standard errors (default is 'HAC' - Heteroskedasticity and 
        Autocorrelation Consistent).

    Returns
    -------
    pd.Series
        A Series containing OLS regression results for each time interval, including coefficients,
        standard errors, and other statistics.

    Description
    -----------
    This function performs Ordinary Least Squares (OLS) regression on a dataset using the specified
    formula. It groups the data by the specified time interval, fits the OLS model for each group,
    and returns the regression results as a Series.

    Example
    -------
    To perform OLS regression on a dataset `df` with a maximum of 2 lags, you can call the function like this:
    
    >>> results = get_OLS_results(df, lags=2, formula='delta_midprice ~ OFI', time_int='30Min', covariation_type='HAC')
    '''
    return dataset.groupby(pd.Grouper(freq = time_int)).apply(lambda dataset: sm.formula.ols(formula, data = dataset)
    .fit(cov_type = covariation_type, cov_kwds={'maxlags':lags}))

def get_beta_coef(OLS_results, Flow_imbalance = 'OFI'):
    '''Extract beta coefficients from a series of OLS regression results.

    Parameters
    ----------
    OLS_results : pd.Series
        A Series containing OLS regression results, typically obtained from the `get_OLS_results` function.

    Flow_imbalance : str, optional
        The flow imbalance variable for which beta coefficients should be extracted (default is 'OFI').

    Returns
    -------
    pd.Series
        A Series containing beta coefficients for the specified flow imbalance variable, indexed by time interval.

    Description
    -----------
    This function extracts beta coefficients (slope) from a series of OLS regression results for a specific
    flow imbalance variable. It returns the beta coefficients as a Series, with each coefficient corresponding
    to a time interval.

    Example
    -------
    To extract beta coefficients for the 'OFI' variable from a series of OLS results `results`, you can call
    the function like this:
    
    >>> beta_coefs = get_beta_coef(results, Flow_imbalance='OFI')
    '''
    aux_list = []
    for i in OLS_results:
        aux_list.append(i.params[Flow_imbalance])
    return pd.Series(aux_list)

def create_dataframe(beta_series, depth_series):
    '''Create a DataFrame from beta coefficients and average depth series.

    Parameters
    ----------
    beta_series : pd.Series
        A Series containing beta coefficients indexed by time interval.

    depth_series : pd.Series
        A Series containing average depth values indexed by time interval.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing beta coefficients and average depth values, aligned by timestamp.

    Description
    -----------
    This function creates a DataFrame by aligning beta coefficients and average depth values based on
    their timestamps. It ensures that the resulting DataFrame has a common timestamp index for both
    beta coefficients and average depth values.

    Example
    -------
    To create a DataFrame from beta coefficients in `beta_series` and average depth values in `depth_series`,
    you can call the function like this:
    
    >>> df = create_dataframe(beta_series, depth_series)
    '''
    start_datetime = depth_series.index.min()
    end_datetime = depth_series.index.max()

    datetime_index = pd.date_range(start=start_datetime, end=end_datetime, freq='30T')
    
    beta_series = pd.Series(beta_series.values, index=datetime_index, name='timestamp')

    return pd.DataFrame({'beta_coef': beta_series, 'avg_depth': depth_series})    

def finished_df(start_date, end_date, lags = 4, formula = 'delta_midprice ~ OFI', time_int = '30Min', covariation_type = 'HAC', Flow_imbalance='OFI'):
    '''Generate a DataFrame containing beta coefficients, average depth values, and other metrics for a specified date range.

    Parameters
    ----------
    start_date : str
        The start date in the format 'YYYY-MM-DD'.

    end_date : str
        The end date in the format 'YYYY-MM-DD'.

    lags : int, optional
        The maximum number of lags to consider for autocorrelation (default is 4).

    formula : str, optional
        The formula for the OLS regression model (default is 'delta_midprice ~ OFI').

    time_int : str, optional
        The time interval for resampling the data (default is '30Min').

    covariation_type : str, optional
        The type of covariance estimator for standard errors (default is 'HAC' - Heteroskedasticity and 
        Autocorrelation Consistent).

    Flow_imbalance : str, optional
        The flow imbalance variable for which beta coefficients should be extracted (default is 'OFI').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing beta coefficients, average depth values, and other metrics for the specified date range.

    Description
    -----------
    This function reads data from CSV files, performs OLS regression, extracts beta coefficients, and creates
    a DataFrame containing beta coefficients, average depth values, and other metrics. The data is filtered
    to include only the specified date range.

    Example
    -------
    To generate a DataFrame for beta coefficients, average depth values, and other metrics from '2023-09-11'
    to '2023-09-15', you can call the function like this:
    
    >>> df = finished_df('2023-09-11', '2023-09-15', lags=4, formula='delta_midprice ~ OFI', time_int='30Min', covariation_type='HAC', Flow_imbalance='OFI')
    '''
    column_types = {
        'delta_midprice': 'float64',
        'OFI': 'float64',
        'TFI': 'float64'
        }

    data = pd.read_csv('Data/2020-11-15_2020-11-30.csv', dtype=column_types, index_col=0, parse_dates=True)
    D = pd.read_csv('Data/avg_depths-2020-11.csv', index_col=0, parse_dates=True)
    D = D.squeeze()

    beta = get_beta_coef(get_OLS_results(data, lags=lags, formula=formula, time_int=time_int, covariation_type=covariation_type), 
                         Flow_imbalance=Flow_imbalance)
    
    if end_date > start_date:
        filtered_data = create_dataframe(beta, D)[start_date:end_date]
        return filtered_data
    else:
        print('End date should be greater than start date.')
        return None

def list_of_halfhour_timestamps():
    '''Generate a list of timestamps representing half-hour intervals in a 24-hour day.

    Returns
    -------
    list
        A list of timestamps in the format 'HH:MM' representing half-hour intervals in a day.

    Description
    -----------
    This function generates a list of timestamps representing half-hour intervals in a 24-hour day.
    The timestamps are in the 'HH:MM' format, where 'HH' represents the hour and 'MM' represents
    the minute.

    Example
    -------
    To generate a list of half-hour timestamps, you can call the function like this:
    
    >>> timestamps = list_of_halfhour_timestamps()
    '''
    times = []
    for hour in range(24):
        for minute in ['00', '30']:
            time_str = f"{hour:02d}:{minute}"
            times.append(time_str)
    return times

def get_graph(beta_D_data):
    '''Generate a line plot showing normalized beta and depth values over half-hour intervals.

    Parameters
    ----------
    beta_D_data : pd.DataFrame
        A DataFrame containing beta coefficients, average depth values, and timestamps.

    Returns
    -------
    None
        Displays a line plot of normalized beta and depth values.

    Description
    -----------
    This function takes a DataFrame `beta_D_data` containing beta coefficients, average depth values, and
    timestamps. It calculates the global means of beta and depth, normalizes the data, and then generates
    a line plot to visualize the normalized values over half-hour intervals.

    Example
    -------
    To generate and display a line plot for normalized beta and depth values from a DataFrame `df`, you can
    call the function like this:
    
    >>> get_graph(df)
    '''
    halfhour_list = list_of_halfhour_timestamps() 
    beta_D_data['time'] = beta_D_data.index.strftime('%H:%M:%S')

    global_depth = beta_D_data.avg_depth.mean()
    global_beta = beta_D_data.beta_coef.mean()
    means_beta = beta_D_data.groupby('time').beta_coef.mean()
    means_depth = beta_D_data.groupby('time').avg_depth.mean()
    
    normalized_beta = means_beta/global_beta
    normalized_depth = means_depth/global_depth

    plt.plot(halfhour_list, normalized_beta ,marker = 's', markersize=5)
    plt.plot(halfhour_list, normalized_depth, linestyle = '-', marker = '^', markersize=5)
    plt.xlabel('Hours')
    ticks, labels = plt.xticks()
    plt.xticks(rotation=90, ticks = ticks[::2])
    #plt.axhline(x = '21:00', color='red', linestyle='--', label='NYSE close')
    plt.legend(labels = ['beta', 'depth', 'lambda', 'c'], loc='lower center', )
    return plt.show()
