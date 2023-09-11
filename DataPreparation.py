import pandas as pd 
from datetime import timedelta, datetime

def load_df_quotes(datum):
    '''Load a DataFrame containing cryptocurrency quote data from a CSV file.

    Parameters:
    -----------
    datum : str
        The date for which you want to load the quote data in the format 'YYYY-MM-DD'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing cryptocurrency quote data with columns for exchange, symbol,
        timestamp, local timestamp, ask amount, ask price, bid price, and bid amount.

    Example:
    --------
    To load quote data for the date '2023-09-11', you can call the function like this:
    
    >>> df = load_df_quotes('2023-09-11')
    '''
    coltypes = {'exchange': str, 'symbol': str, 'timestamp': int, 'local_timestamp': int, 'ask_amount': float, 
                'ask_price': float, 'bid_price': float, 'bid_amount': float}
    return pd.read_csv(f'/Users/marekerben/Desktop/Prakticka/binance-futures/BTCUSDT/quotes/{datum}.csv.gz', dtype = coltypes, 
            index_col=['timestamp'], parse_dates=['timestamp'], date_parser=lambda d: pd.to_datetime(d, unit = 'us'))    

def load_df_trades(datum):
    '''Load a DataFrame containing cryptocurrency trade data from a CSV file.

    Parameters
    ----------
    datum : str
        The date for which you want to load the trade data in the format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing cryptocurrency trade data with columns for exchange, symbol,
        timestamp, local timestamp, trade ID, side, price, and amount.

    Example
    -------
    To load trade data for the date '2023-09-11', you can call the function like this:
    
    >>> df = load_df_trades('2023-09-11')
    '''
    coltypes = {'exchange': str, 'symbol': str, 'timestamp': int, 'local_timestamp': int, 'id': int, 
                'side': str, 'price': float, 'amount': float}
    return pd.read_csv(f'/Users/marekerben/Desktop/Prakticka/binance-futures/BTCUSDT/trades/{datum}.csv.gz', dtype = coltypes, 
            index_col=['timestamp'], parse_dates=['timestamp'], date_parser=lambda d: pd.to_datetime(d, unit = 'us'))   

def get_e_n(data_frame):
    '''Calculate the net exchange flow (e_n) for a given DataFrame of cryptocurrency data.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A DataFrame containing cryptocurrency data with columns for bid price, bid amount, 
        ask price, and ask amount.

    Returns
    -------
    pd.Series
        A Series representing the net exchange flow (e_n) calculated based on the provided data.

    Description
    -----------
    The net exchange flow (e_n) is calculated as the difference between the cumulative bid amount 
    for price increases and the cumulative ask amount for price decreases at each timestamp. 
    It helps analyze the flow of orders in the cryptocurrency market.

    Example
    -------
    To calculate the net exchange flow (e_n) for a DataFrame `df`, you can call the function like this:
    
    >>> en = get_e_n(df)
    '''
    return (data_frame['bid_price'].diff() >= 0) * data_frame['bid_amount'] - \
        (data_frame['bid_price'].diff() <= 0) * data_frame['bid_amount'].shift(1)- \
        (data_frame['ask_price'].diff() <= 0) * data_frame['ask_amount'] + \
        (data_frame['ask_price'].diff() >= 0) * data_frame['ask_amount'].shift(1)

def get_mid_price(ask_price, bid_price, tick_size = 0.01):
    '''Calculate the mid price based on the ask and bid prices.

    Parameters
    ----------
    ask_price : float
        The ask price of the cryptocurrency.

    bid_price : float
        The bid price of the cryptocurrency.

    tick_size : float, optional
        The tick size used for the mid price calculation (default is 0.01).

    Returns
    -------
    float
        The calculated mid price of the cryptocurrency.

    Description
    -----------
    The mid price is calculated as the average of the ask and bid prices, divided by
    twice the tick size. It represents the central price point between the current
    best ask and best bid prices in the market.

    Example
    -------
    To calculate the mid price for an ask price of 100.0, a bid price of 99.5, and a tick size of 0.01, 
    you can call the function like this:
    
    >>> mid_price = get_mid_price(100.0, 99.5, 0.01)
    '''
    return (ask_price + bid_price) / (2 * tick_size)

def get_signed_amount_traded(data_frame):
    '''Calculate the signed amount traded based on the side (buy/sell) in a DataFrame.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A DataFrame containing cryptocurrency trade data with columns for 'side' and 'amount'.

    Returns
    -------
    pd.Series
        A Series representing the signed amount traded, where buy trades have positive values
        and sell trades have negative values.

    Description
    -----------
    This function calculates the signed amount traded for each row in the DataFrame. If the trade
    side is 'buy,' the amount is positive; if the trade side is 'sell,' the amount is negated
    to represent a sell trade.

    Example
    -------
    To calculate the signed amount traded for a DataFrame `df` containing trade data, you can call
    the function like this:
    
    >>> signed_amount = get_signed_amount_traded(df)
    '''
    return data_frame.apply(lambda row: row['amount'] if row['side'] == 'buy' else -1 * row['amount'], axis=1)

def construct_OFI_TFI_dataframe(datum, delta_t = '10S', tick_size = 0.01):
    '''Construct a DataFrame containing Order Flow Imbalance (OFI) and Traded Flow Imbalance (TFI) data.

    Parameters
    ----------
    datum : str
        The date for which you want to construct the OFI and TFI DataFrame in the format 'YYYY-MM-DD'.

    delta_t : str, optional
        The time interval for resampling the data (default is '10S' - 10 seconds).

    tick_size : float, optional
        The tick size used for mid price calculation (default is 0.01).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'delta_midprice' (change in mid price), 'OFI' (Order Flow Imbalance), 
        and 'TFI' (Traded Flow Imbalance) data resampled at the specified time interval.

    Description
    -----------
    This function loads quote and trade data for a specific date, calculates the change in mid price,
    Order Flow Imbalance (OFI), and Traded Flow Imbalance (TFI) over the specified time intervals, and 
    constructs a DataFrame with these metrics.

    Example
    -------
    To construct an OFI and TFI DataFrame for the date '2023-09-11' with a time interval of '10S', 
    you can call the function like this:
    
    >>> df = construct_OFI_TFI_dataframe('2023-09-11', delta_t='10S', tick_size=0.01)
    '''
    quotes_df = load_df_quotes(datum)
    trades_df = load_df_trades(datum)

    return pd.DataFrame({'delta_midprice': get_mid_price(ask_price = quotes_df['ask_price'], \
        bid_price = quotes_df['bid_price'], tick_size=tick_size).resample(delta_t)\
            .apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0), 'OFI': get_e_n(quotes_df).resample(delta_t).sum(),\
             'TFI': get_signed_amount_traded(trades_df).resample(delta_t).sum()})

def date_range_list(start_date, end_date):
    '''Generate a list of dates within a specified date range.

    Parameters
    ----------
    start_date : str
        The start date in the format 'YYYY-MM-DD'.

    end_date : str
        The end date in the format 'YYYY-MM-DD'.

    Returns
    -------
    list
        A list of date strings within the specified date range.

    Description
    -----------
    This function generates a list of dates, including the start date and end date, within the specified
    date range. The dates are represented as strings in the 'YYYY-MM-DD' format.

    Example
    -------
    To generate a list of dates from '2023-09-11' to '2023-09-15', you can call the function like this:
    
    >>> date_list = date_range_list('2023-09-11', '2023-09-15')
    '''
    list_of_dates = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    current_date = start_date
    while current_date <= end_date:
        list_of_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return list_of_dates

def output_df(start_date, end_date, delta_t = '10S', tick_size = 0.01):
    '''Generate and concatenate Order Flow Imbalance (OFI) and Traded Flow Imbalance (TFI) DataFrames
    for a specified date range.

    Parameters
    ----------
    start_date : str
        The start date in the format 'YYYY-MM-DD'.

    end_date : str
        The end date in the format 'YYYY-MM-DD'.

    delta_t : str, optional
        The time interval for resampling the data (default is '10S' - 10 seconds).

    tick_size : float, optional
        The tick size used for mid price calculation (default is 0.01).

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing OFI and TFI data for the specified date range.

    Description
    -----------
    This function generates OFI and TFI DataFrames for a range of dates within the specified date
    range. It catches and handles potential errors during the data processing and concatenates the
    results into a single DataFrame.

    Example
    -------
    To generate OFI and TFI DataFrames for the date range from '2023-09-11' to '2023-09-15' with
    a time interval of '10S' and a tick size of 0.01, you can call the function like this:
    
    >>> df = output_df('2023-09-11', '2023-09-15', delta_t='10S', tick_size=0.01)
    '''
    dates_list = date_range_list(start_date, end_date)

    results = []
    for i in dates_list:
        try:
            results.append(construct_OFI_TFI_dataframe(datum = i, delta_t = delta_t, tick_size = tick_size))
            print(f'{i} done!')
        except IndexError:
            print(f'Index error: {i}')
        except:
            print(f'Other error: {i}')
    
    return pd.concat(results)

def get_avg_depth(df):
    '''Calculate the average depth of the order book from a DataFrame of cryptocurrency quote data.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing cryptocurrency quote data with columns for bid price and bid amount,
        as well as ask price and ask amount.

    Returns
    -------
    float
        The calculated average depth of the order book.

    Description
    -----------
    The average depth of the order book is calculated as the average of the sum of bid and ask amounts
    for price decreases and price increases, respectively, divided by the count of price changes.

    Example
    -------
    To calculate the average depth of the order book for a DataFrame `df`, you can call the function like this:
    
    >>> avg_depth = get_avg_depth(df)
    '''
    return 0.5 *(((((df['bid_price'].diff() < 0) * df['bid_amount'] + (df['bid_price'].diff() > 0) * df['bid_amount'].shift(1))).sum()/((df['bid_price'].diff() != 0).sum()) + 
(((df['ask_price'].diff() > 0) * df['ask_amount'] + (df['ask_price'].diff() < 0) * df['ask_amount'].shift(1))).sum()/((df['ask_price'].diff() != 0).sum())))

def get_all_avg_depths(start_date, end_date, time_int = '30Min'):
    '''Calculate the average depth of the order book over a specified date range and time interval.

    Parameters
    ----------
    start_date : str
        The start date in the format 'YYYY-MM-DD'.

    end_date : str
        The end date in the format 'YYYY-MM-DD'.

    time_int : str, optional
        The time interval for resampling the data (default is '30Min').

    Returns
    -------
    pd.Series
        A Series containing the calculated average depths of the order book, indexed by timestamp.

    Description
    -----------
    This function calculates the average depth of the order book for a range of dates within the specified
    date range and resamples the data at the provided time interval. The results are returned as a Series.

    Example
    -------
    To calculate the average depths of the order book from '2023-09-11' to '2023-09-15' with a time
    interval of '30Min', you can call the function like this:
    
    >>> avg_depths = get_all_avg_depths('2023-09-11', '2023-09-15', time_int='30Min')
    '''
    dates_list = date_range_list(start_date = start_date, end_date = end_date)
    series = pd.concat([load_df_quotes(i).resample(time_int).apply(lambda x: get_avg_depth(x)) for i in dates_list], axis = 0)
    series.name = 'avg_depth'
    return series

#def control(df):
    '''
    Dodělat je to kontrola jestli máme všechny řádky v našem spočítáném df, tedy jestli ve všech 10s intervalech byli kotace a jsou tam data.
    Stalo se mi, že když nebyli data, tak to ty řádky vynechalo. přes np.where(np.array == False)[0]
    '''
    start_timestamp = df.index.min()
    end_timestamp = df.index.max()
    # Generate a new complete date and time range with a 10-second frequency
    complete_index = pd.date_range(start=start_timestamp, end=end_timestamp, freq='10S')
    return sum(df.index == complete_index)
