import pandas as pd 
from datetime import timedelta, datetime

def load_df_quotes(datum):
    coltypes = {'exchange': str, 'symbol': str, 'timestamp': int, 'local_timestamp': int, 'ask_amount': float, 
                'ask_price': float, 'bid_price': float, 'bid_amount': float}
    return pd.read_csv(f'/Users/marekerben/Desktop/Prakticka/binance-futures/BTCUSDT/quotes/{datum}.csv.gz', dtype = coltypes, 
            index_col=['timestamp'], parse_dates=['timestamp'], date_parser=lambda d: pd.to_datetime(d, unit = 'us'))    

def load_df_trades(datum):
    coltypes = {'exchange': str, 'symbol': str, 'timestamp': int, 'local_timestamp': int, 'id': int, 
                'side': str, 'price': float, 'amount': float}
    return pd.read_csv(f'/Users/marekerben/Desktop/Prakticka/binance-futures/BTCUSDT/trades/{datum}.csv.gz', dtype = coltypes, 
            index_col=['timestamp'], parse_dates=['timestamp'], date_parser=lambda d: pd.to_datetime(d, unit = 'us'))   

def get_e_n(data_frame):
    return (data_frame['bid_price'].diff() >= 0) * data_frame['bid_amount'] - \
        (data_frame['bid_price'].diff() <= 0) * data_frame['bid_amount'].shift(1)- \
        (data_frame['ask_price'].diff() <= 0) * data_frame['ask_amount'] + \
        (data_frame['ask_price'].diff() >= 0) * data_frame['ask_amount'].shift(1)

def get_mid_price(ask_price, bid_price, tick_size = 0.01):
    return (ask_price + bid_price) / (2 * tick_size)

def get_signed_amount_traded(data_frame):
    return data_frame.apply(lambda row: row['amount'] if row['side'] == 'buy' else -1 * row['amount'], axis=1)

def construct_OFI_TFI_dataframe(datum, delta_t = '10S', tick_size = 0.01):
    quotes_df = load_df_quotes(datum)
    trades_df = load_df_trades(datum)

    return pd.DataFrame({'delta_midprice': get_mid_price(ask_price = quotes_df['ask_price'], \
        bid_price = quotes_df['bid_price'], tick_size=tick_size).resample(delta_t)\
            .apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0), 'OFI': get_e_n(quotes_df).resample(delta_t).sum(),\
             'TFI': get_signed_amount_traded(trades_df).resample(delta_t).sum()})

def date_range_list(start_date, end_date):
    list_of_dates = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    current_date = start_date
    while current_date <= end_date:
        list_of_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return list_of_dates

def output_df(start_date, end_date, delta_t = '10S', tick_size = 0.01):
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
    return 0.5 *(((((df['bid_price'].diff() < 0) * df['bid_amount'] + (df['bid_price'].diff() > 0) * df['bid_amount'].shift(1))).sum()/((df['bid_price'].diff() != 0).sum()) + 
(((df['ask_price'].diff() > 0) * df['ask_amount'] + (df['ask_price'].diff() < 0) * df['ask_amount'].shift(1))).sum()/((df['ask_price'].diff() != 0).sum())))

def get_all_avg_depths(start_date, end_date, time_int = '30Min'):
    dates_list = date_range_list(start_date = start_date, end_date = end_date)
    series = pd.concat([load_df_quotes(i).resample(time_int).apply(lambda x: get_avg_depth(x)) for i in dates_list], axis = 0)
    series.name = 'avg_depth'
    return series

def control(df):
    '''
    Dodělat je to kontrola jestli máme všechny řádky v našem spočítáném df, tedy jestli ve všech 10s intervalech byli kotace a jsou tam data.
    Stalo se mi, že když nebyli data, tak to ty řádky vynechalo. přes np.where(np.array == False)[0]
    '''
    start_timestamp = df.index.min()
    end_timestamp = df.index.max()
    # Generate a new complete date and time range with a 10-second frequency
    complete_index = pd.date_range(start=start_timestamp, end=end_timestamp, freq='10S')
    return sum(df.index == complete_index)
