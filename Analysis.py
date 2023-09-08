import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_OLS_results(dataset, lags, formula = 'delta_midprice ~ OFI', time_int = '30Min', covariation_type = 'HAC'):
    return dataset.groupby(pd.Grouper(freq = time_int)).apply(lambda dataset: sm.formula.ols(formula, data = dataset)
    .fit(cov_type = covariation_type, cov_kwds={'maxlags':lags}))

def get_beta_coef(OLS_results, Flow_imbalance = 'OFI'):
    aux_list = []
    for i in OLS_results:
        aux_list.append(i.params[Flow_imbalance])
    return pd.Series(aux_list)

def create_dataframe(beta_series, depth_series):
    start_datetime = depth_series.index.min()
    end_datetime = depth_series.index.max()

    datetime_index = pd.date_range(start=start_datetime, end=end_datetime, freq='30T')
    
    beta_series = pd.Series(beta_series.values, index=datetime_index, name='timestamp')

    return pd.DataFrame({'beta_coef': beta_series, 'avg_depth': depth_series})    

def finished_df(start_date, end_date, lags = 4, formula = 'delta_midprice ~ OFI', time_int = '30Min', covariation_type = 'HAC', Flow_imbalance='OFI'):
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
    times = []
    for hour in range(24):
        for minute in ['00', '30']:
            time_str = f"{hour:02d}:{minute}"
            times.append(time_str)
    return times

def get_graph(beta_D_data):
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
