import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    data_day = pd.DataFrame()
    for root, dirs, files in os.walk(r'data'):
        for file in files:
            file_path = os.path.join(root, file)
            if Path(file_path).suffix.lower() == '.csv':
                file_data = pd.read_csv(file_path)
                data_day = data_day.append(file_data, ignore_index=True)
    data_day.drop_duplicates(inplace=True)
    data_day['Date'] = pd.to_datetime(data_day['Date'], format='%d-%b-%Y')
    return data_day.sort_values(by=['Date'])


def window(series, window_size):
    length = series.shape[0]
    for i in range(length-window_size):
        yield series.iloc[i:i+window_size, :]


def pnl(series):
    row = pd.DataFrame(columns=['from', 'to', 'PNL', 'max_profit', 'max_loss'])
    pnl_est = 100*(series['Close'].iloc[-1] - series['Close'].iloc[0])/series['Close'].iloc[0]
    max_profit = 100*(series['Close'].max() - series['Close'].iloc[0])/series['Close'].iloc[0]
    max_loss = 100*(series['Close'].min() - series['Close'].iloc[0])/series['Close'].iloc[0]
    from_date = series['Date'].iloc[0]
    to_date = series['Date'].iloc[-1]
    row = row.append({'from': from_date, 'to': to_date, 'PNL': pnl_est, 'max-profit': max_profit, 'max_loss': max_loss},
                     ignore_index=True)
    return row


if __name__ == '__main__':
    data = load_data()
    a = window(data, 45)
    result = pd.DataFrame(columns=['from', 'to', 'PNL', 'max_profit', 'max_loss'])
    for series_window in a:
        result = result.append(pnl(series_window), ignore_index=True, sort=False)
    print(result['PNL'].describe(percentiles=[0.0015, 0.023, 0.159, 0.25, 0.5, 0.75, 0.841, 0.977, 0.9985]))
    bins = [-10+x*0.25 for x in range(81)]
    binned = pd.cut(result['PNL'], bins)
    plt.hist(result['PNL'], bins=bins)
    plt.xticks(bins, rotation=90)
    pltMgr = plt.get_current_fig_manager()
    pltMgr.window.state('zoomed')
    plt.show()
    print(binned)
