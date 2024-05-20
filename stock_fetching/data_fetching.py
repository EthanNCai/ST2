import tushare as ts
import pandas as pd

"""
HSI
DJI
IXIC
N225
CSI <- another api
CSX5P
"""


def fetch_stock_data():
    ts.set_token('e3c91942d639d3aab6e22959007672d625152044b6dc6918712345c2')
    pro = ts.pro_api()
    start_day = '20140401'
    end_day = '20240401'
    df = pro.index_global(ts_code='HSI', start_date=start_day, end_date=end_day)
    df.to_csv(f'HSI-10.csv')
    print(df.head)
    print(df.columns)


fetch_stock_data()
