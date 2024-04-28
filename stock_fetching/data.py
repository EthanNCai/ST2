import tushare as ts
import pandas as pd


def fetch_stock_data():
    ts.set_token('e3c91942d639d3aab6e22959007672d625152044b6dc6918712345c2')
    pro = ts.pro_api()
    df = pro.us_daily(ts_code='AAPL', start_date='20240101', end_date='20040904')
    df.to_csv('SPX.csv')
    print(df.head)
    print(df.columns)


fetch_stock_data()
