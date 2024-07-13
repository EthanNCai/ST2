import datetime


def get_k_days_before(date_in, k):
    date = datetime.datetime.strptime(date_in, '%Y-%m-%d')
    dates = [date - datetime.timedelta(days=i) for i in range(k - 1, -1, -1)]
    dates_str = [date.strftime('%Y-%m-%d') for date in dates]
    return dates_str

print(get_k_days_before('2024-07-06', 5))
# 输出: ['2024-07-06', '2024-07-05', '2024-07-04', '2024-07-03', '2024-07-02']