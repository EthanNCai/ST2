import datetime



def get_prevention_zone(start_date, end_date):

    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += datetime.timedelta(days=1)
    return date_list


ret = get_prevention_zone('2024-03-09','2024-05-01')
print(ret)