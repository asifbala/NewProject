import pandas as pd 
import matplotlib.pyplot as plt


time_format = '%Y-%m-%d %H:%M'

data = pd.read_json('/Users/asifbala/Documents/ultimate_challenge/logins.json')

date_list = list(data['login_time'])

data = pd.to_datetime(date_list, format=time_format)

series = pd.Series(range(len(data)), index=data)

series.index.name = 'Date'

series.name= 'Counts'

series_15_min = series.resample('15T').count()

print(series_15_min)

series_hourly = series.resample('H').count()

print(series_hourly)

series_daily = series.resample('D').count()

print(series_daily)

series_weekly = series.resample('W').count()

print(series_weekly)

series_monthly = series.resample('M').count()

print(series_monthly)

series_15_min_plot = series_15_min.plot()

series_15_min_plot.set_xlabel('Date in 15 Minute Intervals')

series_15_min_plot.set_ylabel('Number of logins')

plt.show()

series_hourly_plot = series_hourly.plot()

series_hourly_plot.set_xlabel('Date in Hourly Intervals')

series_hourly_plot.set_ylabel('Number of logins')

plt.show()

series_daily_plot = series_daily.plot()

series_daily_plot.set_xlabel('Date in Daily Intervals')

series_daily_plot.set_ylabel('Number of logins')

plt.show()

series_weekly_plot = series_weekly.plot()

series_weekly_plot.set_xlabel('Date in Weekly Intervals')

series_weekly_plot.set_ylabel('Number of logins')

plt.show()

series_monthly_plot = series_monthly.plot()

series_monthly_plot.set_xlabel('Date in Monthly Intervals')

series_monthly_plot.set_ylabel('Number of logins')

plt.show()

