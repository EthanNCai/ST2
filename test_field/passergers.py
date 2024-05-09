import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../datas/airline_passengers.csv')
timeseries = df[["Passengers"]].values.astype('float32')

print(df['Passengers'].tolist())
plt.plot(timeseries)
plt.show()