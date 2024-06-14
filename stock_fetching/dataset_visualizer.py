import pandas as pd
import matplotlib.pyplot as plt
NAME = './HSI-10'
df = pd.read_csv(f"{NAME}-VOF.csv")

vol = df['vol']
vof = df['is_vol_outliers']  # vol means volume outlier flag
outliers_index = df.index[df['is_vol_outliers'] != 0]
outliers_vol:pd.Series = vol.loc[outliers_index]
# vol = vol.diff()

fig, axes = plt.subplots(figsize=(6, 4))
# ax.plot(vol_steps, close, c='r')
axes.scatter(outliers_index, outliers_vol, c='green', zorder=99, marker='x', s=15)
axes.plot(vol, c='r', linewidth='0.8',alpha = 0.5)

# print(outliers_vol)

plt.show()