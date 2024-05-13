import pandas as pd

OUTPUT_NAME = './SPX.csv'

df1 = pd.read_csv('./SPX_20040401_20240401.csv')
df2 = pd.read_csv('./SPX_20040401_20080508.csv')

df_cat = pd.concat([df1,df2])

# print(df_cat.head())
# df_cat = df_cat['close']
df_reversed = df_cat[::-1]


print(df_reversed.head())
print(df_reversed.tail())

df_reversed.to_csv(OUTPUT_NAME)