import pandas as pd

from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split

df_full = pd.read_csv('recent_data.csv')
temp_d = '2022-06-07'
end = '2022-06-09'
df = data_split(df_full, temp_d, end)
df = df.sort_values(by='date', ascending=False)
print()
print(df)
print()
print(df.index.values)
print(df.loc[0, :])