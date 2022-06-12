import datetime

import config

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

df = pd.read_csv('./saved_results/actions_history_sac.csv')

print(df.columns)
df['transaction_date'] = df.index
print(" The average return value per day in rupees is ", df['return_val'].mean())

print("**************************************************************************")

def timed_return_ratio(t):
    d = t * 30
    start_date = config.TEST_START_DATE
    temp = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = temp + timedelta(days=d)
    end_date = datetime.strftime(end_date, '%Y-%m-%d')

    df1 = df.loc[df['date'] >= start_date]
    df2 = df1.loc[df1['date'] <= end_date]
    # df2 = df.loc[start_date:end_date]
    print(f"average return value in rupees in the last {t} months is {df2['return_val'].mean()}")
    print(f"cumulative return value in rupees in the last {t} months is {df2['return_val'].sum()}")
    print()

time_list = [1, 3, 6, 12, 18, 24]

for t in time_list:
    timed_return_ratio(t)