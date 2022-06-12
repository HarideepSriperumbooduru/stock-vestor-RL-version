# import inline as inline
import tkinter
from DataProcessor import DataProcessor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyfolio as pf
import datetime
import config
# %matplotlib inline

# from finrl.apps import config
from finrl.train import train
from finrl.test import test
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint
import itertools
import sys

DP = DataProcessor(data_source='yahoofinance')
data = DP.download_data(start_date=config.TRAIN_START_DATE,
                        end_date=config.TRAIN_END_DATE,
                        ticker_list=config.TICKERS_LIST,
                        time_interval='1d')

data = DP.clean_data(data)
data = DP.add_technical_indicator(data, config.TECHNICAL_INDICATORS_LIST)
data = DP.add_vix(data)
data = DP.add_stock_split(data)

try:
    data.to_csv('MultistockData.csv')
except:
    pass
if config.comments_bool:
    # comment this block later
    print("initial data after adding tech indicators and processing")
    print(data.head(50))
    print("************************************************")
    #########################

price_array, tech_array, turbulence_array, split_array = DP.df_to_array(data, if_vix='True')

# k = np.where(price_array == 419.715087890625)[0]
# print(f'index val in price array is {k} and val in split_array is {split_array[k]} ')

if config.comments_bool:
    # comment this block later
    print("initial data after dividing into four arrays")
    print("split vals array is of len ", len(split_array))
    print(list(split_array))
    print("price array is of len ", len(price_array))
    print(list(price_array))
    print("************************************************")
    print()
    print("tech array is of length ", len(tech_array))
    print(list(tech_array))
    print("************************************************")
    print()
    print("turbulence array is of length ", len(turbulence_array))
    print(list(turbulence_array))
    print("************************************************")
    print()
    #########################


env_params = {"price_array": price_array,
              "tech_array": tech_array,
              "turbulence_array": turbulence_array,
              "split_val_array": split_array,
              "if_train": True}
env = StockTradingEnv(env_params)

# # remove this block after testing from
# current_state = env.reset()
# print("shape of the state ", current_state.shape)
# print("number of tech indicators ", tech_array.shape[1])
# #remove this block after testing to


config.comments_bool = False
config.training_phase = False
if config.training_phase:

    print("inside the training process")
    train(
            start_date=config.TRAIN_START_DATE,
            end_date=config.TRAIN_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="sac",
            cwd="./multi_stock_model_sac",
            agent_params=config.SAC_PARAMS,
            total_timesteps=1e4,
        )

    train(
            start_date=config.TRAIN_START_DATE,
            end_date=config.TRAIN_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="td3",
            cwd="./multi_stock_model_TD3",
            agent_params=config.TD3_PARAMS,
            total_timesteps=1e4,
        )

    train(
            start_date=config.TRAIN_START_DATE,
            end_date=config.TRAIN_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ppo",
            cwd="./multi_stock_model_ppo",
            agent_params=config.PPO_PARAMS,
            total_timesteps=1e4,
        )

    train(
            start_date=config.TRAIN_START_DATE,
            end_date=config.TRAIN_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ddpg",
            cwd="./multi_stock_model_ddpg",
            agent_params=config.DDPG_PARAMS,
            total_timesteps=1e4,
        )



else:

    print("inside testing process")
    config.comments_bool = False
    account_value_history_sac = test(
            start_date=config.TEST_START_DATE,
            end_date=config.TEST_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="sac",
            cwd="./multi_stock_model_sac.zip",
        )


    account_value_history_td3 = test(
            start_date=config.TEST_START_DATE,
            end_date=config.TEST_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="td3",
            cwd="./multi_stock_model_TD3.zip",
        )


    account_value_history_ppo = test(
            start_date=config.TEST_START_DATE,
            end_date=config.TEST_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ppo",
            cwd="./multi_stock_model_ppo.zip",
        )


    account_value_history_ddpg = test(
            start_date=config.TEST_START_DATE,
            end_date=config.TEST_END_DATE,
            ticker_list=config.TICKERS_LIST,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ddpg",
            cwd="./multi_stock_model_ddpg.zip",
        )

    account_value_history_sac = [round(val, 2) for val in account_value_history_sac]
    account_value_history_td3 = [round(val, 2) for val in account_value_history_td3]
    account_value_history_ppo = [round(val, 2) for val in account_value_history_ppo]
    account_value_history_ddpg = [round(val, 2) for val in account_value_history_ddpg]
    print("***************************************************************")
    print(account_value_history_sac)
    print()
    print(account_value_history_td3)
    print()
    print(account_value_history_ppo)
    print()
    print(account_value_history_ddpg)
    print()
    print("****************************************************************")

    print()
    print()
    action_history_df = pd.DataFrame(config.action_info)
    action_history_df.to_csv('action_history.csv')

    df = pd.DataFrame(list(zip(account_value_history_sac, account_value_history_td3, account_value_history_ppo, account_value_history_ddpg)),
                   columns=['sac', 'td3', 'ppo', 'ddpg'])

    returns_sac = pd.DataFrame(list(zip(config.date_list, account_value_history_sac)), columns=['date', 'account_value'])

    # returns_sac.sort_values(by=['date'], ascending=True)
    print(returns_sac.columns)
    print(returns_sac.dtypes)
    print(returns_sac.head(10))
    returns_sac.astype({'account_value': 'float64'})
    print(returns_sac.dtypes)
    try:
        df.to_csv('portfolio_history.csv')
    except:
        pass
    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=returns_sac)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv('perf_stats.csv')

    #baseline stats
    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(
            ticker="^DJI",
            start = returns_sac.loc[0,'date'],
            end = returns_sac.loc[len(returns_sac)-1,'date'])

    stats = backtest_stats(baseline_df, value_col_name = 'close')
    print(stats)
    print(returns_sac.loc[0,'date'])
    print(returns_sac.loc[len(returns_sac)-1,'date'])
    backtest_plot(returns_sac,
                 baseline_ticker = '^NSEI',
                 baseline_start = returns_sac.loc[0,'date'],
                 baseline_end = returns_sac.loc[len(returns_sac)-1,'date'])
