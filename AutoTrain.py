import os
from numpy import random as rd
import numpy as np
import pandas as pd
import json
import exchange_calendars as tc
import config
from DataProcessor import DataProcessor
# from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from datetime import date
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from datetime import timedelta
import yfinance as yf
import itertools
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3


def download_initial_dataset():
    file_exists = os.path.exists('processed_full_auto_train.csv')
    if not file_exists:
        print("downloading total data")
        DP = DataProcessor(data_source='yahoofinance')
        df = DP.download_data(start_date=config.TRAIN_START_DATE,
                                end_date=config.TRAIN_END_DATE,
                                ticker_list=config.TICKERS_LIST,
                                time_interval='1d')
        # df.sort_values(['date', 'tic'], ignore_index=True).head()
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False)
        processed = fe.preprocess_data(df)
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])

        processed_full = processed_full.fillna(0)
        # processed_full = DP.add_stock_split(processed_full)

        try:

            processed_full.to_csv('processed_full_auto_train.csv')
        except:
            pass
        if config.comments_bool:
            # comment this block later
            print("initial data after adding tech indicators and processing")
            print(processed_full.head(50))
            print("************************************************")
            #########################

def set_model(model_name, params,total_timesteps, train_df, env_kwargs):

    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)


    to_be_trained_model = agent.get_model(model_name, model_kwargs=params)
    trained_model = agent.train_model(model=to_be_trained_model,
                                    tb_log_name=model_name,
                                    total_timesteps=total_timesteps, extension = 'auto_trained')


def main():

    tickers_list = []
    print('Enter the ticker number to add it to your tickers list')
    count = 0
    for item in config.TICKERS_LIST:
        print(count, " ", item.replace('.NS', ''))
        count += 1
    while True:
        option = int(input(" 1. add ticker  ;  2. continue : "))
        if option == 1:
            ticker_num = int(input("Enter the index of the ticker to add : "))
            tickers_list.append(config.TICKERS_LIST[ticker_num])
        else:
            break

    tickers_list = sorted(tickers_list)
    h_max = int(input("enter the max number of shares to sell/buy in a single transaction : "))

    num_shares = [0]*len(tickers_list)
    num_shares_option = int(input(" 1. enter number of existing shares of tickers  ;  2. continue  :"))

    if num_shares_option == 1:

        for i in range(len(tickers_list)):
            print("you are currently working on ", tickers_list[i])
            edit_option = int(input(" 1. update number of shares  ; 2. continue  : "))
            if edit_option == 1:
                n_shares = int(input("enter num of shares : "))
                num_shares[i] = n_shares
            else:
                pass

    initial_amount = int(input("enter the initial amount to start training the stock model : "))
    print("enter training start and end dates in yyyy-mm-dd format")
    train_start_date = input("enter training start date : ")
    train_end_date = input("enter training end date : ")

    config.TRAIN_START_DATE = train_start_date
    config.TRAIN_END_DATE = train_end_date
    config.TICKERS_LIST = tickers_list
    config.training_phase =True
    config.model_alive = False


    MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

    download_initial_dataset()
    processed_full = pd.read_csv('processed_full_auto_train.csv')
    train = data_split(processed_full, config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    config.txn_date_lst = train['date']
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    # num_stock_shares = [0] * stock_dimension
    # num_stock_shares = (np.array(num_stock_shares) + rd.randint(10, 64, size=np.array(num_stock_shares).shape)
    #         ).astype(np.int32)

    env_kwargs = {
        "hmax": h_max,
        "initial_amount": initial_amount,
        "num_stock_shares": num_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }


    print("in training phase")

    set_model(
        model_name='sac',
        params=config.SAC_PARAMS,
        total_timesteps=1e5,
        train_df=train,
        env_kwargs=env_kwargs
    )

    set_model(
        model_name='td3',
        params=config.TD3_PARAMS,
        total_timesteps=1e5,
        train_df=train,
        env_kwargs=env_kwargs
    )
    set_model(
        model_name='ppo',
        params=config.PPO_PARAMS,
        total_timesteps=1e5,
        train_df=train,
        env_kwargs=env_kwargs
    )
    set_model(
        model_name='ddpg',
        params=config.DDPG_PARAMS,
        total_timesteps=1e5,
        train_df=train,
        env_kwargs=env_kwargs
    )
    set_model(
        model_name='a2c',
        params=config.A2C_PARAMS,
        total_timesteps=1e5,
        train_df=train,
        env_kwargs=env_kwargs
    )


if __name__ == "__main__":
    main()