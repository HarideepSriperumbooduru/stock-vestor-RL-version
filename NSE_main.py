import os

import pandas as pd
import numpy as np
from numpy import random as rd
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime


from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from DataProcessor import DataProcessor
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

import sys
sys.path.append("../FinRL-Library")
import config
import itertools


# "./" will be added in front of each directory
def check_and_make_directories(directories):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

def download_initial_dataset():
    file_exists = os.path.exists('processed_full.csv')
    if not file_exists:
        print("downloading total data")
        DP = DataProcessor(data_source='yahoofinance')
        df = DP.download_data(start_date=config.TRAIN_START_DATE,
                                end_date=config.TEST_END_DATE,
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

            processed_full.to_csv('processed_full.csv')
        except:
            pass
        if config.comments_bool:
            # comment this block later
            print("initial data after adding tech indicators and processing")
            print(processed_full.head(50))
            print("************************************************")
            #########################

def load_model(model_name, MODELS, cwd):
    if model_name not in MODELS:
        raise NotImplementedError("NotImplementedError")
    try:
        # load agent
        model = MODELS[model_name].load(cwd)
        print("Successfully load model", cwd)
    except BaseException:
        raise ValueError("Fail to load agent!")

    return model

def set_model(model_name, params,total_timesteps, train_df, env_kwargs):

    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)


    to_be_trained_model = agent.get_model(model_name, model_kwargs=params)
    trained_model = agent.train_model(model=to_be_trained_model,
                                    tb_log_name=model_name,
                                    total_timesteps=total_timesteps)


def test_model(model_name, MODELS, cwd, data_df, env_kwargs):
    config.current_model_name = model_name
    model = load_model(model_name, MODELS, cwd)
    e_trade_gym = StockTradingEnv(df=data_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model,
        environment=e_trade_gym)

    if config.comments_bool:
        print("shape of the df_account value is ", df_account_value.shape)

        print(df_account_value.tail())
        print(df_actions.head())
    try:
        df_actions.to_csv(f"./saved_results/actions_history_{model_name}.csv")
        df_account_value.to_csv(f"./saved_results/account_history_{model_name}.csv")
    except:
        pass


def backtest_model(df_account_value):
    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + '.csv')

    # baseline stats
    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(
        ticker="^NSEI",
        start=df_account_value.loc[0, 'date'],
        end=df_account_value.loc[len(df_account_value) - 1, 'date'])

    stats = backtest_stats(baseline_df, value_col_name='close')

    print("==============Compare to DJIA===========")

    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value,
                  baseline_ticker='^NSEI',
                  baseline_start=df_account_value.loc[0, 'date'],
                  baseline_end=df_account_value.loc[len(df_account_value) - 1, 'date'])


def main():

    MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
    check_and_make_directories(config.DIR_LIST)
    download_initial_dataset()
    processed_full = pd.read_csv('processed_full.csv')
    train = data_split(processed_full, '2009-01-01', '2020-07-01')
    trade = data_split(processed_full, '2020-07-01', '2022-04-20')
    config.txn_date_lst = train['date']
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    num_stock_shares = (np.array(num_stock_shares) + rd.randint(10, 64, size=np.array(num_stock_shares).shape)
            ).astype(np.int32)

    env_kwargs = {
        "hmax": 5,
        "initial_amount": 10000000,
        "num_stock_shares": list(num_stock_shares),
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    if config.training_phase:
        print("in training phase")
        # e_train_gym = StockTradingEnv(df=train, **env_kwargs)
        #
        # env_train, _ = e_train_gym.get_sb_env()
        # agent = DRLAgent(env = env_train)
        # SAC_PARAMS = {
        #     "batch_size": 128,
        #     "buffer_size": 1000000,
        #     "learning_rate": 0.0001,
        #     "learning_starts": 100,
        #     "ent_coef": "auto_0.1",
        # }
        #
        # model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
        # trained_sac = agent.train_model(model=model_sac,
        #                              tb_log_name='sac',
        #                              total_timesteps=100000)




        # uncomment later

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

    else:
        num_stock_shares = [0] * stock_dimension
        num_stock_shares = (np.array(num_stock_shares) + rd.randint(10, 64, size=np.array(num_stock_shares).shape)
                            ).astype(np.int32)
        env_kwargs = {
            "hmax": 5,
            "initial_amount": 10000000,
            "num_stock_shares": list(num_stock_shares),
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }
        config.txn_date_lst = trade['date']
        print("in validating phase")
        # model = load_model('sac', MODELS, './trained_models/SAC_full_data.zip')
        # e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
        #
        # df_account_value, df_actions = DRLAgent.DRL_prediction(
        #     model=model,
        #     environment=e_trade_gym)
        #
        # print("shape of the df_account value is ", df_account_value.shape)
        #
        # print(df_account_value.tail())
        # print(df_actions.head())

        test_model(
            model_name='sac',
            MODELS=MODELS,
            cwd='./trained_models/SAC_full_data.zip',
            data_df=trade,
            env_kwargs=env_kwargs
        )

        test_model(
            model_name='ppo',
            MODELS=MODELS,
            cwd='./trained_models/ppo_3.pth',
            data_df=trade,
            env_kwargs=env_kwargs
        )

        test_model(
            model_name='ddpg',
            MODELS=MODELS,
            cwd='./trained_models/ddpg_3.pth',
            data_df=trade,
            env_kwargs=env_kwargs
        )

        test_model(
            model_name='td3',
            MODELS=MODELS,
            cwd='./trained_models/td3_6.pth',
            data_df=trade,
            env_kwargs=env_kwargs
        )
        test_model(
            model_name='a2c',
            MODELS=MODELS,
            cwd='./trained_models/a2c_5.pth',
            data_df=trade,
            env_kwargs=env_kwargs
        )
    try:
        action_history_df = pd.DataFrame(config.action_info)
        action_history_df.to_csv('./saved_results/portfolio_history.csv')
    except:
        pass


if __name__ == "__main__":
    main()


