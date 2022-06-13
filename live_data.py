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

def test_model(model_name, MODELS, cwd, data_df, env_kwargs):
    file_exists = os.path.exists('mydata.json')
    if not file_exists:
        status = {}
    else:
        f = open('mydata.json')
        status = json.load(f)
        env_kwargs["initial_amount"] = status['amount']
        env_kwargs["num_stock_shares"] = [int(p) for p in status['stocks']]

    config.current_model_name = model_name
    model = load_model(model_name, MODELS, cwd)
    e_trade_gym = StockTradingEnv(df=data_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    print('amount before today is ', e_trade_gym.state[0])
    df_account_value, df_actions, step_status = DRLAgent.DRL_prediction(
        model=model,
        environment=e_trade_gym, status_bool=True)


    # print("current step status is ")
    # print(step_status[0])
    action_history_df = pd.DataFrame.from_dict(step_status[0])
    file_exists = os.path.exists('action_history_live.csv')
    if file_exists:
        # df_live = pd.read_csv('action_history_live.csv')
        # frames = [df_live, action_history_df]
        # final_live_df = pd.concat(frames, ignore_index=True)
        # final_live_df.to_csv("action_history_live.csv")
        action_history_df.to_csv('action_history_live.csv', mode='a', index=False, header=False)
    else:
        action_history_df.to_csv("action_history_live.csv", index=False)


    balance_left = list(action_history_df.balance.values)[-1]
    print('final amount is ', balance_left)

    action_history_df = action_history_df.sort_values(by='tic')
    # print('sorted tics are')
    # print(list(action_history_df.tic.values))
    # print()
    # print('sorted share numbers')
    # print(list(action_history_df.total_num_shares.values))
    # print()

    stocks = list(action_history_df.total_num_shares.values)


    if 'stocks' not in status:
        status['stocks'] = None
    status['stocks'] = [str(k) for k in stocks]
    # if 'stocks_cool_down' not in status:
    #     status['stocks_cool_down'] = None
    # status['stocks_cool_down'] = [int(p) for p in env_instance.stocks_cool_down]

    if 'amount' not in status:
        status['amount'] = None
    status['amount'] = balance_left


    # current_date = date.today()
    # current_date = current_date.strftime("%Y-%m-%d")
    # if current_date not in status:
    #     status[current_date] = []
    # status[current_date].append(step_status)
    # status.update(step_status)

    # print(status)
    with open('mydata.json', 'w') as f:
        json.dump(status, f)





config.model_alive = True
bse_calendar = tc.get_calendar("XBOM")
today = date.today()

# dd/mm/YY
start = today.strftime("%Y-%m-%d")
# start = '2022-06-08'
start_d = "2020-06-07"
if bse_calendar.is_session(start):
    last_working_date = bse_calendar.next_session(start)
    end = last_working_date.strftime("%Y-%m-%d")
    print(start)
    print(end)

    prev_day_date = bse_calendar.previous_session(start)
    prev_day_date = prev_day_date.strftime("%Y-%m-%d")
    # prev_day_date = '2022-06-07'
    print("previous day date is ", prev_day_date)
    DP = DataProcessor(data_source='yahoofinance')
    df = DP.download_data(start_date=start_d,
                            end_date=end,
                            ticker_list=config.TICKERS_LIST,
                            time_interval='1d')



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
    processed_full.to_csv('recent_data.csv')
    data = data_split(processed_full, prev_day_date, end)

    print("printing the data set for today ", len(data.tic.unique()))
    data = data.sort_values(by='date', ascending=False)

    print(data)

    model_name = 'a2c'
    cwd="./trained_models/a2c_5.pth"
    MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

    stock_dimension = len(data.tic.unique())

    tickers = list(data.tic.values)
    for item in config.TICKERS_LIST:
        if item not in tickers:
            print('missing ticker is ', item)
    state_space = 1 + 2 * stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    num_stock_shares = (np.array(num_stock_shares) + rd.randint(10, 30, size=np.array(num_stock_shares).shape)
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

    test_model(model_name, MODELS, cwd, data, env_kwargs)
    # get_portfolio_state(price_array, tech_array, turbulence_array, split_array, model_name, cwd )

    # curr_state = self.get_state(price_array, tech_array, turbulence_array)
    # print(curr_state)

else:
    print("today is not a working day for the exchange")
