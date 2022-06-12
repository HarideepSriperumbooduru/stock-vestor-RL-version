import ray

import config
import pandas as pd
from config import (
    TICKERS_LIST,
    TECHNICAL_INDICATORS_LIST,
    TEST_END_DATE,
    TEST_START_DATE,
    # RLlib_PARAMS,
)
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


def test(
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        drl_lib,
        # env,
        model_name,
        if_vix=True,
        **kwargs
):
    from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
    from DataProcessor import DataProcessor

    # fetch data
    config.current_model_name = model_name
    DP = DataProcessor(data_source)
    '''
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    # 

    if if_vix:
        data = DP.add_vix(data)
    data = DP.add_stock_split(data)
    config.test_stock_split_vals = list(data['split_val'].values)
    '''
    test_data = pd.read_csv('testing_data.csv')
    config.date_list = list(test_data['time'].values)
    data = test_data[(test_data['time'] > start_date) & (test_data['time'] < end_date)]
    price_array, tech_array, turbulence_array, split_array, txn_dates_array = DP.df_to_array(data, if_vix)
    config.txn_date_lst = txn_dates_array
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print(txn_dates_array)
    config.test_price_list = list(price_array)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "split_val_array": split_array,
        "if_train": False,
    }
    env_instance = StockTradingEnv(config_params=env_config)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("cwd is :", cwd)
    if config.comments_bool:
        print("cwd during the testing is ", cwd)

    if drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for stable baselines3
    account_value_sb3 = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=TICKERS_LIST,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="stable_baselines3",
        env=env,
        model_name="sac",
        cwd="./test_sac.zip",
    )
