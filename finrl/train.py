import config
from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from DataProcessor import DataProcessor
import pandas as pd
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv



def train(
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
    # fetch data
    config.current_model_name = model_name
    DP = DataProcessor(data_source)
    # data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    # data = DP.clean_data(data)
    # data = DP.add_technical_indicator(data, technical_indicator_list)
    # if if_vix:
    #     data = DP.add_vix(data)
    #
    # data = DP.add_stock_split(data)
    train_data = pd.read_csv('training_data.csv')
    data = train_data[(train_data['time'] > start_date) & (train_data['time'] < end_date)]
    price_array, tech_array, turbulence_array, split_array, txn_dates_array = DP.df_to_array(data, if_vix)
    config.txn_date_lst = train_data['time']
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "split_val_array": split_array,
        "if_train": True,
    }
    env_instance = StockTradingEnv(config_params=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))


    if drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")

        agent = DRLAgent_sb3(env=env_instance)

        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training finished!")
        trained_model.save(cwd)
        print("Trained model saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")



