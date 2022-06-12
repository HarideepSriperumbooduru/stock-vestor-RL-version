import optuna
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import optuna
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
import config
from config import TICKERS_LIST
from optuna.integration import PyTorchLightningPruningCallback
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy
from finrl.drl_agents.stablebaselines3.models import DRLAgent
# from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.finrl_meta.data_processor import DataProcessor
import joblib
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
# import ray
from pprint import pprint
from numpy import random as rd

import os

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
processed_full = pd.read_csv('processed_full.csv')
train = data_split(processed_full, '2009-01-01', '2020-07-01')
trade = data_split(processed_full, '2020-07-01', '2022-04-20')
config.txn_date_lst = train['date']
stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
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
e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))
agent = DRLAgent(env=env_train)

e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=None, **env_kwargs)


def sample_sac_params(trial: optuna.Trial):
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 10000, 20000])

    ent_coef = "auto"


    target_entropy = "auto"
    if ent_coef == 'auto':
        # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
        target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "ent_coef": ent_coef,
        "target_entropy": target_entropy,

    }
    return hyperparams


def calculate_sharpe(df):
    df['daily_return'] = df['account_value'].pct_change(1)
    if df['daily_return'].std() != 0:
        sharpe = (252 ** 0.5) * df['daily_return'].mean() / \
                 df['daily_return'].std()
        return sharpe
    else:
        return 0


class LoggingCallback:
    def __init__(self, threshold, trial_number, patience):
        '''
        threshold:int tolerance for increase in sharpe ratio
        trial_number: int Prune after minimum number of trials
        patience: int patience for the threshold
        '''
        self.threshold = threshold
        self.trial_number = trial_number
        self.patience = patience
        self.cb_list = []  # Trials list for which threshold is reached

    def __call__(self, study: optuna.study, frozen_trial: optuna.Trial):
        # Setting the best value in the current trial
        study.set_user_attr("previous_best_value", study.best_value)

        # Checking if the minimum number of trials have pass
        if frozen_trial.number > self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            # Checking if the previous and current objective values have the same sign
            if previous_best_value * study.best_value >= 0:
                # Checking for the threshold condition
                if abs(previous_best_value - study.best_value) < self.threshold:
                    self.cb_list.append(frozen_trial.number)
                    # If threshold is achieved for the patience amount of time
                    if len(self.cb_list) > self.patience:
                        print('The study stops now...')
                        print('With number', frozen_trial.number, 'and value ', frozen_trial.value)
                        print('The previous and current best values are {} and {} respectively'
                              .format(previous_best_value, study.best_value))
                        study.stop()


from IPython.display import clear_output
import sys

os.makedirs("models",exist_ok=True)

def objective(trial:optuna.Trial):
  #Trial will suggest a set of hyperparamters from the specified range
  hyperparameters = sample_sac_params(trial)
  model_sac = agent.get_model("sac", model_kwargs = hyperparameters )
  #You can increase it for better comparison
  trained_sac = agent.train_model(model=model_sac,
                                  tb_log_name="sac" ,
                             total_timesteps=100000)
  trained_sac.save('models/sac_{}.pth'.format(trial.number))
  clear_output(wait=True)
  #For the given hyperparamters, determine the account value in the trading period
  df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_sac,
    environment = e_trade_gym)
  #Calculate sharpe from the account value
  sharpe = calculate_sharpe(df_account_value)

  return sharpe

#Create a study object and specify the direction as 'maximize'
#As you want to maximize sharpe
#Pruner stops not promising iterations
#Use a pruner, else you will get error related to divergence of model
#You can also use Multivariate samplere
#sampler = optuna.samplers.TPESampler(multivarite=True,seed=42)
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(study_name="sac_study",direction='maximize',
                            sampler = sampler, pruner=optuna.pruners.HyperbandPruner())

logging_callback = LoggingCallback(threshold=1e-5,patience=30,trial_number=5)
#You can increase the n_trials for a better search space scanning
study.optimize(objective, n_trials=30,catch=(ValueError,), callbacks=[logging_callback])

joblib.dump(study, "final_sac_study__.pkl")

print('Hyperparameters after tuning', study.best_params)
print('Hyperparameters before tuning', config.SAC_PARAMS)

print(study.best_trial)

tuned_model_sac = SAC.load('models/sac_{}.pth'.format(study.best_trial.number),env=env_train)
df_account_value_tuned, df_actions_tuned = DRLAgent.DRL_prediction(
    model=tuned_model_sac,
    environment = e_trade_gym)

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all_tuned = backtest_stats(account_value=df_account_value_tuned)
perf_stats_all_tuned = pd.DataFrame(perf_stats_all_tuned)
perf_stats_all_tuned.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_tuned_"+now+'.csv')