
DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"
SAVED_RESULTS = 'saved_results'
DIR_LIST = [RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR, DATA_SAVE_DIR, SAVED_RESULTS]
# last 6 months kind of input
previous_total_asset_val = 0
training_phase = False
comments_bool = False
date_list = []
START_DATE = "2010-02-17"
END_DATE = "2022-03-14"

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'

TEMP_START_DATE = '2019-02-18'
TEMP_DATE = '2021-04-21'
TEST_START_DATE = '2020-07-01'
TEST_END_DATE = '2022-04-20'
txn_date_lst = []

action_info = {}
# action_info = {'transaction_date':[], 'tic': [], 'share_price': [], 'model_name': [], 'action': [], 'no_of_shares_for_action': [], 'amount_for_action': [], 'balance_left': [], 'total_asset_value': [], 'return_ratio': [], 'total_no_of_shares':[]}
current_model_name = ''
model_alive = True
# TRAIN_START_DATE = "2022-02-18"
# TRAIN_END_DATE = "2022-02-28"
#
# TEST_START_DATE = "2022-03-01"
# TEST_END_DATE = "2022-03-10"

START_TRADE_DATE = "2020-02-18"

# list of top 3 large cap companies in Technology, banking, consumer, and pharma sectors
# 'SBILIFE.NS', 'HDFCLIFE.NS', 'COALINDIA.NS'
T_LIST = ['TCS.NS', 'INFY.NS','IOC.NS','HINDALCO.NS', 'WIPRO.NS','ASIANPAINT.NS','BAJAJFINSV.NS','BAJFINANCE.NS','AXISBANK.NS', 'BAJAJ-AUTO.NS','BHARTIARTL.NS', 'GRASIM.NS','EICHERMOT.NS','DRREDDY.NS', 'HCLTECH.NS', 'HDFC.NS', 'MARUTI.NS','M&M.NS','LT.NS','INDUSINDBK.NS','HEROMOTOCO.NS', 'KOTAKBANK.NS', 'JSWSTEEL.NS','TECHM.NS','TATASTEEL.NS', 'SHREECEM.NS', 'RELIANCE.NS', 'POWERGRID.NS', 'ONGC.NS', 'NTPC.NS', 'TATAMOTORS.NS','TATACONSUM.NS', 'HDFCBANK.NS', 'SBIN.NS', 'ICICIBANK.NS', 'ITC.NS', 'UPL.NS','ULTRACEMCO.NS','TITAN.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'SUNPHARMA.NS', 'LUPIN.NS', 'CIPLA.NS']
# T_LIST = ['TCS.NS', 'INFY.NS']
TICKERS_LIST = sorted(T_LIST)
sector = {'TCS.NS': 'Technology', 'INFY.NS': 'Technology', 'WIPRO.NS': 'Technology', 'TECHM.NS': 'Technology', 'HCLTECH.NS': 'Technology',
          'HDFCBANK.NS': 'Banking', 'SBIN.NS': 'Banking', 'ICICIBANK.NS': 'Banking', 'KOTAKBANK.NS': 'Banking','INDUSINDBK.NS':'Banking', 'AXISBANK.NS': 'Banking',
          'ITC.NS': 'Consumer','TATACONSUM.NS': 'Consumer', 'HINDUNILVR.NS': 'Consumer', 'NESTLEIND.NS': 'Consumer', 'UPL.NS': 'Consumer', 'ULTRACEMCO.NS': 'Consumer', 'TITAN.NS': 'Consumer',
          'SUNPHARMA.NS': 'Pharma', 'LUPIN.NS': 'Pharma', 'CIPLA.NS': 'Pharma', 'DRREDDY.NS': 'Pharma',
          'TATASTEEL.NS': 'Manufacturing', 'SHREECEM.NS': 'Manufacturing', 'JSWSTEEL.NS': 'Manufacturing', 'HINDALCO.NS': 'Manufacturing',
          'TATAMOTORS.NS': 'Automobile', 'MARUTI.NS': 'Automobile', 'M&M.NS': 'Automobile', 'HEROMOTOCO.NS': 'Automobile', 'EICHERMOT.NS': 'Automobile', 'BAJAJ-AUTO.NS': 'Automobile',
          'SBILIFE.NS': 'Insurance', 'HDFCLIFE.NS': 'Insurance',
          'RELIANCE.NS': 'Energy', 'POWERGRID.NS': 'Energy', 'ONGC.NS': 'Energy', 'NTPC.NS': 'Energy',
          'LT.NS': 'Engineering',
          'IOC.NS': 'Oil & Gas',
          'HDFC.NS': 'Finance', 'BAJAJFINSV.NS': 'Finance', 'BAJFINANCE.NS': 'Finance',
          'GRASIM.NS': 'Textile',
          'COALINDIA.NS': 'Mining',
          'BHARTIARTL.NS': 'Telecom',
          'ASIANPAINT.NS': 'Paints'}

# T_LIST = ['TCS.NS', 'SBIN.NS']
# TICKERS_LIST = sorted(T_LIST)
TEST_TICKERS_LIST = ['INFY.NS']


price_values = []
action_taken = []
action_days = []
test_price_list = []
train_stock_split_vals = []
test_stock_split_vals = []

TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Model Parameters
# SAC_PARAMS = {
#     "batch_size": 64,
#     "buffer_size": 100000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1",
# }
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}


A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
# PPO_PARAMS = {
#     "n_steps": 2048,
#     "ent_coef": 0.01,
#     "learning_rate": 0.00025,
#     "batch_size": 64,
# }
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
# TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100,
              "buffer_size": 1000000,
              "learning_rate": 0.001}

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}

########################################################
#######Possible time zones##############################
TIME_ZONE_SHANGHAI = 'Asia/Shanghai'  ## Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = 'US/Eastern'  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = 'Europe/Paris'  # CAC,
TIME_ZONE_BERLIN = 'Europe/Berlin'  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = 'Asia/Jakarta'  # LQ45
TIME_ZONE_SELFDEFINED = 'xxx'  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

