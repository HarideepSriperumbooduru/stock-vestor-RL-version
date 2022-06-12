import gym
import numpy as np
from numpy import random as rd
import config
import json
class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config_params,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=30,
        initial_capital=1e7,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2 ** -11,
        initial_stocks=None,
        txn_date=None,
    ):
        price_ary = config_params["price_array"]
        tech_ary = config_params["tech_array"]
        turbulence_ary = config_params["turbulence_array"]
        split_ary = config_params["split_val_array"]
        if_train = config_params["if_train"]
        config.training_phase = config_params["if_train"]

        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary
        self.split_ary = split_ary
        self.tech_ary = self.tech_ary * 2 ** -7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2 ** -5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0
        self.avg_price_array = []
        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
    def resume(self):

        return None

    def reset(self):
        self.day = 0
        # print(self.price_ary)
        price = self.price_ary[self.day]
        self.avg_price_array = price
        if config.comments_bool:
            print("price array of day 0 is ", price)

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(10, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            if config.comments_bool:
                print("number of stocks at the beginning are ", self.stocks)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
            if config.comments_bool:
                print("initial amount with us is ", self.amount)
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            if config.comments_bool:
                print("initial stocks ", self.stocks)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital
            if config.comments_bool:
                print("initial amount ", self.amount)
        self.total_asset = self.amount + (self.stocks * price).sum()
        if config.comments_bool:
            print("initial total protfolio value ", self.total_asset)
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        current_state = self.get_state(price)
        if config.comments_bool:
            print("current state is ", current_state)
        return current_state  # state

    def live_reset(self):

        # retrieve the json file
        try:
            f = open('mydata.json')
            status = json.load(f)
        except:
            print("json file not found")

        self.day = 0

        price = self.price_ary[self.day]
        if config.comments_bool:
            print("price array of day 0 is ", price)

        self.stocks = status['stocks']
        if config.comments_bool:
            print("initial stocks ", self.stocks)
        self.stocks_cool_down = status['stocks_cool_down']
        self.amount = status['amount']
        if config.comments_bool:
            print("initial amount ", self.amount)
        self.total_asset = self.amount + (self.stocks * price).sum()
        if config.comments_bool:
            print("initial total protfolio value ", self.total_asset)
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        current_state = self.get_state(price)
        if config.comments_bool:
            print("current state is ", current_state)
        return current_state  # state


    def step(self, actions):

        step_status = {}
        # if config.comments_bool:
            # comment out later from
            # print("Following are the actions suggested by SAC algo before processing ***********")
            # print(type(actions))
            # print(actions)
            # print('***********************************************************')
            # comment out later to

        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(actions)
        actions = (actions * self.max_stock).astype(int)


        if config.comments_bool:
            # comment out later from
            # print("#############################################################################################")
            print()
            print("Following are the actions predicted by SAC agent in raw form")
            print(actions)
            # print('***********************************************************')
            # comment out later to

        self.day += 1
        if not config.model_alive:
            txn_date_lst = config.txn_date_lst[self.day]
        price = self.price_ary[self.day]
        # if config.comments_bool:
        #     # comment out later from
        #     print(f"price row for the current day {self.day} ***********")
        #     print(price)
        #     print('***********************************************************')
        #     # comment out later to

        # self.stocks_cool_down += 1

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            # min_action = 2
            # if config.comments_bool:
            #     # comment out later from
            #     print("min action ***********")
            #     print(type(min_action))
            #     print(min_action)
            #     print('***********************************************************')
            #     # comment out later to

            # if config.comments_bool:
            #     print()
            #     print("The following are the actions predicted by the agent to perform on the stock environment")
            #     print()
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    if config.comments_bool:
                        print("sell_num_shares is min of ", self.stocks[index], " and ", -actions[index])
                    sell_num_shares = min(self.stocks[index], -(actions[index]))

                    if not self.if_train and sell_num_shares > 0:
                        config.price_values.append(price[index])
                        config.action_taken.append(0)
                        config.action_days.append(self.day)

                    self.stocks[index] -= sell_num_shares
                    amount_to_be_added = price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    amount_with_avg_price = self.avg_price_array[index] * sell_num_shares
                    self.amount += amount_to_be_added

                    if not self.if_train:
                        total_asset_val = self.amount + (self.stocks * price).sum()

                        if not config.model_alive:
                            self.populate_action_history(txn_date_lst[index], config.TICKERS_LIST[index], config.sector[config.TICKERS_LIST[index]], round(price[index], 2), self.avg_price_array[index], config.current_model_name, 'sell', sell_num_shares, round(amount_to_be_added, 2),
                                                         round(self.amount, 2), total_asset_val, (amount_to_be_added - amount_with_avg_price),
                                                         self.stocks[index])
                    if config.model_alive:
                        if 'action' not in step_status:
                            step_status['action'] = []
                        step_status['action'].append('sell')

                        if 'tic' not in step_status:
                            step_status['tic'] = []
                        step_status['tic'].append(config.TICKERS_LIST[index])

                        if 'num_of_shares' not in step_status:
                            step_status['num_of_shares'] = []
                        step_status['num_of_shares'].append(str(sell_num_shares))

                        if 'price' not in step_status:
                            step_status['price'] = []
                        step_status['price'].append(str(round(price[index], 2)))
                    if config.comments_bool:

                        print(f"Action predicted : sell {sell_num_shares} number of {config.TICKERS_LIST[index]} shares at {round(price[index], 2)}")
                        print(f"total number of {config.TICKERS_LIST[index]} shares after the current action is {self.stocks[index]}")

                        print(
                            f"amount gained on selling {sell_num_shares} number of {config.TICKERS_LIST[index]} shares is ** {round(amount_to_be_added, 2)} **")
                        print("balance left after the current action is ", round(self.amount, 2))
                        print()
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                    if config.comments_bool:
                        print("buy_num_shares is min of  ", self.amount // price[index], " and ", actions[index])
                    buy_num_shares = min(self.amount // price[index], actions[index])
                    if not self.if_train and buy_num_shares > 0:
                        config.price_values.append(price[index])
                        config.action_taken.append(1)
                        config.action_days.append(self.day)
                    self.stocks[index] += buy_num_shares
                    amount_to_be_deducted = price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.amount -= amount_to_be_deducted

                    if not self.if_train:
                        total_asset_val = self.amount + (self.stocks * price).sum()
                        if not config.model_alive:
                            if price[index] and buy_num_shares and self.avg_price_array[index] and self.stocks[index]:
                                if buy_num_shares > 0 or self.stocks[index] > 0:
                                    wt_avg_price_buy = self.weighted_avg(price[index], buy_num_shares, self.avg_price_array[index],
                                                                     self.stocks[index])
                                    self.avg_price_array[index] = wt_avg_price_buy
                            self.populate_action_history(txn_date_lst[index], config.TICKERS_LIST[index], config.sector[config.TICKERS_LIST[index]],round(price[index], 2), self.avg_price_array[index], config.current_model_name, 'buy',
                                                         buy_num_shares, round(amount_to_be_deducted, 2),
                                                         round(self.amount, 2), total_asset_val, 0,
                                                         self.stocks[index])
                    if config.model_alive:
                        if 'action' not in step_status:
                            step_status['action'] = []
                        step_status['action'].append('buy')

                        if 'tic' not in step_status:
                            step_status['tic'] = []
                        step_status['tic'].append(config.TICKERS_LIST[index])

                        if 'num_of_shares' not in step_status:
                            step_status['num_of_shares'] = []
                        step_status['num_of_shares'].append(str(buy_num_shares))

                        if 'price' not in step_status:
                            step_status['price'] = []
                        step_status['price'].append(str(round(price[index], 2)))

                    if config.comments_bool:

                        print(f"Action predicted : buy {buy_num_shares} number of {config.TICKERS_LIST[index]} shares at {price[index]}")
                        print(
                            f"total number of {config.TICKERS_LIST[index]} shares after the current action is {self.stocks[index]}")

                        print(f"amount spent on buying {buy_num_shares} number of {config.TICKERS_LIST[index]} shares is ** {amount_to_be_deducted} **")
                        print("balance left after the current action of buying is ", self.amount)
                        print()
                        # print("#######################################################################################")
                        # print()
                    self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence

            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            if not self.if_train:
                for val in price:
                    config.price_values.append(val)
                    config.action_taken.append(0)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset


        # handling stock split by increasing the amount of shares of a company by split ratio
        n = len(config.TICKERS_LIST)

        split_value = self.split_ary[self.day]
        for i, tic in enumerate(config.TICKERS_LIST):
            if split_value[i] > 1:
                # print("###################  day is ", self.day, " and the split val is  ", split_value[i], ' and tic is ', config.TICKERS_LIST[i], ' and the price is ', price[i])
                self.stocks[i] = int(self.stocks[i] * split_value[i])

        return state, reward, done, step_status

    # populating action history

    def populate_action_history(self, txn_date, tic, sector, share_price, avg_share_price, model_name, action, no_of_shares_for_action, amount_for_action, balance_left, total_asset_value, return_val, total_no_of_shares):

        if 'transaction_date' not in config.action_info:
            config.action_info['transaction_date'] = []
        config.action_info['transaction_date'].append(txn_date)
        if 'tic' not in config.action_info:
            config.action_info['tic'] = []
        config.action_info['tic'].append(tic)
        if 'sector' not in config.action_info:
            config.action_info['sector'] = []
        config.action_info['sector'].append(sector)

        if 'share_price' not in config.action_info:
            config.action_info['share_price'] = []
        config.action_info['share_price'].append(share_price)

        if 'avg_share_price' not in config.action_info:
            config.action_info['avg_share_price'] = []
        config.action_info['avg_share_price'].append(avg_share_price)

        if 'model_name' not in config.action_info:
            config.action_info['model_name'] = []
        config.action_info['model_name'].append(model_name)
        if 'action' not in config.action_info:
            config.action_info['action'] = []
        config.action_info['action'].append(action)
        if 'no_of_shares_for_action' not in config.action_info:
            config.action_info['no_of_shares_for_action'] = []
        config.action_info['no_of_shares_for_action'].append(
            no_of_shares_for_action)
        if 'amount_for_action' not in config.action_info:
            config.action_info['amount_for_action'] = []
        config.action_info['amount_for_action'].append(
            amount_for_action)
        if 'balance_left' not in config.action_info:
            config.action_info['balance_left'] = []
        config.action_info['balance_left'].append(balance_left)
        if 'total_asset_value' not in config.action_info:
            config.action_info['total_asset_value'] = []
        config.action_info['total_asset_value'].append(total_asset_value)
        if 'return_val' not in config.action_info:
            config.action_info['return_val'] = []
        config.action_info['return_val'].append(return_val)
        if 'total_no_of_shares' not in config.action_info:
            config.action_info['total_no_of_shares'] = []
        config.action_info['total_no_of_shares'].append(total_no_of_shares)

    def get_state(self, price):
        # if config.comments_bool:
        #     print("in get state")
        #     print("initial stocks cool down is ", self.stocks_cool_down)
        #     print("amount before scaling is ", self.amount)
        amount = np.array(self.amount * (2 ** -12), dtype=np.float32)
        # if config.comments_bool:
        #     print("amount after scaling is ", amount)
        scale = np.array(2 ** -6, dtype=np.float32)
        # if config.comments_bool:
        #     print("scaling factor ", scale)
        return np.hstack(
            (
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    def weighted_avg(self, current_price, num_of_stocks_in_transaction, avg_price, existing_stocks_number):
        x = (avg_price * existing_stocks_number)
        y = (current_price * num_of_stocks_in_transaction)
        z = num_of_stocks_in_transaction + existing_stocks_number

        try:
            wt_avg_price = (x + y) / z
        except:
            print(x , "  ", y, "  ", z)
        return round(wt_avg_price, 2)
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
