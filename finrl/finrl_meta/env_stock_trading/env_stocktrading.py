import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List
matplotlib.use("Agg")
import config
# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: List[int],
        buy_cost_pct: List[float],
        sell_cost_pct: List[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: List[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool =False,
        print_verbosity=10,
        day=00,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares=num_stock_shares
        self.initial_amount = initial_amount # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        self.data = self.df.loc[self.day, :]

        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount+np.sum(np.array(self.num_stock_shares)*np.array(self.state[1:1+self.stock_dim]))] # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        config.previous_total_asset_val = self.asset_memory[0]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory=[] # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action, step_status):
        def _do_sell_normal():
            if self.state[index + 2*self.stock_dim + 1]!=True : # check if the stock is able to sell, for simlicity we just add it in techical index
            # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    total_asset_val_before_action = self.state[0] + sum(
                        np.array(self.state[1: (self.stock_dim + 1)])
                        * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    )

                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            if self.state[index + self.stock_dim + 1] > 0:
                if not config.training_phase:
                    if not config.model_alive:
                        amount_with_avg_price = self.avg_price_array[index] * sell_num_shares
                        total_asset_val = self.state[0] + sum(
                            np.array(self.state[1: (self.stock_dim + 1)])
                            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                        )
                        self.populate_action_history(self.date_memory[-1], self.data.tic.values[index],
                                                     config.sector[self.data.tic.values[index]], round(self.state[index + 1], 2),
                                                     self.avg_price_array[index], config.current_model_name, 'sell',
                                                     sell_num_shares, round(sell_amount, 2),
                                                     round(self.state[0], 2), round(total_asset_val, 2),
                                                     round((sell_amount - amount_with_avg_price), 2),
                                                     round((total_asset_val - config.previous_total_asset_val), 2),
                                                     self.state[index + self.stock_dim + 1])

                        config.previous_total_asset_val = total_asset_val

            if config.model_alive:
                if 'date' not in step_status:
                    step_status['date'] = []
                step_status['date'].append(self.date_memory[-1])
                if 'action' not in step_status:
                    step_status['action'] = []
                step_status['action'].append('sell')

                if 'tic' not in step_status:
                    step_status['tic'] = []
                step_status['tic'].append(self.data.tic.values[index])

                if 'num_of_shares' not in step_status:
                    step_status['num_of_shares'] = []
                step_status['num_of_shares'].append(str(sell_num_shares))

                if 'price' not in step_status:
                    step_status['price'] = []
                step_status['price'].append(str(round(self.state[index + 1], 2)))

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action, step_status):
        def _do_buy():
            if self.state[index + 2*self.stock_dim+ 1] !=True: # check if the stock is able to buy
            # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = int(self.state[0] / (self.state[index + 1]*(1 + self.buy_cost_pct[index])) )# when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                total_asset_val_before_action = self.state[0] + sum(
                    np.array(self.state[1: (self.stock_dim + 1)])
                    * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                )

                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                self.trades += 1
            else:
                buy_num_shares = 0

            if not config.training_phase:
                if not config.model_alive:
                    existing_shares = (self.state[index + self.stock_dim + 1] - buy_num_shares)
                    if self.state[index + 1] and buy_num_shares and self.avg_price_array[index] and existing_shares:
                        if buy_num_shares > 0 or existing_shares > 0:
                            wt_avg_price_buy = self.weighted_avg(self.state[index + 1], buy_num_shares,
                                                                 self.avg_price_array[index],
                                                                 existing_shares)
                            self.avg_price_array[index] = wt_avg_price_buy
                    # amount_with_avg_price = self.avg_price_array[index] * buy_num_shares
                    total_asset_val = self.state[0] + sum(
                        np.array(self.state[1: (self.stock_dim + 1)])
                        * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    )
                    self.populate_action_history(self.date_memory[-1], self.data.tic.values[index],
                                                 config.sector[self.data.tic.values[index]],
                                                 round(self.state[index + 1], 2),
                                                 self.avg_price_array[index], config.current_model_name, 'buy',
                                                 buy_num_shares, round(buy_amount, 2),
                                                 round(self.state[0], 2), round(total_asset_val, 2),
                                                 0,
                                                 round((total_asset_val - config.previous_total_asset_val), 2),
                                                 self.state[index + self.stock_dim + 1])

                    config.previous_total_asset_val = total_asset_val

            if config.model_alive:
                if 'date' not in step_status:
                    step_status['date'] = []
                step_status['date'].append(self.date_memory[-1])
                if 'action' not in step_status:
                    step_status['action'] = []
                step_status['action'].append('buy')

                if 'tic' not in step_status:
                    step_status['tic'] = []
                step_status['tic'].append(self.data.tic.values[index])

                if 'num_of_shares' not in step_status:
                    step_status['num_of_shares'] = []
                step_status['num_of_shares'].append(str(buy_num_shares))

                if 'price' not in step_status:
                    step_status['price'] = []
                step_status['price'].append(str(round(self.state[index + 1], 2)))
            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_trade_{}.png".format(self.episode))
        plt.close()

    def step(self, actions):
        step_status = {}
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal and not config.model_alive:
            print("in the terminal stage ", self.terminal)
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            ) # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            print("**********&&&&&&&& ", df_rewards.shape)
            print(df_rewards)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:
            # print("entered step")
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            # print("********************************")
            # print(actions)
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index], step_status) * -1

                # actions[index] = -1 * sell_shares_count
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index], step_status)

            self.actions_memory.append(actions)

            # state: s -> s+1
            if not config.model_alive:
                self.day += 1

            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print('getting the todays date ', self._get_date())
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            # print("new length of date memory is ", len(self.date_memory))
            self.reward = end_total_asset - begin_total_asset
            # print("reward at the end is ", self.reward)
            self.rewards_memory.append(self.reward)
            # actn_lst = actions + [self.reward]
            # self.actions_memory.append(actn_lst)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state) # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, step_status

    def reset(self):
        print("in the reset method")
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount+np.sum(np.array(self.num_stock_shares)*np.array(self.state[1:1+self.stock_dim]))]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.day = 0

        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.avg_price_array = self.data.close.values.tolist()
        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                # print("data inside initiate state is ")
                # print(self.data)
                self.avg_price_array = self.data.close.values.tolist()
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                ) # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                # self.avg_price_array = self.data.close.values.tolist()
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                # self.avg_price_array = self.data.close.values.tolist()
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            # self.avg_price_array = self.data.close.values.tolist()
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
            )

        else:
            # for single stock
            # self.avg_price_array = self.data.close.values.tolist()
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(state_list,columns=['cash','Bitcoin_price','Gold_price','Bitcoin_num','Gold_num','Bitcoin_Disable','Gold_Disable'])
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            # print("date memory is ", self.date_memory)

            if config.model_alive:
                date_list = [self.date_memory[:-1]]
            else:
                date_list = self.date_memory[:-1]
            # print("date list is  ", date_list)
            df_date = pd.DataFrame(date_list)
            # print("last date from memory ", df_date.shape, "  ", df_date)
            df_date.columns = ["date"]

            action_list = self.actions_memory

            print("****************************")
            # print('type of action list is ', len(action_list))
            print('length of action list is ', len(action_list))
            print('action list item ', action_list)
            df_actions = pd.DataFrame(action_list)
            # cols = self.data.tic.values + ['return_val']
            print(self.data.tic.values)
            df_actions.columns = self.data.tic.values
            df_actions['return_val'] = self.rewards_memory
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            if config.model_alive:
                date_list = [self.date_memory[:-1]]
            else:
                date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    # populating action history

    def populate_action_history(self, txn_date, tic, sector, share_price, avg_share_price, model_name, action,
                                no_of_shares_for_action, amount_for_action, balance_left, total_asset_value,
                                return_val, reward, total_no_of_shares):

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
        if 'reward_val' not in config.action_info:
            config.action_info['reward_val'] = []
        config.action_info['reward_val'].append(reward)
        if 'total_no_of_shares' not in config.action_info:
            config.action_info['total_no_of_shares'] = []
        config.action_info['total_no_of_shares'].append(total_no_of_shares)

    def weighted_avg(self, current_price, num_of_stocks_in_transaction, avg_price, existing_stocks_number):
        x = (avg_price * existing_stocks_number)
        y = (current_price * num_of_stocks_in_transaction)
        z = num_of_stocks_in_transaction + existing_stocks_number

        try:
            wt_avg_price = (x + y) / z
        except:
            print(x , "  ", y, "  ", z)
        return round(wt_avg_price, 2)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        print("from get sb env")
        obs = e.reset()
        return e, obs