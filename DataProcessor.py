import config as config
import numpy as np
import pandas as pd

from YahooFinanceProcessor import YahooFinanceProcessor


class DataProcessor:
    def __init__(self, data_source):
        if data_source == "yahoofinance":
            self.processor = YahooFinanceProcessor()

    def download_data(self, ticker_list, start_date, end_date, time_interval) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)

        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array, split_vals, txn_dates_array= self.processor.df_to_array(
            df, config.TECHNICAL_INDICATORS_LIST, if_vix
        )
        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0

        return price_array, tech_array, turbulence_array, split_vals, txn_dates_array

    def add_stock_split(self, df) -> pd.DataFrame:
        return self.processor.add_stock_split_values(df)

if __name__ == "__main__":


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
    ticker_list = ['INFY.NS', 'TCS.NS']
    DP = DataProcessor(data_source='yahoofinance')
    data = DP.download_data(start_date='2010-02-17',
                            end_date='2022-02-17',
                            ticker_list=ticker_list,
                            time_interval='1d')

    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, TECHNICAL_INDICATORS_LIST)
    data = DP.add_vix(data)
    data.to_csv('stockData.csv')
    print("**********************************************")
    print(data.shape)
    print()
    print(data.head(50))