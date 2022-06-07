import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from uti import DataLoader, Logger
from Broker import get_pnl, ricsregion
from Model.rules import asia_df, am_df, eu_df

logger = Logger()
DL = DataLoader()


def benchmark_filter(df, bt=False):
    asia_data = asia_df(df)
    am_data = am_df(df)
    euro_data = eu_df(df)
    all_data = pd.concat([asia_data, euro_data, am_data], axis=0).reset_index(drop=True) # , am_data
    # all_data = pd.concat([asia_data], axis=0).reset_index(drop=True)


    if bt is True:
        all_data = get_pnl(all_data)
    return all_data


class DataCleaner:
    def __init__(self):
        pass

    @classmethod
    def preprocess_trade_df(cls, _df):
        assert type(_df) == pd.DataFrame, 'df must be pd.DataFrame, please check.'
        df = _df.copy(deep=True)
        df = df[df['Summary'] != ''].copy(deep=True)
        df = df[df['Headline'] != ''].copy(deep=True)
        df = ricsregion(df)

        outliers = 'SNGRq.L'
        df = df[df['Ticker'] != outliers].reset_index(drop=True)  # Outlier

        # Grouping market capitalization
        bins_mc = [-np.inf, 5e+7, 3e+8, 2e+9, 1e+10, 2e+11, np.inf]
        mc_labels = ['NanoCap', 'MicroCap', 'SmallCap', 'MidCap', 'LargeCap', 'MegaCap']
        df['market_cap_grp'] = pd.cut(x=df['market_cap_usd'], bins=bins_mc, include_lowest=True, labels=mc_labels)
        df['No. of trades'] = 1

        return df

    @staticmethod
    def split_data(X, y, time=None):
        if time is None:
            # Stratify to ensure the proportion of data
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=9988)
            for train_index, test_index in skf.split(X[['Side', 'exch_location']], y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train = X[X['Date'] < time]  # datetime(2021, 6, 30)
            X_test = X[X['Date'] >= time]
        return X_train, X_test

    def train_test_data(self):
        trade_df = DL.loadDB('price_df.csv', parse_dates=['Date', 'Time'])
        trade_df = self.preprocess_trade_df(trade_df)
        trade_df = trade_df[trade_df['release_period'] != 'Within'].reset_index(drop=True)

        X_train, X_test = self.split_data(trade_df, trade_df['exch_location'], time=datetime(2021, 3, 1))
        DL.toDB(X_train, 'Backtest/Benchmark data/X_train.csv', index=None)
        DL.toDB(X_test, 'Backtest/Benchmark data/X_test.csv', index=None)

        trade_df['side'] = 'long'
        trade_df = get_pnl(trade_df)
        DL.toDB(trade_df, 'Backtest/Benchmark data/blind_long.csv')

        trade_df['side'] = 'short'
        trade_df = get_pnl(trade_df)
        DL.toDB(trade_df, 'Backtest/Benchmark data/blind_short.csv')

        trade_df = benchmark_filter(trade_df)
        trade_df = get_pnl(trade_df)
        DL.toDB(trade_df, 'Backtest/Benchmark data/benchmark.csv')

    def get_benchmark_test_data(self, update=False):
        if not DL.checkDB('Backtest/Benchmark data/X_train.csv') or update:
            self.train_test_data()

        train_data = DL.loadDB('Backtest/Benchmark data/X_train.csv', parse_dates=['Date', 'Time'])
        test_data = DL.loadDB('Backtest/Benchmark data/X_test.csv', parse_dates=['Date', 'Time'])
        train_data['side'] = train_data['Headline sentiment'].copy()
        test_data['side'] = test_data['Headline sentiment'].copy()
        test_data = test_data[test_data['d0_open'] > 0].reset_index(drop=True)

        # Historical expectancy based on day 0 R
        holding_period = 'd0_r'
        if holding_period not in train_data.columns:
            train_data = get_pnl(train_data)
            test_data = get_pnl(test_data)
            DL.toDB(train_data, 'Backtest/Benchmark data/X_train.csv')
            DL.toDB(test_data, 'Backtest/Benchmark data/X_test.csv')

        train_data['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)
        test_data['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)

        train_data = train_data[~train_data['d0_date'].isna()]
        test_data = test_data[~test_data['d0_date'].isna()]

        return train_data, test_data

    def get_model_test_data(self, strategy):
        if DL.checkDB(f'Backtest/{strategy}.csv'):
            test_data = DL.loadDB(f'Backtest/{strategy}.csv', parse_dates=['Date', 'Time'])
        else:
            test_data = DL.loadDB('Backtest/Model data/X_test.csv', parse_dates=['Date', 'Time'])

            test_data = test_data[test_data['d0_open'] > 0].reset_index(drop=True)
            # test_data = test_data[~test_data['Side'].isin(['neutral'])]
            test_data = test_data[~test_data['Headline sentiment'].isin(['neutral'])].reset_index(drop=True)
            # test_data = test_data[~test_data['Summary sentiment'].isin(['neutral'])].reset_index(drop=True)

        logger.info(test_data)
        return test_data


def get_y_expectancy(df):
    R_df = pd.DataFrame(columns=['R0', 'R1', 'R2'])
    R_df['R0'] = (df['D0 Close'] - df['d0_open']) / df['ATR']
    R_df['R1'] = (df['D1 Close'] - df['d0_open']) / df['ATR']
    R_df['R2'] = (df['D2 Close'] - df['d0_open']) / df['ATR']
    R_df['Output'] = R_df.mean(axis=1)
    # R_df['Output'] = R_df['R0'].copy()
    return R_df['Output']


if __name__ == '__main__':
    senti_cat = ['positive', 'negative', 'neutral']
    x_columns = [
        'Head analyst',
        'exch_location', 'Region', 'Sector', 'MarketCap',
        'Headline sentiment',
        'Headline neutral score', 'Headline positive score', 'Headline negative score',
        'Summary sentiment',
        'Summary neutral score', 'Summary positive score', 'Summary negative score',
        'Earnings', 'Initiate', 'Rating', 'Estimates', 'Guidance', 'Review'
    ]

    # filename = 'Headline strategy'
    trade_df = DL.loadDB('price_df.csv', parse_dates=['Date', 'Time'])
    print(trade_df)
    trade_df = DataCleaner().preprocess_trade_df(trade_df)

    # Labeling for categorical data
    analyst_cat = np.unique(trade_df['Head analyst'])
    sector_cat = np.unique(trade_df['Sector'])
    region_cat = np.unique(trade_df['Region'])
    market_cat = np.unique(trade_df['exch_location'])
    mc_cat = np.unique(trade_df['MarketCap'])
    report_type_cat = [0, 1]

    # Model Data
    # bins = [-np.inf, -2.5, -0.5, 0.5, 2.5, np.inf] # Future improvement on segmentations.
    bins = [-np.inf, -0.45, 0.45, np.inf]  # Future improvement on segmentations.
    # bins = [-np.inf, 0, np.inf]
    X_train = DL.loadDB('Backtest/Model data/X_train.csv')
    # y_train = get_y_expectancy(X_train)
    # y_train = pd.cut(x=y_train, bins=bins,
    #                  include_lowest=True, labels=range(len(bins) - 1))

    # X_test = X_test[~X_test['Side'].isin(['neutral'])]
    X_test = DL.loadDB('Backtest/Model data/X_test.csv')
    # y_test = get_y_expectancy(X_test)
    # y_test = pd.cut(x=y_test, bins=bins,
    #                 include_lowest=True, labels=range(len(bins) - 1))

    # X_train = X_train[x_columns]
    # X_test = X_test[x_columns]
