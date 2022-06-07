import pandas as pd
import numpy as np

from uti import DataLoader, Logger
from Broker import get_pnl, MARKET_MAPPING_DICT, REGION_MAPPING_DICT

logger = Logger()
DL = DataLoader()


class DataCleaner:
    def __init__(self):
        pass

    @staticmethod
    def preprocess_trade_df(_df):
        """
            1. Group the exch_locations to regions of trading interest
            2. Clean sectors
            3. Converg Market Cap in USD into Market Cap Group
        """

        assert type(_df) == pd.DataFrame, 'df must be pd.DataFrame, please check.'
        df = _df.copy(deep=True)
        df = df[df['summary'] != '']
        df = df[df['headline'] != '']

        df['exch_region2'] = df['exch_region']
        df['exch_location'] = df['exch_location'].replace(MARKET_MAPPING_DICT)
        df['exch_region'] = df['exch_location'].replace(REGION_MAPPING_DICT)
        df.loc[df['exch_region2'] == 'Europe', 'exch_region'] = 'Europe'
        df['Sector'] = df['Sector'].fillna('').apply(lambda x: x.split(',')[0])

        # Grouping market capitalization
        if 'market_cap_grp' not in df.columns:
            bins_mc = [-np.inf, 5e+7, 3e+8, 2e+9, 1e+10, 2e+11, np.inf]
            mc_labels = ['NanoCap', 'MicroCap', 'SmallCap', 'MidCap', 'LargeCap', 'MegaCap']
            print(df.columns)
            df['market_cap_grp'] = pd.cut(x=df['market_cap_usd'], bins=bins_mc, include_lowest=True, labels=mc_labels)

        return df

    def get_benchmark_test_data(self, update=False):
        """
            1. Split price_df into train_df, test_df, and calculate pnl based on headline sentiment
                [long positive, short negative]
            2. Output to database
            3. If not update mode, simply read train_df, test_df from database.

        """

        if not DL.checkDB('Backtest/Benchmark data/X_train.csv') or update:
            df = DL.loadDB('price_df.csv', parse_dates=['Date', 'Time'])
            df = self.preprocess_trade_df(df)
            df = df.sort_values('Date', ascending=True)

            DL.create_folder(f'{DL.database_path}/Backtest/Benchmark data')
            # Blind long
            df['side'] = 'long'
            df = get_pnl(df)
            DL.toBT(df, 'Benchmark data/blind long')

            # Blind short
            df['side'] = 'short'
            df = get_pnl(df)
            DL.toBT(df, 'Benchmark data/blind short')

            # Benchmark rules
            from Model.rules import benchmark_rule, benchmark_rule2, benchmark_rule3
            df['side'] = benchmark_rule(df)
            df['side'] = benchmark_rule2(df)
            df['side'] = benchmark_rule3(df)
            df = get_pnl(df)

            train_data = df.iloc[:int(len(df) * 0.75)]
            test_data = df.iloc[int(len(df) * 0.75):]
            DL.toBT(train_data, 'Benchmark data/X_train')
            DL.toBT(test_data, 'Benchmark data/X_test')

        else:
            train_data = DL.loadDB('Backtest/Benchmark data/X_train.csv', parse_dates=['Date', 'Time'])
            test_data = DL.loadDB('Backtest/Benchmark data/X_test.csv', parse_dates=['Date', 'Time'])

        test_data = test_data[test_data['d0_open'] > 0].reset_index(drop=True)
        return train_data, test_data

    def get_model_test_data(self, strategy):
        if DL.checkDB(f'Backtest/{strategy}.csv'):
            test_data = DL.loadDB(f'Backtest/{strategy}.csv', parse_dates=['Time'])
        else:
            test_data = DL.loadDB('Backtest/Model data/X_test.csv', parse_dates=['Time'])

            test_data = test_data[test_data['d0_open'] > 0].reset_index(drop=True)
            # test_data = test_data[~test_data['Side'].isin(['neutral'])]
            test_data = test_data[~test_data['Headline sentiment'].isin(['neutral'])].reset_index(drop=True)
            # test_data = test_data[~test_data['Summary sentiment'].isin(['neutral'])].reset_index(drop=True)

        logger.info(test_data)
        return test_data


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
    trade_df = DL.loadDB('price_df.csv', parse_dates=['Time'])
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
