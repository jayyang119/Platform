import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from uti import DataLoader, Logger
from library import Dataset
from Model import DataCleaner, benchmark_expectancy
from Broker import ricsregion
from Crawler import REPORT_TYPE_GLOBAL_DICT

DC = DataCleaner()
DL = DataLoader()
logger = Logger()
DATABASE_PATH = DL.database_path
NOW_STR = logger.NOW_STR


class GSPriceDf:
    def __init__(self):
        pass

    def add_new_columns(self, _df):
        df = _df.copy(deep=True)
        df['Sector'] = df[['Ticker']].replace({'Ticker': self._valid_tickers_dict['Industry']})
        df = ricsregion(df)

        df['atr'] = 0.00
        df['atrx'] = 1.00
        df['atr_used'] = 0.00
        df['gap'] = 0.00
        df['market_cap_usd'] = 0.00
        df['volume_d_10_sma'] = 0.00

        for D in range(-1, 3):
            if D == -1:
                df[f"prev1_date"] = ''
            else:
                df[f"d{D}_date"] = ''
            for col in ["open", "high", "low", "close"]:
                if D == -1:
                    df[f"prev1_{col}"] = 0.0
                else:
                    df[f"d{D}_{col}"] = 0.0

        DS = Dataset(df)
        DS.clean(mode='eikon')
        df = DS.df.copy(deep=True)

        return df

    @staticmethod
    def get_new_columns_index_dict(df):
        column_index_dict = {}
        price_df_columns = list(df.columns)

        column_index_dict['atr'] = price_df_columns.index('atr')
        column_index_dict['atrx'] = price_df_columns.index('atrx')
        column_index_dict['atr_used'] = price_df_columns.index('atr_used')
        column_index_dict['market_cap_usd'] = price_df_columns.index('market_cap_usd')
        column_index_dict['gap'] = price_df_columns.index('gap')
        column_index_dict['volume_d_10_sma'] = price_df_columns.index('volume_d_10_sma')

        for D in range(-1, 3):
            if D == -1:
                column_index_dict[f"prev1_date"] = price_df_columns.index(f"prev1_date")
            else:
                column_index_dict[f"d{D}_date"] = price_df_columns.index(f"d{D}_date")
            for col in ["open", "high", "low", "close"]:
                if D == -1:
                    column_index_dict[f"prev1_{col}"] = price_df_columns.index(f"prev1_{col}")
                else:
                    column_index_dict[f"d{D}_{col}"] = price_df_columns.index(f"d{D}_{col}")

        return column_index_dict

    def get_valid_tickers_dict(self):
        self._valid_tickers = DL.loadTickers()
        self._valid_tickers_dict = self._valid_tickers.set_index('Ticker(old)').to_dict()

    def get_sentiment_df(self, file='GS_sentiment'):
        logger.info(f'Getting {file}')
        df = DL.loadDB(f'{file}.csv', parse_dates=(['Date', 'Time']))
        df = df[df['Asset Class'].isin(['Equity'])]
        df = df[df['Ticker'].isin(self._valid_tickers['Ticker(old)'])].reset_index(drop=True)
        return df

    def update_price(self, _df):
        df = _df.copy(deep=True)  # A copy
        df = self.add_new_columns(df)
        column_index_dict = self.get_new_columns_index_dict(df)

        errors = []
        errors_log = []
        for i, row in df.iterrows():
            try:
                date = row['date_local']
                ticker = row['Ticker']
                data = DL.loadDaily(self._valid_tickers_dict['ticker'][ticker])

                data_date0_index = list(data.index).index(data.loc[date:].index[0])
                if row['release_period'] == 'After':
                    data_date0_index += 1
                data_index_dict = {0: data_date0_index, 1: data_date0_index + 1,
                                   2: data_date0_index + 2, -1: data_date0_index - 1}

                for D in range(-1, 3):
                    data_date_D = data.iloc[data_index_dict[D]]

                    for col in ["open", "high", "low", "close"]:
                        if D == -1:
                            df.iat[i, column_index_dict[f"prev1_date"]] = data_date_D.name
                            df.iat[i, column_index_dict[f"prev1_{col}"]] = data_date_D[col.upper()]
                        else:
                            df.iat[i, column_index_dict[f"d{D}_date"]] = data_date_D.name
                            df.iat[i, column_index_dict[f"d{D}_{col}"]] = data_date_D[col.upper()]

                    if D == -1:
                        df.iat[i, column_index_dict['market_cap_usd']] = data_date_D['MarketCap']
                        df.iat[i, column_index_dict['gap']] = data_date_D['Gap']
                        df.iat[i, column_index_dict['atr']] = data_date_D['ATR']
                        df.iat[i, column_index_dict['atr_used']] = data_date_D['ATR'] * df.iat[i, column_index_dict['atrx']]

                df.iat[i, column_index_dict['volume_d_10_sma']] = data.iloc[:data_date0_index].iloc[-10:][
                    'VOLUME'].mean()
                if i % 1000 == 0:
                    logger.info(f'Line {i} {ticker} completed.')

            except Exception as e:
                logger.error(row['Ticker'])
                errors.append(row['Ticker'])
                errors_log.append(e)
        anomalies_df = pd.DataFrame({'Ticker': errors, 'Log': errors_log})

        if len(anomalies_df) > 0:
            filename = f'Log/{NOW_STR[:8]}/Anomalies_MC_{NOW_STR[8:]}.csv'
            DL.create_folder(os.path.join(DATABASE_PATH, filename))
            DL.toDB(anomalies_df, filename, mode='a')
        else:
            logger.info('All tickers completed without anomalies.')

        logger.info(df)
        df = df[df['atr_used'] != 0].copy()
        df['gap_in_atr'] = df['gap'] / df['atr_used']
        df['gap_in_atr'] = pd.to_numeric(df['gap_in_atr'])
        return df

    def GS_update_price_df(self, update=False):
        self.get_valid_tickers_dict()
        self._sentiment_df = self.get_sentiment_df()

        if not DL.checkDB('price_df.csv'):
            logger.info('Regenerating price_df.csv')

            if update is True:
                tickers_tbu = np.unique(self._sentiment_df['Ticker'].replace(self._valid_tickers_dict['Ticker']))
                len_tickers_tbu = str(len(tickers_tbu))
                order = input(f'Total length of tickers tbu {len_tickers_tbu}, continue? (y/n)')
                if order.lower() == 'y':
                    from Eikon import Eikon_update_price_enhanced
                    Eikon_update_price_enhanced(tickers_tbu, threadcount=16)

            self._price_df = self.update_price(self._sentiment_df)

        else:
            logger.info('Updating price_df.csv')
            price_df_tbu = self.get_sentiment_df('price_df')

            price_df = price_df_tbu[price_df_tbu['d2_open'] != 0]
            price_df_tbu = price_df_tbu[price_df_tbu['d2_open'] == 0]

            price_df_new = pd.concat([self._sentiment_df, price_df[self._sentiment_df.columns]], axis=0).drop_duplicates(
                ['Time', 'Headline'], keep=False)

            price_df_tbu = pd.concat([price_df_new, price_df_tbu], axis=0)
            price_df_tbu = self.add_new_columns(price_df_tbu)
            logger.info(price_df_tbu)

            if update is True:
                tickers_tbu = np.unique(price_df_tbu['Ticker'].fillna('').replace(self._valid_tickers_dict['ticker']))
                logger.info(tickers_tbu)
                len_tickers_tbu = str(len(tickers_tbu))
                order = input(f'Total length of tickers tbu {len_tickers_tbu}, continue? (y/n)')
                if order.lower() == 'y':
                    from Eikon import Eikon_update_price_enhanced
                    Eikon_update_price_enhanced(tickers_tbu, threadcount=16)

            price_df_tbu = self.update_price(price_df_tbu)
            logger.info(price_df)
            logger.info(price_df_tbu)

            self._price_df = pd.concat([price_df_tbu[price_df.columns], price_df], axis=0).reset_index(drop=True)

        self._price_df = self._price_df.sort_values('Date', ascending=False)
        self._price_df = self._price_df.drop_duplicates()

        DL.toDB(self._price_df, 'price_df.csv')

        DC = DataCleaner()
        DC.get_benchmark_test_data(update=True)

    def update_market_cap(self, df):
        if 'market_cap_usd' not in df.columns:
            df['market_cap_usd'] = 0.0
        id_mc = list(df.columns).index('market_cap_usd')
        for i, row in df.iterrows():
            try:
                ticker = row['Ticker']
                data = DL.loadDaily(self._valid_tickers_dict['ticker'][ticker])

                df.iat[i, id_mc] = data['MarketCap'][-1]
            except:
                logger.error(row['Ticker'])
        df = DC.preprocess_trade_df(df)
        return df

    def GS_predict_price_df(self, days=3):
        self.get_valid_tickers_dict()
        # self._sentiment_df = DL.loadDB('GS_sentiment.csv', parse_dates=(['Date', 'Time']))
        # self._sentiment_df = self._sentiment_df[self._sentiment_df['Asset Class'].isin(['Equity'])]
        self._sentiment_df = self.get_sentiment_df()

        # date-local - today release_period - before
        # date_local - yesterday release_period - after    # filter out date_local - yesterday, release_period before
        now = datetime.today()
        end = datetime.date(datetime(now.year, now.month, now.day))
        start = end - timedelta(days=days)
        date_range = [x.date() for x in pd.date_range(start, end)]
        print(date_range)

        self._sentiment_df['Date'] = self._sentiment_df['Date'].apply(lambda x: x.date())
        self._sentiment_df = self.add_new_columns(self._sentiment_df)

        price_df = self._sentiment_df[self._sentiment_df['date_local'].isin(date_range)].copy(deep=True)
        price_df = price_df[price_df['release_period'].isin(['After', 'Within', 'Before'])].reset_index(drop=True)
        start_after = (price_df['date_local'] == start) & (price_df['release_period'] == 'After')
        print(start)
        logger.info(price_df.loc[start_after])
        logger.info(price_df.loc[price_df['date_local'] > start])
        price_df = price_df.loc[start_after | (price_df['date_local'] > start)].reset_index(drop=True)  # Filter out start day

        DC = DataCleaner()
        price_df = self.update_market_cap(price_df)
        price_df = DC.preprocess_trade_df(price_df)

        train_data, test_data = DC.get_benchmark_test_data()
        train_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

        results = benchmark_expectancy(sort_by='exch_region', train_data=train_data, test_data=price_df)
        results.sort_values('Time', ascending=False, inplace=True)
        today_str = datetime.strftime(datetime.today(), '%Y%m%d')

        results = results.drop(['volume_d_10_sma', 'prev1_open', 'prev1_high', 'prev1_low', 'prev1_close', 'gap',
                                'd0_date', 'd0_open', 'd0_high', 'd0_low', 'd0_close', 'd1_date', 'd1_open', 'd1_high',
                                'd1_low', 'd1_close',
                                'd2_date', 'd2_open', 'd2_high', 'd2_low', 'd2_close',
                                'prev1_date', 'ticker', 'atr', 'atrx', 'atr_used', 'ticker_updated', 'No. of trades'], axis=1)

        results['rating_curr'] = None
        results['rating_prev'] = None
        results['tp_curr'] = None
        results['tp_prev'] = None
        results['tp_chg_pct'] = None
        results['up_or_down_side_pct'] = None
        # results['d0_exp'] = None
        results['d1_exp'] = None
        results['d2_exp'] = None
        results['top_analyst_long'] = None
        results['top_analyst_short'] = None
        results['analyst_pri'] = None

        results['broker'] = 'Goldman'
        results['Ticker(BBG)'] = results['Ticker'].replace(self._valid_tickers_dict['Ticker(BBG)'])
        results['Report Type Global'] = results['Report Type'].replace(REPORT_TYPE_GLOBAL_DICT)

        results = results[
            ['broker', 'Ticker(BBG)', 'date_local', 'release_period', 'side', 'Report Type', 'Report Type Global', 'Sector',
             'Headline sentiment', 'Summary sentiment', 'Headline', 'Summary',
             'rating_curr', 'rating_prev', 'tp_curr', 'tp_prev', 'tp_chg_pct', 'up_or_down_side_pct',
             'Expectancy', 'd1_exp', 'd2_exp', 'Expectancy (blind long)', 'Expectancy (blind short)',
             'top_analyst_long', 'top_analyst_short', 'analyst_pri', 'exch_location', 'exch_region', 'Ticker', 'Time',  'market_cap_grp']]

        DL.toDB(results, f'GS daily prediction {today_str}.csv')

        return price_df


if __name__ == '__main__':
    GSP = GSPriceDf()
    GSP.GS_update_price_df(update=True)
