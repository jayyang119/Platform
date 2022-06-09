import os
import pandas as pd
import glob
from uti.logger import Logger


class DataLoader(Logger):
    """
        This class wraps up necessary functions to read data from database.
    """
    def __init__(self, mode='stable'):
        super().__init__()

        if mode.lower() == 'stable':
            self.output_path = self.database_path
        elif mode.lower() == 'beta':
            self.output_path = os.path.join(self.database_path, 'Beta')
        else:
            raise Exception('Please reinput operation mode.')

    def loadDaily(self, ticker, start=None):
        _path = os.path.join(self.daily_path, f"{ticker}.csv")
        if os.path.exists(_path):
            df = pd.read_csv(_path, parse_dates=['Date'], dtype={
                                        'OPEN': float,
                                        'HIGH': float,
                                        'LOW': float,
                                        'CLOSE': float,
                                        'MarketCap': float})
            # df.rename(columns={'Date': 'date', 'Price': 'close'}, inplace=True)
            df = df.set_index('Date')

            if start is not None:
                df = df.loc[start:]

            return df
        else:
            self.info(f"{ticker} unexist in the database. Please check.")
            return []
            # df.index = [datetime.strftime(" for _ in df.index]

    def loadDB(self, file, parse_dates=None):
        _path = os.path.join(self.database_path, f"{file}")
        self.info(f'Loading {_path}')
        if self.checkDB(file):
            if parse_dates is not None:
                assert type(parse_dates) == list, 'parse_dates should be Type list, please insert again.'
                if len(parse_dates) > 0:
                    df = pd.read_csv(_path, parse_dates=parse_dates, low_memory=True)
            else:
                df = pd.read_csv(_path)
            return df
        else:
            self.info(f"{_path} directory error. Please check.")
            return []

    def loadBT(self, file):
        _path = os.path.join(self.database_path, f"Backtest/{file}.csv")
        self.info(f'Loading {_path}')
        if self.checkDB(f'Backtest/{file}.csv'):
            df = pd.read_csv(_path, parse_dates=['publish_date_and_time'])
            return df
        else:
            self.info(f"{_path} directory error. Please check.")
            return []

    @classmethod
    def loadLog(cls, log_file=None):
        path = os.path.join(cls().database_path, f'Log/{cls().NOW_STR[:8]}')
        cls().create_folder(path)
        if log_file is not None:
            log_files = [os.path.basename(x) for x in glob.glob(path + '/*.csv')]
            log_file = max(log_files)
        else:
            log_files = [os.path.basename(x) for x in glob.glob(path + '/*.txt')]
            log_file = max(log_files)
        cls().info(f'Log file: {log_file}')
        return cls().loadDB(f'Log/{cls().NOW_STR[:8]}/{log_file}')

    def loadTickers(self):
        tickers_path = os.path.join(self.daily_path, 'tickers universe.csv')
        all_tickers = pd.read_csv(tickers_path)

        return all_tickers

    def toDB(self, data, filename, index=None, mode=None):
        _path = os.path.join(self.output_path, f"{filename}")
        if mode is None or not self.checkDB(_path):
            data.to_csv(_path, index=index, encoding='utf-8-sig')
        elif mode == 'a':
            data.to_csv(_path, mode='a', encoding='utf-8-sig')

    def toDaily(self, data, filename, index=None, mode=None):
        _path = os.path.join(self.daily_path, f"{filename}")
        if mode is None or not self.checkDB(_path):
            data.to_csv(_path, index=index, encoding='utf-8-sig')
        elif mode == 'a':
            data.to_csv(_path, mode='a', encoding='utf-8-sig')

    def toBT(self, data, filename, index=None, mode=None):
        _path = os.path.join(self.database_path, f"Backtest/{filename}.csv")
        if mode is None or not self.checkDB(_path):
            data.to_csv(_path, index=index, encoding='utf-8-sig')
        elif mode == 'a':
            data.to_csv(_path, mode='a', encoding='utf-8-sig')

    def checkDB(self, file, path=None):
        if path is None:
            return os.path.exists(os.path.join(self.database_path, file))
        else:
            return os.path.exists(os.path.join(path, file))

    def loadAnalytic(self, ticker, analytic='app', start=None):
        _path = os.path.join(self.database_path, f"analytics/{analytic}/{ticker}_{analytic}.csv")
        if os.path.exists(_path):
            df = pd.read_csv(_path, parse_dates=['date']).sort_values('date')
            df = df.set_index('date')
            if start is not None:
                df = df.loc[start:]

            df = df.reset_index()
            return df
        else:
            self.info(f"{ticker} unexist in the analytics {analytic}. Please check.")


if __name__ == '__main__':
    ll = DataLoader()
    print(ll.database_path)
