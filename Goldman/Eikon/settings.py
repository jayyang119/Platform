import threading
import os
import pandas as pd
import numpy as np
import eikon as ek

ek.set_app_key('cec2abe836754fe692a01de15032769539511a71')
import time
from datetime import datetime

from uti import Logger, DataLoader
from Broker import get_tr, get_atr
from Path import DATABASE_PATH, DAILY_PATH

DL = DataLoader()
logger = Logger()
lock = threading.RLock()

NOW_STR = logger.NOW_STR
testing = False


def Eikon_update_price(tickers=[], daily_update=False):
    def merge_market_cap(_ticker, _df):
        if 'MarketCap' not in _df.columns:
            mc = get_marketcap_ts(_ticker, sdate=_df.index[0], edate=_df.index[-1])
            _df = pd.concat([_df, mc], axis=1)

            _df = _df.fillna(method='ffill').fillna(method='bfill')
        return _df

    def get_marketcap_ts(_ticker, sdate, edate):
        if type(sdate) != str:
            sdate = datetime.strftime(sdate, '%Y%m%d')
            edate = datetime.strftime(edate, '%Y%m%d')

        mc = ek.get_data(_ticker, ['TR.CompanyMarketCap.Date', {'TR.CompanyMarketCap': {'params': {'Curn': 'USD'}}}],
                         {'Sdate': sdate,
                          'EDate': edate, 'Frq': 'D'})[0]
        mc = mc.replace('', np.nan).dropna()
        mc['Date'] = mc['Date'].apply(lambda x: datetime.strptime(x.split('T')[0], '%Y-%m-%d'))
        mc = mc.loc[~mc.duplicated('Date')]
        mc = mc.drop(['Instrument'], axis=1)
        mc = mc.set_index('Date')
        mc.rename(columns={'Company Market Cap': 'MarketCap'}, inplace=True)
        return mc

    anomalies_mc = []
    log = []

    for ticker in tickers:
        try:
            ticker = ticker.rstrip('.csv')
            # print("%s" % (datetime.strftime(datetime.now(), "%H:%M:%S")), end=' ')

            writepath = f"{DAILY_PATH}/{ticker}.csv"
            if os.path.exists(writepath) and daily_update:
                df = pd.read_csv(writepath, parse_dates=['Date'])
                df = df.set_index('Date')
                df = get_tr(df)

            else:
                print(f"New ticker: {ticker}, downloading from Eikon")
                df = ek.get_timeseries(ticker, fields='*', start_date='2015-01-01')
                df = merge_market_cap(ticker, df)
                df = get_tr(df)
                df = get_atr(df)

            if 'MarketCap' not in df.columns:
                df = merge_market_cap(ticker, df)

            if daily_update:
                new_df = ek.get_timeseries(ticker, fields='*',
                                           start_date=datetime.strftime(df.index[-1], "%Y-%m-%d"),
                                           # end_date=datetime.strftime(now, "%Y-%m-%d")
                                           )
                new_df = merge_market_cap(ticker, new_df)

                previous_df = df.iloc[-11:-1][new_df.columns]
                new_df = pd.concat([previous_df, new_df], axis=0)
                new_df = get_tr(new_df)
                df = pd.concat([df.iloc[:-1], new_df.iloc[10:]], axis=0)
                df = get_atr(df)

            # TR, ATR, liquidity, Gap.
            if not testing:
                df = df[~df.index.duplicated(keep='last')].reset_index(drop=False)
                DL.toDaily(df, f'{ticker}.csv')
            else:
                return df

        except Exception as e:
            log.append(e)
            anomalies_mc.append(ticker)
            logger.error(ticker)
        time.sleep(0.00001)
    anomalies_df = pd.DataFrame({'Ticker': anomalies_mc, 'Log': log})
    if len(anomalies_df) > 0:
        filename = f'Log/{NOW_STR[:8]}/Anomalies_MC_{NOW_STR[8:]}.csv'
        DL.create_folder(os.path.join(DATABASE_PATH, filename))
        DL.toDB(anomalies_df, filename, mode='a')


exitFlag = 0


class myThread(threading.Thread):
    def __init__(self, threadID, tickers):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = 'Thread-' + str(self.threadID)
        self.tickers = tickers

    def run(self):
        logger.info('Starting' + self.name + ' # of tickers: ' + str(len(self.tickers)))
        logger.info(self.tickers)
        # Eikon_update_price(self.tickers, daily_update=True)
        # if lock.acquire(1):
        lock.acquire()
        try:
            Eikon_update_price(self.tickers, daily_update=True)
        except Exception as e:
            logger.error(e)
        finally:
            lock.release()
        print('Exiting', self.name)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


