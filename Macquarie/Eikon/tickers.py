import eikon as ek
import os
import re
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from Broker import ricsregion
from uti import DataLoader, Logger, timeit
from Eikon.downloader import Eikon_update_price_enhanced

DL = DataLoader()
logger = Logger()
NOW_STR = DL.NOW_STR
DATABASE_PATH = DL.database_path

def Eikon_rewrite_tickers():
    data_df, valid_tickers, delisted = DL.loadTickers()
    data_dict = defaultdict(list)

    anomalies_ticker = []
    log = []
    for ticker in valid_tickers:
        print("%s" % (datetime.strftime(datetime.now(), "%H:%M:%S")), end=' ')
        try:
            data = ek.get_data(ticker, fields=['CF_CURR', 'TR.GICSIndustryGroup',
                                               'TR.CompanyMarketCap'])[0]
            if len(data) > 0:
                data_dict['Ticker'].append(ticker)
                data_dict['Sector'].append(data['GICS Industry Group Name'][0])
                data_dict['Currency'].append(data['CF_CURR'][0])
                data_dict['MarketCap'].append(data['Company Market Cap'][0])
            else:
                data_dict['Ticker'].append(ticker)
                data_dict['Sector'].append('')
                data_dict['Currency'].append('')
                data_dict['MarketCap'].append(0)
            print(ticker, 'Completed')
        except Exception as e:
            print(ticker, f'Error:{e}')
            anomalies_ticker.append(ticker)
            log.append(e)
    data_df = pd.DataFrame(data_dict)
    data_df = ricsregion(data_df)
    data_df = pd.concat([data_df, delisted], axis=0)
    DL.toDB(data_df, 'Ticker.csv', index=None)

    anomalies_df = pd.DataFrame({'Ticker': anomalies_ticker, 'Log': log})
    if len(anomalies_df) > 0:
        filename = f'Log/{NOW_STR[:8]}/Anomalies_MC_{NOW_STR[8:]}.csv'
        DL.create_folder(os.path.join(DATABASE_PATH, filename))
        DL.toDB(anomalies_df, filename, mode='a')
    return data_df

@timeit
def Eikon_check_tickers():
    data_df, valid_tickers, delisted = DL.loadTickers()
    data_df['Currency'] = data_df['Currency'].fillna('')
    need_update = []
    updated_i = []
    for i, row in data_df.iterrows():
        if len(row['Currency']) == 0 and row['Delisted'] != 'Y':
            ticker_old = row['Ticker(old)']
            ticker = row['Ticker']

            print("%s" % (datetime.strftime(datetime.now(), "%H:%M:%S")), end=' ')
            print(ticker_old, ticker, end=' ')
            if row['Ticker'] not in need_update:
                need_update.append(row['Ticker'])

            ticker = row['Ticker']
            try:
                data = ek.get_data(ticker, fields=['CF_CURR', 'TR.GICSIndustryGroup',
                                                   'TR.CompanyMarketCap'])[0]
                if len(data) > 0:
                    data_df.loc[i, 'Sector'] = data['GICS Industry Group Name'][0]
                    data_df.loc[i, 'Currency'] = data['CF_CURR'][0]
                    data_df.loc[i, 'MarketCap'] = data['Company Market Cap'][0]

                else:
                    print('no data', end=' ')
                    data_df.loc[i, 'Sector'] = ''
                    data_df.loc[i, 'Currency'] = ''
                    data_df.loc[i, 'MarketCap'] = 0
                if not os.path.exists(os.path.join(DATABASE_PATH, f'Daily/{ticker_old}.csv')) and not os.path.exists \
                        (os.path.join(DATABASE_PATH, f'Daily/{ticker}.csv')):
                    updated_i.append(i)
                print('Completed')
            except Exception as e:
                logger.error(ticker + 'Error.')
                logger.error(e)
        # if i % 100 == 0:
        # data_df.to_csv(writefile_path)

    valid_tickers = ricsregion(valid_tickers)
    logger.info('Valid tickers: ')
    logger.info(valid_tickers)
    data_df = pd.concat([valid_tickers, delisted], axis=0)
    data_df = data_df.sort_index(ascending=True)

    logger.info('Updated rows:\n')
    logger.info(data_df.loc[updated_i])
    DL.toDB(data_df, 'Tickers.csv', index=None)

    logger.info('Tickers TBU: ')
    logger.info(need_update)
    if len(need_update) > 0:
        Eikon_update_price_enhanced(need_update)
    return data_df, need_update

@timeit
def Eikon_update_tickers():
    all_tickers, valid_tickers, delisted = DL.loadTickers()

    sentiment_df = DL.loadDB('GS_sentiment.csv', parse_dates=['Date', 'Time'])
    sentiment_df = sentiment_df[sentiment_df['Asset Class'].isin(['Equity'])]

    today = datetime.today()
    Y = today.year
    m = today.month
    d = today.day
    date_range = pd.date_range(datetime(Y, m, d ) -timedelta(60), datetime(Y, m, d))
    # date_range = pd.date_range(min(sentiment_df['Date']), datetime(Y, m, d))

    try:
        recent_tickers = sentiment_df[sentiment_df['Date'].
            isin(date_range)][['Ticker']].drop_duplicates()
        recent_tickers = recent_tickers[['Ticker']].applymap(lambda x: re.split('[/|]', x)[0].strip())

        recent_tickers = recent_tickers[~recent_tickers['Ticker']
            .isin(all_tickers['Ticker(old)'])]
        recent_tickers['Ticker(old)'] = recent_tickers['Ticker'].copy(deep=True)

        new_tickers = pd.concat([valid_tickers, delisted], axis=0)
        new_tickers = pd.concat([new_tickers, recent_tickers], axis=0)
        new_tickers = new_tickers.drop_duplicates()
        DL.toDB(new_tickers, 'Tickers.csv', index=None)

        data_df, need_update = Eikon_check_tickers()
        return data_df, need_update
    except Exception as e:
        logger.error(e)
    return


def anomalies_handling(guess='.O'):
    def get_anomalies_tickers():
        # anomalies_tickers = pd.read_csv(os.Path.join(DATABASE_PATH, 'Anomalies.csv'), index_col=0)
        anomalies_tickers = DL.loadLog('csv')
        valid_tickers = DL.loadDB('Tickers.csv')
        anomalies_tickers = anomalies_tickers[~anomalies_tickers['Ticker'].isin(valid_tickers['Ticker(old)'])]
        return anomalies_tickers


    def try_anomalies(guess=guess):
        anomalies_tickers = get_anomalies_tickers()
        O_tickers = anomalies_tickers[~anomalies_tickers['Ticker'].str.contains('\.')] + guess
        Eikon_update_price_enhanced(O_tickers['Ticker'], daily_update=True)

        error_log_file = max(os.listdir(os.path.join(DATABASE_PATH, 'Log')))
        error_df = pd.read_csv(os.path.join(DATABASE_PATH, f'Log/{error_log_file}'), index_col=0)
        error_df.head()
        O_tickers = O_tickers[~O_tickers['Ticker'].isin(error_df['Ticker'])]

        O_tickers['Ticker(old)'] = O_tickers['Ticker'].str.split('.', 0, expand=True)[0]
        DL.toDB(O_tickers[['Ticker(old)', 'Ticker']],
                f'Log/{guess}_tickers.csv')

        anomalies_tickers = anomalies_tickers[~anomalies_tickers['Ticker'].isin(O_tickers['Ticker(old)'])]

        return anomalies_tickers

    anomalies_tickers = try_anomalies(guess)
    return anomalies_tickers

