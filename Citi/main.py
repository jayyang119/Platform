import argparse
from datetime import datetime, date
from uti import DataLoader, Logger
from Database import GSDatabase, GSPriceDf
import re

GSD = GSDatabase()
logger = Logger()
DL = DataLoader()

if __name__ == '__main__':
    ######################## For debugging ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='price_df')
    parser.add_argument('--mode', type=str, default='stable')
    parser.add_argument('--day', type=int, default=1)
    args = parser.parse_args()

    MODE = args.mode
    TASK = args.task
    DAYS = args.day
    if MODE == 'beta':
        DL = DataLoader(mode='beta')
    ###########################################################################

    if TASK == 'database_update':
        GSD.GS_update_sentiment(update=True)

    elif TASK == 'price_df':
        GSPdf = GSPriceDf()
        GSPdf.GS_update_price_df(update=False)

    elif TASK == 'price_df_update':
        GSPdf = GSPriceDf()
        GSPdf.GS_update_price_df(update=True)

    elif TASK == 'price_df_predict':
        # GSD.GS_update_sentiment(update=True)
        GSPdf = GSPriceDf()

        if DAYS == 1 and datetime.today().weekday() == 0:
            DAYS = 3

        GSPdf.GS_predict_price_df(days=DAYS)


    else:

        raw = DL.loadDB('raw_og.csv')

        def get_ticker_from_headline(_headline):
            _ticker = ''
            if ':' not in _headline and '(' not in _headline:
                _ticker = ''
            elif '(' in _headline and ':' in _headline:   #  and _headline.find('(') < _headline.find(':')
                try:
                    _ticker = re.findall(r'\((.*?)\)[ /:]', _headline)[0]
                except Exception as e:
                    logger.error(f'Please check {_headline}')
                    _ticker = ''
            return _ticker

        # Cleansing
        # raw_og_ticker = DL.loadDB('raw_og_ticker.csv')
        # raw_og_ticker['Ticker'].fillna('', inplace=True)
        # raw_og_ticker.loc[raw_og_ticker['Ticker'] == '', 'Ticker'] = raw_og_ticker[raw_og_ticker['Ticker'] == ''][
        #     'ticker']
        # japan_no_dot = raw_og_ticker[
        #     (raw_og_ticker['regions'] == 'Japan') & ['.' not in x for x in raw_og_ticker['Ticker']]]
        # raw_og_ticker.loc[japan_no_dot.index, 'Ticker'] += '.T'
        #
        # DL.toDB(raw_og_ticker, 'raw_og_ticker_new.csv')
        # all_tickers, valid_tickers, delisted = DL.loadTickers()
        # valid_tickers = all_tickers[~all_tickers['Delisted'].isin(['Y'])]
        # valid_tickers_dict = all_tickers.set_index('Ticker(old)')[['Ticker']].to_dict()['Ticker']
        #
        # raw_og_ticker = DL.loadDB('raw_og_ticker_new.csv')
        # tickers = raw_og_ticker['Ticker'].unique()
        # tickers_valid = [x for x in tickers if x not in delisted['Ticker(old)'].values]
        # # tickers_tbu = [x for x in tickers if x in valid_tickers['Ticker(old)'].values]
        #
        # files = os.listdir(f'{DATABASE_PATH}/Daily')
        # files = [x.rstrip('.csv') for x in files]
        # tickers_tbu = [valid_tickers_dict[x] for x in tickers_valid if x not in files and valid_tickers_dict[x] not in files]
        #
        # Eikon_update_price_enhanced(tickers_tbu, threadcount=8)

        today_str = datetime.strftime(datetime.today(), '%Y%m%d')
        GSPdf = GSPriceDf()
        GSPdf.get_valid_tickers_dict()
        sentiment = DL.loadDB('sentiment_20220317.csv')


        # deal with edt, est
        def handle_edt():
            from datetime import timedelta, datetime
            import pandas as pd
            edt = sentiment[sentiment['Time'].str.contains('EDT')].copy()
            est = sentiment[~sentiment['Time'].str.contains('EDT')].copy()

            edt['Time'] = edt['Time'].str.rstrip(' EDT')
            est['Time'] = est['Time'].str.rstrip(' EST')
            DL.toDB(edt, 'edt.csv')
            DL.toDB(est, 'est.csv')

            edt = DL.loadDB('edt.csv', parse_dates=['Time'])
            est = DL.loadDB('est.csv', parse_dates=['Time'])

            edt['Time'] = edt['Time'] + timedelta(hours=12)
            est['Time'] = est['Time'] + timedelta(hours=13)
            sentiment_new = pd.concat([edt, est], axis=0)
            sentiment_new['Date'] = sentiment_new['Time'].apply(lambda x: x.date())
            sentiment_new = sentiment_new.sort_values('Time', ascending=False)
            sentiment_new = sentiment_new[['Date', 'Time', 'Ticker', 'company', 'Headline', 'Summary',  'regions',
           'sectors', 'Head analyst', 'Report Type', 'pubId', 'Headline sentiment',
           'Summary sentiment']]
            DL.toDB(sentiment_new, 'sentiment_20220317.csv')
        handle_edt()

