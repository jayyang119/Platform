import os
import argparse

from uti import DataLoader, Logger
from Database import GSDatabase
from Crawler import GS_crawler, GS_crawler_report_type
from Path import DATABASE_PATH
from Database import GSPriceDf
from datetime import datetime

GSD = GSDatabase()
logger = Logger()
DL = DataLoader()

if __name__ == '__main__':
    ######################## For debugging ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='crawler')
    # parser.add_argument('--task', type=str, default='analysis')
    # parser.add_argument('--task', type=str, default='price_df_predict')
    # parser.add_argument('--task', type=str, default='update_database')
    # parser.add_argument('--task', type=str, default='else')
    parser.add_argument('--mode', type=str, default='stable')

    parser.add_argument('--day', type=int, default=1)
    # modes = ['stable', 'beta', 'backtest', 'prediction']
    # parser.add_argument('--mode', type=str, default='beta')
    args = parser.parse_args()

    MODE = args.mode
    TASK = args.task
    DAYS = args.day
    if MODE == 'beta':
        DL = DataLoader(mode='beta')
    ###########################################################################

    if TASK == 'rewrite_database':
        GSD.GS_rewrite_database(
            latest_dates=sorted(os.listdir(os.path.join(DATABASE_PATH, 'Goldman Reports')), reverse=True))

    elif TASK == 'crawler':
        GS_crawler(crawl_older_dates=False)  # Daily update

    elif TASK == 'crawler_older_dates':
        GS_crawler(crawl_older_dates=True)  # Crawl data from the past
        GSD.GS_update_sentiment()

    elif TASK == 'database_update':
        GSD.GS_update_sentiment()

    elif TASK == 'price_df':
        GSPdf = GSPriceDf()
        GSPdf.GS_update_price_df(update=False)

    elif TASK == 'price_df_update':
        GSD.GS_update_sentiment()
        GSPdf = GSPriceDf()
        GSPdf.GS_update_price_df(update=True)

    elif TASK == 'price_df_predict':
        GSD.GS_update_sentiment()
        GSPdf = GSPriceDf()

        if DAYS == 1 and datetime.today().weekday() == 0:
            DAYS = 3

        GSPdf.GS_predict_price_df(days=DAYS)


    else:
        import pandas as pd

        sentiment = DL.loadDB('GS_sentiment.csv')
        sentiment = sentiment[sentiment['Asset Class'] == 'Equity']
        universe = DL.loadDB('GS stock universe coverage.csv')

        tickers_unique = sentiment['Ticker'].unique()
        tickers_not_in_universe = [x for x in tickers_unique if x not in universe['Ticker'].values]
        tickers_not_in_universe_df = pd.DataFrame(tickers_not_in_universe)
        DL.toDB(tickers_not_in_universe_df, 'tickers_not_in_universe_df.csv')
        all_tickers = DL.loadTickers()

        tickers_not_in_universe_in_mapping = [x for x in tickers_not_in_universe if x not in all_tickers['Ticker(old)']]
        # valid_tickers = all_tickers[~all_tickers['Delisted'].isin(['Y'])]
        # valid_tickers_dict = all_tickers.set_index('Ticker(old)')[['Ticker']].to_dict()['Ticker']
        #
        # raw_og_ticker = DL.loadDB('raw_og_ticker_new.csv')
        # tickers = raw_og_ticker['Ticker'].unique()
        # tickers_valid = [x for x in tickers if x not in delisted['Ticker(old)'].values]
        # # tickers_tbu = [x for x in tickers if x in valid_tickers['Ticker(old)'].values]
        #
        files = os.listdir(f'{DATABASE_PATH}/Daily')
        files = [x.rstrip('.csv') for x in files]
        # tickers_tbu = [valid_tickers_dict[x] for x in tickers_valid if x not in files and valid_tickers_dict[x] not in files]

        # gs_raw = DL.loadDB('GS_raw.csv')
        # gs_sentiment = DL.loadDB('GS_sentiment.csv')
        # price_df = DL.loadDB('price_df.csv')
        #         #
        #         # _, valid_tickers, _ = DL.loadTickers()
        #         # valid_tikers_dict = valid_tickers.set_index('Ticker(old)').to_dict()
        #         # valid_tickers_ticker_dict_bbg = valid_tickers_dict['Ticker(BBG)']
        #         #
        #         # gs_raw['Ticker'] = gs_raw['Ticker'].replace(valid_tickers_ticker_dict_bbg)
        #         # gs_sentiment['Ticker'] = gs_sentiment['Ticker'].replace(valid_tickers_ticker_dict_bbg)
        #         # price_df['Ticker'] = price_df['Ticker'].replace(valid_tickers_ticker_dict_bbg)
        #         #
        #         # DL.toDB(gs_raw, 'GS_raw_upload.csv')
        #         # DL.toDB(gs_sentiment, 'GS_sentiment_upload.csv')
        #         # DL.toDB(price_df, 'price_df_upload.csv')

