import argparse

from uti import DataLoader, Logger
from Database import GSDatabase
from Crawler import GS_crawler
from Database import GSPriceDf
from datetime import datetime

GSD = GSDatabase()
logger = Logger()
DL = DataLoader()

if __name__ == '__main__':
    ######################## For debugging ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='crawler')
    parser.add_argument('--mode', type=str, default='stable')

    parser.add_argument('--day', type=int, default=1)
    args = parser.parse_args()

    MODE = args.mode
    TASK = args.task
    DAYS = args.day
    if MODE == 'beta':
        DL = DataLoader(mode='beta')
    ###########################################################################
    if TASK == 'crawler':
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
