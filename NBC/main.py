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
    parser.add_argument('--update', type=str, default='True')
    parser.add_argument('--mode', type=str, default='stable')
    parser.add_argument('--day', type=int, default=1)
    args = parser.parse_args()

    MODE = args.mode
    UPDATE = args.update
    TASK = args.task
    DAYS = args.day
    if MODE == 'beta':
        DL = DataLoader(mode='beta')
    ###########################################################################

    if TASK == 'database_update':
        GSD.GS_update_sentiment(update=bool(UPDATE))

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
