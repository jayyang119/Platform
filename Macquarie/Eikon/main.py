import argparse

from uti import DataLoader, Logger
from downloader import Eikon_update_price_enhanced
from tickers import Eikon_check_tickers, Eikon_update_tickers

DL = DataLoader()
logger = Logger()
DATABASE_PATH = DL.database_path

if __name__ == '__main__':

    ######################## For debugging ########################
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='crawl')
    parser.add_argument('--task', type=str, default='price_update')
    # parser.add_argument('--task', type=str, default='tickers_update')
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--daily', type=bool, default=True)
    args = parser.parse_args()

    TASK = args.task
    MODE = args.mode
    DAILY = args.daily

    if TASK == 'price_update':
        all_tickers = DL.loadTickers()
        all_tickers = all_tickers[~all_tickers['ISIN'].isna()]
        if MODE == 'all':
            Eikon_update_price_enhanced(all_tickers['Ticker'])

    elif TASK == 'tickers_update':
        data_df, need_update = Eikon_update_tickers()
    elif TASK == 'tickers_check':
        Eikon_check_tickers()