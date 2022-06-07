from uti import timeit, DataLoader, Logger
DL = DataLoader()
logger = Logger()
tpc_df_combined = DL.loadDB()
from Database.price_df import GSPriceDf

GSP = GSPriceDf
GSP = GSPriceDf()
GSP.get_valid_tickers_dict()
GSP._valid_tickers_ticker_dict_bbg = GSP._valid_tickers_dict['Ticker(BBG)']

GSP._valid_tickers_dict

GSP._valid_tickers_dict.keys()
valid_tickers = DL.loadTickers()
ticker_universe = DL.loadDB('ticker_universe.csv')
ticker_universe = DL.loadDB('ticker\ universe.csv')
ticker_universe = DL.loadDB('ticker universe.csv')
ticker_universe = DL.loadDB('tickers universe.csv')
ticker_universe_dict = ticker_universe.set_index('Ticker').to_dict()
ticker_universe_dict = ticker_universe.set_index('Ticker').to_dict()['Ticker(BBG)']
valid_tickers.set_index('Ticker')

valid_tickers = valid_tickers.set_index('Ticker')
ticker_universe

ticker_universe[ticker_universe['Ticker'].isin(valid_tickers.index)]
ticker_universe[ticker_universe['Ticker(old)'].isin(valid_tickers.index)]

valid_tickers = valid_tickers.reset_index()

import pandas as pd
citi_no_intersect1 = valid_tickers[~valid_tickers['Ticker'].isin(ticker_universe['Ticker'])]
citi_no_intersect2 = valid_tickers[~valid_tickers['Ticker(old)'].isin(ticker_universe['Ticker'])]
citi_no_intersect = pd.concat([citi_no_intersect2, citi_no_intersect1], axis=0).drop_duplicates(['Ticker'])

citi_no_intersect[~citi_no_intersect['Ticker'].isin(ticker_universe['Ticker(old)'])]['Ticker']