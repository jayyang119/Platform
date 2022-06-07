import pandas as pd
from uti import DataLoader, Logger
from Broker import ricsregion

DL = DataLoader()
logger = Logger()


def tickers_reconcilation():
    GS_universe = DL.loadDB('GS stock universe coverage.csv')
    GS_universe.rename(columns={'Ticker': 'Ticker(old)'}, inplace=True)
    GS_universe['Cover'] = 'Y'
    GS_universe['CL'].replace({'No': 'N', 'Yes': 'Y'})

    GS_universe_dict = GS_universe.set_index('Ticker(old)').to_dict()

    tickers_df = DL.loadDB('Tickers.csv')
    # tickers_df['Ticker(BBG)'] = tickers_df['Ticker(old)'].replace(GS_universe_dict['BBG Ticker'])
    # tickers_df['Industry'] = tickers_df['Ticker(old)'].replace(GS_universe_dict['Industry'])
    # tickers_df['Company'] = tickers_df['Ticker(old)'].replace(GS_universe_dict['Company'])
    tickers_df = pd.merge(tickers_df, GS_universe[['Ticker(old)', 'BBG Ticker', 'Company', 'Industry', 'Cover', 'CL']],
                          on='Ticker(old)', how='outer')
    tickers_df.rename(columns={'BBG Ticker': 'Ticker(BBG)'}, inplace=True)

    tickers_df = tickers_df[['Ticker(old)', 'Ticker', 'Ticker(BBG)', 'Sector', 'Industry', 'MarketCap',
                             'Currency', 'Region', 'Delisted', 'Cover', 'CL']]

    tickers_df = ricsregion(tickers_df)
    DL.toDB(tickers_df, 'testing_tickers.csv')

if __name__ == '__main__':
    tickers_reconcilation()


