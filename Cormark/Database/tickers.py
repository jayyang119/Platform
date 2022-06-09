import pandas as pd
from uti import DataLoader, Logger
from Broker import ricsregion
from Eikon import Eikon_update_price_enhanced
from Crawler import REPORT_TYPE_DICT

DL = DataLoader()
logger = Logger()


if __name__ == '__main__':
    price_df = DL.loadDB('price_df.csv')
    price_df = DL.loadDB('citi0317.csv')

    # citi_raw = DL.loadDB('citi_raw.csv')
    # citi_raw_headline_rt_dict = citi_raw.set_index('headline').loc[price_df['Headline']]['broker_lable'].to_dict()
    # price_df['Report Type'] = price_df['Headline'].replace(citi_raw_headline_rt_dict)
    #
    def define_report_type(s):
        try:
            for tag, rt in REPORT_TYPE_DICT.items():
                if tag in s.split(','):
                    return rt
        except:
            return 'ad-hoc'
        return 'ad-hoc'
    #
    from Database.database import update_sentiment

    price_df['Report Type'] = price_df['Report Type'].apply(lambda x: define_report_type(x))
    price_df = price_df.dropna(subset=['Ticker'])
    price_df['Ticker'] = price_df['Ticker'].apply(lambda x: x.split(',')[0])
    price_df['Headline sentiment'], headline_sentiments_score = update_sentiment(price_df['Headline'])
    price_df['Summary sentiment'], summary_sentiments_score = update_sentiment(price_df['Summary'])

    DL.toDB(price_df, 'sentiment_20220317.csv')
    # DL.toDB(price_df, 'price_df.csv')

    import pandas as pd
    # universe = DL.loadDB('universe.csv')
    # from Broker import ricsregion
    # universe = ricsregion(universe)
    # print(universe)
    # DL.toDB(universe, 'universe.csv')


    # universe.loc[universe['Region'] == 'Japan', 'Ticker'] += '.T'
    # DL.toDB(universe, 'universe.csv')
    # raw = DL.loadDB('raw_og_ticker_new.csv')
    # raw_valid = raw[raw['Ticker'].isin(universe['Ticker'])]
    # raw_invalid = raw[~raw['Ticker'].isin(universe['Ticker'])]
    # raw_valid_tickers = raw_valid['Ticker'].unique()
    # tickers_diff = set(raw_valid_tickers).difference(set(universe['Ticker(old)']))
    # not_in_old_tick = [x for x in tickers_diff if x not in universe['Ticker(old)'].values]
    #
    # import os
    # files = os.listdir(f'{DL.database_path}/Daily')
    # files = [x.rstrip('.csv') for x in files]
    #
    # not_in_daily = [x for x in raw_valid_tickers if x not in files]

    # import re
    # from bs4 import BeautifulSoup
    # import pandas as pd
    # citi_universe_file = open(f'{DL.database_path}/Citi universe.json').readlines()
    # citi_universe = BeautifulSoup(citi_universe_file[0], parser='html.parser', features="lxml")
    # rows = citi_universe.find_all('div', class_='rowline')
    #
    # universe_df = pd.DataFrame(columns=['Company Name', 'Ticker', 'Analyst Name', 'Sector', 'Region'])
    #
    # for row in rows:
    #     divs = row.find_all('div')
    #     try:
    #         company_name = divs[0].text
    #         analyst_name = divs[1].text
    #         sector = divs[2].text
    #         region = divs[3].text
    #         ticker = re.findall(r'\((.*?)\)', company_name)[0]
    #
    #         row_df = pd.DataFrame({'Company Name': [company_name], 'Ticker': [ticker], 'Analyst Name': [analyst_name],
    #                                'Sector': [sector], 'Region': [region]})
    #         universe_df = universe_df.append(row_df)
    #     except Exception as e:
    #         logger.error(row.text)
    #
    # DL.toDB(universe_df, 'universe.csv')
    #
    #
