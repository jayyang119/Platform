import os
import pandas as pd
import numpy as np
import glob
import json
import torch
import sys

from Database.settings import REPORT_TYPE_DICT
from uti import timeit, DataLoader, Logger
from Path import DATABASE_PATH, ONEDRIVE_PATH, BASE_PATH, FINBERT_PATH
from Database.rating_change import tpc_scanner, rc_scanner, rating_scanner
from flatten_json import flatten_json
from datetime import timedelta

print('Current working directory', os.getcwd())

DL = DataLoader()
logger = Logger()


@timeit
def update_sentiment(headline, base_path=os.path.join(FINBERT_PATH, 'finBERT')):
    """
        This function clears gpu memory cache and returns the finBERT's predicted sentiments given a list of sentences.
    """
    if FINBERT_PATH not in sys.path:
        sys.path.append(FINBERT_PATH)
    from finBERT import get_sentiments_finbert  # Temporarily, exploring a new way to import from outer project directory

    torch.cuda.empty_cache()
    sentiment_headlines = get_sentiments_finbert(headline, base_path)
    return sentiment_headlines


class GSDatabase:
    """
        This class summarizes necessary functions to update the GS_raw.csv and GS_sentiment.csv.
    """
    def __init__(self):
        pass

    def define_report_type(self, s):
        try:
            for tag, rt in REPORT_TYPE_DICT.items():
                if tag in s.split(','):
                    return rt
        except:
            return 'ad-hoc'
        return 'ad-hoc'

    def json_to_df(self, path):
        files = glob.glob(path)
        citi = []
        for file in files:
            with open(file, encoding='utf8') as f:
                try:
                    data = json.load(f)
                    data = data['list']
                    data = [flatten_json(each) for each in data]
                    df1 = pd.DataFrame(data)
                    citi.append(df1)
                except Exception as e:
                    f.close()
                    logger.info(file)
                    logger.error(e)

        df = pd.concat(citi).reset_index(drop=True)
        df = df[['headLine', 'pubDate', 'OBOPreferredName', 'synopsis',
                 'tickers', 'regions', 'sectors', 'company', 'assetClass',
                 'subject', 'pubId']]
        df = df.rename(columns={"pubDate": "Time",
                                "OBOPreferredName": "analyst_pri",
                                "synopsis": "summary",
                                "tickers": "tickers",
                                "subject": "report_types",
                                "headLine": "headline",
                                "regions": "Region",
                                "sectors": "Industry"
                                })

        df = self.handle_edt(df)
        df = df.drop_duplicates(subset=['pubId']).reset_index(drop=True)
        df = df.dropna(subset=['Industry']).reset_index(drop=True)
        df['company'] = df['company'].fillna('').apply(lambda x: x.split(',')[0])
        df['Region'] = df['Region'].fillna('').apply(lambda x: x.split(',')[0])
        df['Industry'] = df['Industry'].fillna('').apply(lambda x: x.split(',')[0])
        df['report_type'] = df['report_types'].apply(lambda x: self.define_report_type(x))
        df = df.dropna(subset=['tickers'])
        df['headline_senti'] = None
        df['summary_senti'] = None
        df['ticker'] = df['tickers'].apply(lambda x: x.split(',')[0])
        df.loc[df['Region'] == 'Japan', 'ticker'] = df.loc[df['Region'] == 'Japan', 'ticker'].apply(lambda x: x+'.T' if len(x) == 4 else x)  # Japan tickers

        return df


    @timeit
    def GS_update_sentiment(self, update=True):
        """
            This function update the sentiment data
            1. Sort by Ticker, Date
            2. Use regex search to fill in target prices, ratings
            3. Backfill target prices, ratings
        """

        logger.info('Updating database')
        sentiment_df = DL.loadDB('Citi sentiment.csv', parse_dates=['Date', 'Time'])
        if update:
            path = f"{DATABASE_PATH}/*.json"
        else:
            path = f"{DATABASE_PATH}/citi json/*.json"
        df = self.json_to_df(path)
        df[['TPS_new', 'tp_curr', 'tp_prev', 'tp_chg', 'tp_chg_pct', 'abs(tp_chg)', 'RC_upgrade', 'RC_downgrade', 'rating_new']] = None
        df[['rating_prev', 'rating_curr', 'RC_upgrade', 'RC_downgrade']] = ''
        if len(sentiment_df) == 0:
            new_df = df.copy()
        else:
            sentiment_df = pd.concat([sentiment_df, df[sentiment_df.columns]], axis=0)
            sentiment_df_duplicated = sentiment_df[sentiment_df.duplicated(['Time', 'summary'], keep=False)].copy()
            sentiment_df = sentiment_df[~sentiment_df.duplicated(['Time', 'summary'], keep=False)].copy()
            sentiment_df_duplicated_updated = sentiment_df_duplicated[~sentiment_df_duplicated['rating_new'].isna() | ~sentiment_df_duplicated['TPS_new'].isna()]
            sentiment_df_duplicated = pd.concat([sentiment_df_duplicated_updated, sentiment_df_duplicated]).drop_duplicates(['Time', 'summary'], keep='first')
            sentiment_df = pd.concat([sentiment_df, sentiment_df_duplicated], axis=0).reset_index(drop=True)

            sentiment_df['summary'] = sentiment_df['summary'].fillna('')
            sentiment_df.loc[sentiment_df['headline_senti'].isna(), 'headline_senti'], _ = update_sentiment(sentiment_df.loc[sentiment_df['headline_senti'].isna(), 'headline'])
            sentiment_df.loc[sentiment_df['summary_senti'].isna(), 'summary_senti'], _ = update_sentiment(sentiment_df.loc[sentiment_df['summary_senti'].isna(), 'summary'])

            new_df = sentiment_df.loc[sentiment_df['rating_curr'].isna() | sentiment_df['tp_curr'].isna()].reset_index(drop=True).copy()

        if len(new_df) > 0:
            # Update sentiment, target prices, ratings, rating changes
            TPC_mask = new_df['report_types'].fillna('').str.contains('Target Price Change') & new_df['TPS_new'].isna()
            if TPC_mask.any():
                tpc_list = tpc_scanner(new_df[TPC_mask].reset_index()['summary'])
                new_df.loc[TPC_mask, 'TPS_new'] = tpc_list

            # Rating change
            RC_mask = new_df['report_types'].fillna('').str.contains('Rating Change') & ~new_df['tickers'].fillna('').str.contains(',') & new_df['rating_new'].isna()
            if RC_mask.any():
                rc_list, rc_list_real = rc_scanner(new_df[RC_mask]['summary'])
                new_df.loc[RC_mask, 'RC_upgrade'] = np.array(rc_list)[:, 0]
                new_df.loc[RC_mask, 'RC_downgrade'] = np.array(rc_list)[:, 1]
                new_df.loc[RC_mask, 'rating_new'] = rc_list_real

            # RE search for current ratings
            RATING_mask = ~new_df['report_types'].fillna('').str.contains('Rating Change') & ~new_df['tickers'].fillna('').str.contains(',') & new_df['rating_new'].isna()
            if RATING_mask.any():
                rating_list = rating_scanner(new_df[RATING_mask]['summary'], new_df[RATING_mask]['tickers'])
                new_df.loc[RATING_mask, 'rating_new'] = rating_list

        logger.info(new_df)
        # Backfill rating_prev with previous rating_curr
        sentiment_df = pd.concat([new_df[sentiment_df.columns], sentiment_df], axis=0).sort_values(['Time'], ascending=False).drop_duplicates(['Time', 'summary'], keep='first')
        sentiment_df = sentiment_df.sort_values(['Time'], ascending=True).reset_index(drop=True)
        sentiment_df['rating_curr'] = sentiment_df.groupby(['ticker'])['rating_new'].apply(lambda x: x.fillna(method='ffill'))
        sentiment_df['rating_prev'] = sentiment_df.groupby(['ticker'])['rating_curr'].apply(lambda x: x.shift())

        # Clean target prices mannually with Artificial Intelligence
        sentiment_df_tbu = sentiment_df[sentiment_df['TPS_new'].fillna('').str.contains('\[').fillna(False)].copy().reset_index(drop=True)
        sentiment_df = sentiment_df[~sentiment_df['TPS_new'].fillna('').str.contains('\[').fillna(False)]
        for i, row in sentiment_df_tbu.iterrows():
            print(row['ticker'], row['tickers'])
            print(row['summary'], '\n')
            tp = input(f'Please insert the real target prices mannually, thanks. If no target price this is a sector report for multiple tickers'
                        ' just press Enter...')
            if len(tp) != 0:
                print(tp)
                sentiment_df_tbu.loc[i, 'TPS_new'] = float(tp)
            else:
                sentiment_df_tbu.loc[i, 'TPS_new'] = ''
        sentiment_df = pd.concat([sentiment_df_tbu, sentiment_df], axis=0)

        sentiment_df = sentiment_df.sort_values(['Time'], ascending=True).reset_index(drop=True)
        sentiment_df['TPS_new'] = sentiment_df['TPS_new'].replace('', np.nan)
        sentiment_df['tp_curr'] = sentiment_df.groupby(['ticker'])['TPS_new'].apply(lambda x: x.fillna(method='ffill'))

        for col in ['tp_curr', 'tp_prev', 'tp_chg_pct', 'abs(tp_chg)']:
            sentiment_df[col] = sentiment_df[col].astype(float)
        sentiment_df['tp_prev'] = sentiment_df.groupby(['ticker'])['tp_curr'].apply(lambda x: x.shift())
        sentiment_df['tp_chg'] = sentiment_df.groupby(['ticker'])['tp_curr'].apply(lambda x: x - x.shift())
        sentiment_df['abs(tp_chg)'] = abs(sentiment_df['tp_chg'])
        sentiment_df['tp_chg_pct'] = sentiment_df['tp_chg'] / sentiment_df['tp_curr']
        sentiment_df['rating_new'] = sentiment_df['rating_new'].replace('', np.nan)
        sentiment_df['rating_curr'] = sentiment_df.groupby(['ticker'])['rating_new'].apply(lambda x: x.fillna(method='ffill'))
        sentiment_df['rating_prev'] = sentiment_df.groupby(['ticker'])['rating_curr'].apply(lambda x: x.shift())
        DL.toDB(sentiment_df, f'Citi sentiment.csv')

        return sentiment_df

    def handle_edt(self, sentiment):
        '''
            Clean time format with EDT, EST.
        '''
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
        return sentiment_new

if __name__ == '__main__':
    GSD = GSDatabase()
    GSD.GS_update_sentiment(update=True)
