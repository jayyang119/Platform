import os
import time
import glob
import pandas as pd
import numpy as np
import torch
import sys

from uti import timeit, DataLoader, Logger
from Crawler import gs_get_page_data
from Path import ONEDRIVE_PATH, DATABASE_PATH

from datetime import datetime
from bs4 import BeautifulSoup
from progress.bar import ChargingBar

print('Current working directory', os.getcwd())
DL = DataLoader()
logger = Logger()


@timeit
def update_sentiment(headline, base_path=os.path.join(ONEDRIVE_PATH, 'finBERT')):
    if ONEDRIVE_PATH not in sys.path:
        sys.path.append(ONEDRIVE_PATH)
    from finBERT import get_sentiments_finbert  # Temporarily, exploring a new way to import from outer project directory

    if ONEDRIVE_PATH not in sys.path:
        sys.path.append(ONEDRIVE_PATH)
    torch.cuda.empty_cache()
    sentiment_headlines = get_sentiments_finbert(headline, base_path)
    return sentiment_headlines


class GSDatabase:
    def __init__(self):
        pass

    def _gs_rewrite_database(self, path):
        writefile_path = os.path.join(path, 'output.csv')
        # GS_raw = DL.loadDB('GS_raw.csv')
        writefile = open(writefile_path, 'w', encoding='utf-8')
        writefile.write('Date,Time,Ticker,Source,Asset Class,Headline,Summary,Analysts\n')

        # writefile = DL.loadDB(os.Path.join(Path, 'output.csv'))
        dates_range = os.listdir(path)
        logger.info(str(len(dates_range)))
        print(dates_range)
        bar_name = f'Rewriting database {path}...'
        bar = ChargingBar("{0:<38}".format(bar_name), max=len(dates_range))
        try:
            for date in dates_range:  # latest_dates: a list of dates in string
                if '.csv' not in date and date >= '20180801':
                    files = glob.glob(os.path.join(path, date) + '/*.json')
                    for file_name in files:
                        if '.csv' not in file_name:
                            with open(file_name, encoding='utf-8') as f:
                                soup = BeautifulSoup(f, 'html.parser')
                                for row in soup.find_all('tr')[1:]:
                                    writeline = gs_get_page_data(row)
                                    # logger.info(writeline)
                                    # writefile = pd.concat([writefile, pd.DataFrame(writeline)], axis=0)
                                    writefile.write(','.join(writeline))
                                    writefile.write('\n')
                bar.next()
                time.sleep(0.000001)
        except Exception as e:
            logger.error(e)
        writefile.close()
        bar.finish()
        return

    def _get_writefile(self, field='raw'):
        if field == 'raw':
            if DL.checkDB('GS_raw.csv'):
                writefile = DL.loadDB('GS_raw.csv', parse_dates=['Date', 'Time'])
                return writefile
            else:
                raise Exception('GS_raw.csv unexists in database.')

        elif field == 'sentiment':
            if DL.checkDB('GS_sentiment.csv'):
                writefile = DL.loadDB('GS_sentiment.csv', parse_dates=['Date', 'Time'])
                writefile['Summary'] = writefile['Summary'].fillna('')
                # if 'Earnings' not in writefile.columns:
                #     writefile = gs_get_report_type(writefile)
                return writefile, None
            else:
                logger.info('Regenerate headline sentiments')
                assert DL.checkDB('GS_raw.csv'), 'Please check raw data file.'
                new_df = DL.loadDB('GS_raw.csv', parse_dates=['Date', 'Time'])
                writefile = pd.DataFrame([], columns=new_df.reset_index().columns)
                return writefile, new_df
        else:
            raise Exception('Please check input, valid input: raw/sentiment')

    def _update_date_range(self, writefile):
        """
        Return the index of the dates needed to be updated
        """
        # date_min = min(writefile['Date'])
        files = glob.glob(f'{DATABASE_PATH}/Goldman Reports/*')
        files_datetime = [datetime.strptime(os.path.basename(_), '%Y%m%d') for _ in files]
        latest_record = max(files_datetime)  # datetime.strptime(max([os.Path.basename(x) for x in files]), '%Y%m%d')
        second_latest_record = max([x for x in files_datetime if x != latest_record])
        earliest_record = min(files_datetime)  # datetime.strptime(min([os.Path.basename(x) for x in files]), '%Y%m%d')

        if len(writefile) == 0:
            return pd.date_range(earliest_record, latest_record)

        # date_range = pd.date_range(date_min, latest_record)
        date_new = list(set(files_datetime) - set(writefile['Date']))
        if latest_record not in date_new:
            date_new.extend([latest_record])
        if second_latest_record not in date_new:
            date_new.extend([second_latest_record])

        logger.info('Date range needed to be updated:')
        # logger.info(date_new)
        date_range = pd.date_range(min(date_new), max(date_new))
        print(date_range)

        # userinput = input('Continue? (y/n)')
        # if userinput.lower() in ['n']:
        #     return

        return date_range


    def _update_gs_raw_new_df(self, date_range):
        new_df = np.array([])

        for dt in date_range:
            files = glob.glob(
                os.path.join(f'{DATABASE_PATH}/Goldman Reports', datetime.strftime(dt, '%Y%m%d')) + '/*.json')
            for file_name in files:
                with open(file_name, encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    for row in soup.find_all('tr')[1:]:
                        writeline = gs_get_page_data(row, 'list')
                        new_df = np.append(new_df, writeline)

        new_df = new_df.reshape(-1, 8)
        new_df = pd.DataFrame(new_df, columns=['Date', 'Time', 'Ticker', 'Source', 'Asset Class', 'Headline', 'Summary',
                                               'Analysts'])
        new_df['Headline'] = new_df['Headline'].fillna('')
        new_df['Summary'] = new_df['Summary'].fillna('')
        new_df = new_df.sort_values(['Time'], ascending=False)
        new_df['Date'] = new_df['Date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d'))
        new_df['Time'] = new_df['Time'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))

        return new_df

    def _update_gs_raw(self):
        writefile = self._get_writefile('raw')
        date_range = self._update_date_range(writefile)

        new_df = self._update_gs_raw_new_df(date_range)

        new_df = pd.concat([new_df, writefile[writefile['Date'].isin(date_range)]], axis=0)
        new_df.drop_duplicates(['Headline'], inplace=True)

        writefile = pd.concat([new_df, writefile[~writefile['Date'].isin(date_range)]], axis=0)
        writefile['Summary'] = writefile['Summary'].fillna('')
        writefile.drop_duplicates(['Headline'], inplace=True)
        writefile = writefile.sort_values(['Time'], ascending=False)
        DL.toDB(writefile, 'GS_raw.csv', index=None)
        print("%s: %s" % (datetime.strftime(datetime.now(), "%H:%M:%S"), 'GS_raw completed.'))
        return writefile

    # Part2: Update GS_sentiment.csv
    def _update_gs_sentiment(self, readfile, **kwarg):
        writefile, new_df = self._get_writefile('sentiment')
        date_range = self._update_date_range(writefile)
        if new_df is None:
            new_df = readfile[readfile['Date'].isin(date_range)].copy(deep=True)

        new_df['Headline sentiment'], headline_sentiments_score = update_sentiment(new_df['Headline'].to_list())
        new_df['Headline neutral score'] = headline_sentiments_score[:, 0]
        new_df['Headline positive score'] = headline_sentiments_score[:, 1]
        new_df['Headline negative score'] = headline_sentiments_score[:, 2]

        new_df['Summary sentiment'], summary_sentiments_score = update_sentiment(new_df['Summary'].to_list())
        new_df['Summary neutral score'] = summary_sentiments_score[:, 0]
        new_df['Summary positive score'] = summary_sentiments_score[:, 1]
        new_df['Summary negative score'] = summary_sentiments_score[:, 2]

        if 'Analysts' in new_df.columns:
            new_df['Head analyst'] = new_df['Analysts'].apply(lambda x: x.split('|')[0].strip())

        writefile = pd.concat([new_df, writefile[~writefile['Date'].isin(date_range)]], axis=0)
        writefile.drop_duplicates(inplace=True)
        writefile = writefile.sort_values(['Time'], ascending=False)

        DL.toDB(writefile, 'GS_sentiment.csv', index=None)

        print("%s: %s" % (datetime.strftime(datetime.now(), "%H:%M:%S"), 'GS_sentiment completed.'))
        return writefile

    def GS_get_raw(self):

        return DL.loadDB('GS_raw.csv', parse_dates=['Date', 'Time'])

    @timeit
    def GS_update_raw_and_sentiment(self, file='raw'):
        logger.info('Updating database')

        raw_df = self._update_gs_raw()
        sentiment_df = self._update_gs_sentiment(raw_df)
        return sentiment_df

    @timeit
    def GS_update_sentiment(self, file='sentiment'):
        logger.info('Updating database')

        raw_df = DL.loadDB('GS_raw.csv', parse_dates=['Date', 'Time'])
        sentiment_df = self._update_gs_sentiment(raw_df)
        return sentiment_df

    @timeit
    def GS_rewrite_database(self, path):
        df = self._gs_rewrite_database(path)
        return df

if __name__ == '__main__':
    GSD = GSDatabase()
    GSD.GS_update_sentiment()
