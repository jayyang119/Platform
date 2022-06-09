from uti import timeit, DataLoader, Logger, UrlManager, By
from Path import ONEDRIVE_PATH
from collections import defaultdict
from Database.report_type import rc_filter, er_filter, io_filter, ec_filter
from Database.rating_change import tpc_scanner, tpc_prev_scanner, rating_scanner

import os
import time
import pandas as pd
import numpy as np
import torch
import sys
import re
from bs4 import BeautifulSoup

print('Current working directory', os.getcwd())

DL = DataLoader()
logger = Logger()
UM = UrlManager()

@timeit
def update_sentiment(headline, base_path=os.path.join(ONEDRIVE_PATH, 'finBERT')):
    """
        This function clears gpu memory cache and returns the finBERT's predicted sentiments given a list of sentences.
    """
    if ONEDRIVE_PATH not in sys.path:
        sys.path.append(ONEDRIVE_PATH)
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

    def cormark_login(self, driver, need_un=True):
        """
            This function auto login Cormark platform with credentials.
        """
        username = 'tsang@a-s-capital.com'
        password = 'ytf73bei06'
        login_form = driver.find_elements(By.XPATH, f'//input[@class="form-control"]')

        if len(login_form) > 0:
            login_form_username = login_form[0]
            login_form_password = login_form[1]

            if need_un is True:
                login_form_username.send_keys(username)
                login_form_password.send_keys(password)

        UM.navigate(driver, xpath=f'//div/input[@type="submit"]', move=True, click=True)
        state = 'n'
        while state == 'n':
            state = input('Please press enter if login successful. [y/n]')

    def clean_cormark_ticker(self, ticker: str):
        """
            This function cleans tickers from Cormark platform html.
        """
        ticker = ticker.replace('-CA', ' CN')
        ticker = ticker.replace('-US', ' US')

        if '.' in ticker:
            ticker = ticker.replace('.UN', '-U')
            ticker = ticker.replace('.UT', '-U')
            ticker = ticker.replace('.USD', '/U')
            ticker = ticker.replace('.', '/')

        return ticker

    def crawler(self, driver):
        """
            This function crawls report information from NBC's report page.
        """
        uid_pattern = re.compile(r'pdf\/(.*)\?')

        # Need to get page length:
        result = defaultdict(list)
        last_page_button = driver.find_elements(By.XPATH, '//li[@class="paginate_button page-item "]')
        if len(last_page_button) > 0:
            last_page_button = last_page_button[-1]
            total_pages = int(last_page_button.text)
        else:
            total_pages = 1

        for i in range(total_pages):
            content = UM.navigate(driver, xpath='//table[@id="searchDetails"]')
            trs = content.find_elements(By.XPATH, '//tr[@role="row"]')[1:]

            for tr in trs:
                tr_soup = BeautifulSoup(tr.get_attribute('innerHTML'), 'html.parser')
                tds = tr_soup.find_all('td')

                result['headline'].append(tds[0].find('a').text)
                result['url'].append(tds[0].find('a')['href'])
                ticker = self.clean_cormark_ticker(tds[1].find('a').text)
                result['ticker'].append(ticker)
                result['analyst_pri'].append(tds[2].find('a').text)
                result['broker_label'].append(tds[3].text.strip('\n '))
                result['publish_date_and_time'].append(tds[4].text.strip('\n '))

            next_button = driver.find_elements(By.XPATH, '//li[@class="paginate_button page-item next"]')
            if len(next_button) > 0:
                UM.navigate(driver, xpath='//li[@class="paginate_button page-item next"]', move=True, click=True)

        df = pd.DataFrame(result)
        for i in range(len(df)):
            driver.get(df.at[i, 'url'].replace('pdf', 'html'))
            time.sleep(0.1)

            trs = driver.find_elements(By.XPATH, '//td/table/tbody')[1]
            soup = BeautifulSoup(trs.get_attribute('innerHTML'), 'html.parser')

            summary = ' '.join([x.text for x in soup.find_all('p')[1:]])
            df.at[i, 'summary'] = summary

            uid = uid_pattern.findall(df.at[i, 'url'])[0]
            df.at[i, 'uid'] = uid

            time.sleep(0.002)
        df['report_type'] = ''

        return df

    def update_sentiment_df(self, df):
        """
            This function updates sentiment, backfills the target prices, ratings of df, and
            define report type based on regex search.
        """
        for senti_col in ['headline_senti', 'summary_senti']:
            print(senti_col)
            senti_empty_mask = df[senti_col].isna()
            senti = senti_col.replace('_senti', '')
            df.loc[senti_empty_mask, senti_col], _ = update_sentiment(
                df.loc[senti_empty_mask, senti].fillna(''))

        # Update target prices, ratings, rating changes
        TPC_mask = df['tp_curr'].isna()
        TPC_prev_mask = df['tp_prev'].isna()
        RATING_mask = df['rating_curr'].isna()
        if TPC_mask.any():
            tpc_list = tpc_scanner(df[TPC_mask]['summary'])
            df.loc[TPC_mask, 'tp_curr'] = tpc_list
        if TPC_prev_mask.any():
            tpc_list = tpc_prev_scanner(df[TPC_mask]['summary'])
            df.loc[TPC_prev_mask, 'tp_prev'] = tpc_list
        if RATING_mask.any():
            rating_list = rating_scanner(df.loc[RATING_mask, 'summary'], df.loc[RATING_mask, 'ticker'])
            df.loc[RATING_mask, 'rating_curr'] = rating_list

        df = pd.concat([df.loc[TPC_mask|RATING_mask|TPC_prev_mask, df.columns], df], axis=0). \
            sort_values('publish_date_and_time', ascending=False). \
            drop_duplicates(['publish_date_and_time', 'uid'], keep='first').reset_index(drop=True)
        df = df.sort_values(['publish_date_and_time'], ascending=True).reset_index(drop=True)

        # Clean target prices mannually with Artificial Intelligence
        for tp_tbu in ['tp_curr', 'tp_prev']:
            tp_tbu_mask = df[tp_tbu].fillna('').str.contains('\[').fillna(False)
            for i, row in df[tp_tbu_mask].iterrows():
                print(row['ticker'])
                print(row['summary'], '\n')
                tp = input(
                    f'Please insert the real {tp_tbu} mannually, thanks. If no target price this is a sector report for multiple tickers just press Enter...')
                if len(tp) != 0:
                    df.loc[i, tp_tbu] = float(tp)
                else:
                    df.loc[i, tp_tbu] = ''

        df['tp_curr'] = df.groupby(['ticker'])['tp_curr'].apply(lambda x: x.fillna(method='ffill'))
        tpc_mask = (df['tp_curr'] > df['tp_prev']) | (df['tp_curr'] < df['tp_prev'])
        df.loc[tpc_mask, 'tp_chg'] = df.loc[tpc_mask, 'tp_curr'] - df.loc[tpc_mask, 'tp_prev']
        df['tp_chg_pct'] = df['tp_chg'] / df['tp_curr']
        df['rating_curr'] = df.groupby(['ticker'])['rating_curr'].apply(lambda x: x.fillna(method='ffill'))
        df['rating_prev'] = df.groupby(['ticker'])['rating_curr'].apply(lambda x: x.shift())

        # Define report types
        er_mask = er_filter(df['headline'], df['summary'])
        rc_mask = (df['rating_curr'] != df['rating_prev']) &\
                  (~df['rating_curr'].isna()) & (~df['rating_prev'].isna())
        io_mask = io_filter(df['headline'], df['summary'])
        ec_mask = ec_filter(df['headline'], df['summary'])

        df['report_type'] = 'ad-hoc'
        df['report_type'] = np.where(er_mask, 'Earning\'s Review', df['report_type'])
        df['report_type'] = np.where(ec_mask, 'Estimate Change', df['report_type'])
        df['report_type'] = np.where(io_mask, 'Initiation', df['report_type'])
        df['report_type'] = np.where(tpc_mask, 'Target Price Change', df['report_type'])
        df['report_type'] = np.where(rc_mask, 'Rating Change', df['report_type'])

        return df


    @timeit
    def GS_update_sentiment(self, update=True):
        """
            This function initiates the platform crawler and updates sentiment data.
        """
        logger.info('Updating database')
        sentiment_df = DL.loadDB('Cormark sentiment.csv', parse_dates=['publish_date_and_time'])
        if len(sentiment_df) == 0:
            sentiment_df = pd.DataFrame()

        if not update:
            self.update_sentiment_df(sentiment_df)
            DL.toDB(sentiment_df, f'Cormark sentiment.csv')
        else:
            driver = UM.start_browser()
            UM.get_url(driver, 'https://cormark-research.bluematrix.com/client/library.jsp?mode=search')
            self.cormark_login(driver)

            status = 'y'
            while status == 'y':
                print('Please navigate to Search page, and insert correct dates and report types:\n'
                      'Research Report (Issuer), Special Alert (Issuer), Emerging Ideas (Issuer), Morning Note (Issuer).')
                status1 = 'n'
                while status1 != 'y':
                    status1 = input('Continue? [y/n]').lower()
                searchweb_url = driver.current_url
                df = self.crawler(driver)
                df[['headline_senti', 'summary_senti']] = None
                df[['tp_curr', 'tp_prev', 'tp_chg', 'tp_chg_pct']] = None
                df[['rating_prev', 'rating_curr']] = None
                df[['report_type']] = ''
                DL.toDB(df, 'new_df.csv')
                df = DL.loadDB('new_df.csv', parse_dates=['publish_date_and_time'])

                if len(sentiment_df) == 0:
                    sentiment_df = df.copy()
                else:
                    sentiment_df = pd.concat([sentiment_df, df[sentiment_df.columns]], axis=0).drop_duplicates(
                        ['publish_date_and_time', 'uid'], keep='first').reset_index(drop=True)

                sentiment_df = self.update_sentiment_df(sentiment_df)
                DL.toDB(sentiment_df, f'Cormark sentiment.csv')
                UM.get_url(driver, searchweb_url)
                status = input('Continue scraping (Please refresh the search page)? [y/n]').lower()


if __name__ == '__main__':
    GSD = GSDatabase()
    GSD.GS_update_sentiment(update=True)
    # GSD.GS_update_sentiment(update=False)
