from uti import timeit, DataLoader, Logger, UrlManager, By
from Path import ONEDRIVE_PATH
from collections import defaultdict
from Database.report_type import rc_filter, er_filter, io_filter, ec_filter
from Database.rating_change import tpc_scanner, rc_scanner, rating_scanner

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
    if ONEDRIVE_PATH not in sys.path:
        sys.path.append(ONEDRIVE_PATH)
    from finBERT import get_sentiments_finbert  # Temporarily, exploring a new way to import from outer project directory

    torch.cuda.empty_cache()
    sentiment_headlines = get_sentiments_finbert(headline, base_path)
    return sentiment_headlines


class GSDatabase:
    def __init__(self):
        pass

    def cormark_login(self, driver, need_un=True):
        username = 'tsang@a-s-capital.com'
        password = 'ytf73bei06'
        login_form = driver.find_elements(By.XPATH, f'//input[@class="form-control"]')
        #     pdb.set_trace()

        if len(login_form) > 0:
            login_form_username = login_form[0]
            login_form_password = login_form[1]

            if need_un is True:
                login_form_username.send_keys(username)
                login_form_password.send_keys(password)

        UM.navigate(driver, xpath=f'//div/input[@type="submit"]', move=True, click=True)
        state = ''
        while state == '':
            state = input('Please press enter if login successful.')

    def crawler(self, driver):
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
                result['ticker'].append(tds[1].find('a').text)
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

    @timeit
    def GS_update_sentiment(self, update=True):

        def update_sentiment_df(sentiment_df):
            senti_empty_mask = sentiment_df['headline_senti'].isna()
            senti_empty_mask = sentiment_df['summary_senti'].isna()
            # sentiment_df.loc[senti_empty_mask, 'headline_senti'], _ = update_sentiment(
            #     sentiment_df.loc[senti_empty_mask, 'headline'].fillna(''))
            # sentiment_df.loc[senti_empty_mask, 'summary_senti'], _ = update_sentiment(
            #     sentiment_df.loc[senti_empty_mask, 'summary'].fillna(''))

            new_df = sentiment_df.loc[sentiment_df['rating_new'].isna() | sentiment_df['tp_new'].isna()]. \
                reset_index(drop=True).copy()

            if len(new_df) > 0:
                # Update target prices, ratings, rating changes
                TPC_mask = new_df['tp_curr'].isna()
                if TPC_mask.any():
                    tpc_list = tpc_scanner(new_df[TPC_mask].reset_index()['summary'])
                    new_df.loc[TPC_mask, 'tp_new'] = tpc_list

                RATING_mask = new_df['rating_new'].isna()
                if RATING_mask.any():
                    rating_list = rating_scanner(new_df[RATING_mask]['summary'], new_df[RATING_mask]['ticker'])
                    new_df.loc[RATING_mask, 'rating_new'] = rating_list

            sentiment_df = pd.concat([new_df[sentiment_df.columns], sentiment_df], axis=0). \
                sort_values('publish_date_and_time', ascending=False). \
                drop_duplicates(['publish_date_and_time', 'uid'], keep='first').reset_index(drop=True)
            sentiment_df = sentiment_df.sort_values(['publish_date_and_time'], ascending=True).reset_index(drop=True)

            # Clean target prices mannually with Artificial Intelligence
            sentiment_df_tbu = sentiment_df[
                sentiment_df['tp_new'].fillna('').str.contains('\[').fillna(False)].copy().reset_index(drop=True)
            sentiment_df = sentiment_df[~sentiment_df['tp_new'].fillna('').str.contains('\[').fillna(False)]
            for i, row in sentiment_df_tbu.iterrows():
                print()
                print(row['ticker'])
                print(row['tickers'])
                print(row['summary'])
                print()
                tp = input(
                    f'Please insert the real target prices mannually, thanks. If no target price this is a sector report for multiple tickers'
                    ' just press Enter...')
                if len(tp) != 0:
                    print(tp)
                    sentiment_df_tbu.loc[i, 'tp_new'] = float(tp)
                else:
                    sentiment_df_tbu.loc[i, 'tp_new'] = ''
            sentiment_df = pd.concat([sentiment_df_tbu, sentiment_df], axis=0)
            sentiment_df['tp_curr'] = sentiment_df.groupby(['ticker'])['tp_new'].apply(
                lambda x: x.fillna(method='ffill'))
            # sentiment_df['tp_curr'] = sentiment_df['tp_curr'].replace('', np.nan)
            for col in ['tp_curr', 'tp_chg_pct', 'abs(tp_chg)']:
                sentiment_df[col] = sentiment_df[col].astype(float)
            sentiment_df['tp_prev'] = sentiment_df.groupby(['ticker'])['tp_curr'].apply(lambda x: x.shift())
            sentiment_df['tp_chg'] = sentiment_df.groupby(['ticker'])['tp_curr'].apply(lambda x: x - x.shift())
            sentiment_df['abs(tp_chg)'] = abs(sentiment_df['tp_chg'])
            sentiment_df['tp_chg_pct'] = sentiment_df['tp_chg'] / sentiment_df['tp_curr']

            sentiment_df['rating_curr'] = sentiment_df.groupby(['ticker'])['rating_new'].apply(
                lambda x: x.fillna(method='ffill'))
            sentiment_df['rating_prev'] = sentiment_df.groupby(['ticker'])['rating_curr'].apply(lambda x: x.shift())

            # Define report types
            er_mask = er_filter(sentiment_df['headline'], sentiment_df['summary'])
            rc_mask = sentiment_df['rating_curr'] != sentiment_df['rating_prev']
            # Backfill rating_prev with previous rating_curr
            tp_mask = sentiment_df['tp_curr'] != sentiment_df['tp_prev']
            io_mask = io_filter(sentiment_df['headline'], sentiment_df['summary'])
            ec_mask = ec_filter(sentiment_df['headline'], sentiment_df['summary'])

            sentiment_df['report_type'] = 'ad-hoc'
            sentiment_df['report_type'] = np.where(er_mask, 'Earning\'s Review', sentiment_df['report_type'])
            sentiment_df['report_type'] = np.where(ec_mask, 'Estimate Change', sentiment_df['report_type'])
            sentiment_df['report_type'] = np.where(io_mask, 'Initiation', sentiment_df['report_type'])
            sentiment_df['report_type'] = np.where(tp_mask, 'Target Price Change', sentiment_df['report_type'])
            sentiment_df['report_type'] = np.where(rc_mask, 'Rating Change', sentiment_df['report_type'])

            DL.toDB(sentiment_df, f'Cormark sentiment.csv')

        logger.info('Updating database')
        # sentiment_df = DL.loadDB('Cormark sentiment.csv', parse_dates=['publish_date_and_time'])
        sentiment_df = DL.loadDB('new_senti.csv', parse_dates=['publish_date_and_time'])
        if len(sentiment_df) == 0:
            sentiment_df = pd.DataFrame()

        if update:
            driver = UM.start_browser()
            UM.get_url(driver, 'https://cormark-research.bluematrix.com/client/library.jsp?mode=search')
            self.cormark_login(driver)

            type_lst = ['Research Report (Issuer)', 'Special Alert (Issuer)',
                        'Emerging Ideas (Issuer)', 'Morning Note (Issuer)']

            status = 'y'
            while status == 'y':
                input('Please navigate to Search page, and insert correct dates and report types:\n'
                      'Research Report (Issuer), Special Alert (Issuer), Emerging Ideas (Issuer), Morning Note (Issuer)')
                df = self.crawler(driver)
                df['headline_senti'] = None
                df['summary_senti'] = None

                # Update TPC, RC
                df[['tp_new', 'tp_prev', 'tp_chg', 'tp_chg_pct', 'abs(tp_chg)']] = None
                df[['rating_new', 'rating_prev', 'rating_curr']] = None
                df[['report_type']] = ''
                DL.toDB(df, 'new_df.csv')
                df = DL.loadDB('new_df.csv', parse_dates=['publish_date_and_time'])

                if len(sentiment_df) == 0:
                    sentiment_df = df.copy()
                else:
                    new_cols = set(df.columns) - set(sentiment_df.columns)
                    for col in new_cols:
                        sentiment_df[col] = None
                    sentiment_df = pd.concat([sentiment_df, df[sentiment_df.columns]], axis=0).drop_duplicates(
                        ['publish_date_and_time', 'uid'])

                update_sentiment_df(sentiment_df)
                status = input('Continue scraping? [y/n]')
        else:
            update_sentiment_df(sentiment_df)

if __name__ == '__main__':
    GSD = GSDatabase()
    # GSD.GS_update_sentiment(update=True)
    GSD.GS_update_sentiment(update=False)
