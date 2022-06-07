from uti import timeit, DataLoader, Logger, UrlManager, By
from Path import ONEDRIVE_PATH
from collections import defaultdict
from Database.report_type import rc_filter, er_filter, io_filter, ec_filter

import os
import time
import pandas as pd
import numpy as np
import torch
import sys

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

    def nbc_login(self, driver, need_un=True):
        username = 'donnert@a-s-capital.com'
        password = 'Pass4321?'
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

    def crawler(self, driver=None):
        driver = UM.start_browser()
        UM.get_url(driver, 'https://nbf-library.bluematrix.com/client/library.jsp')
        self.nbc_login(driver)

        type_lst = ['Research Flash']
        result = defaultdict(list)
        flag = False

        # Need to get page length:
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
                result['broker_label'].append(tds[3].text.strip('\n').strip())
                result['publish_date_and_time'].append(tds[4].text.strip('\n').strip())
            #     pdb.set_trace()

            next_button = driver.find_elements(By.XPATH, '//li[@class="paginate_button page-item next"]')
            if len(next_button) > 0:
                UM.navigate(driver, xpath='//li[@class="paginate_button page-item next"]', move=True, click=True)

        df = pd.DataFrame(result)
        for i in range(len(df)):
            driver.get(df.at[i, 'url'].replace('pdf', 'html'))
            time.sleep(0.1)

            trs = driver.find_elements(By.XPATH, '//td/table/tbody')[1]
            soup = BeautifulSoup(trs.get_attribute('innerHTML'), 'html.parser')

            rating = soup.find_all('tbody')[0].find_all('td')[2].text.replace(u'\xa0', u' ').strip('\n ')
            rating = rating.split(': ')[1]
            df.at[i, 'rating_curr'] = rating

            tp = soup.find_all('tbody')[0].find_all('td')[3].text.replace(u'\xa0', u' ').strip('\n ')
            if ';' in tp:
                tp = tp.split(';')[0]
            try:
                tp = tp.split('$')[1]
            except:
                print(tp)
                tp = ''
            df.at[i, 'tp_curr'] = tp

            summary = ' '.join([x.text for x in soup.find_all('p')])
            df.at[i, 'summary'] = summary

            uid = df.at[i, 'url'].split('encrypt=')[1].split('&')[0]
            df.at[i, 'uid'] = uid

            time.sleep(0.002)
        df['report_type'] = ''

        return df

    @timeit
    def GS_update_sentiment(self, update=True):
        logger.info('Updating database')
        sentiment_df = DL.loadDB('NBC sentiment.csv', parse_dates=['publish_date_and_time'])

        if update:
            df = self.crawler()
            df['headline_senti'] = None
            df['summary_senti'] = None

            # Update TPC, RC
            df[['tp_prev', 'tp_chg', 'tp_chg_pct', 'abs(tp_chg)']] = None
            df[['rating_prev']] = None
            df[['report_type']] = ''

            if len(sentiment_df) == 0:
                new_df = df.copy()
            else:
                sentiment_df = pd.concat([sentiment_df, df[sentiment_df.columns]], axis=0).drop_duplicates(
                    ['publish_date_and_time', 'uid'])
                new_df = sentiment_df[sentiment_df['headline_senti'].isna()].copy().reset_index(drop=True)
                sentiment_df = sentiment_df[~sentiment_df['headline_senti'].isna()].reset_index(drop=True)

            if len(new_df) > 0:
                new_df['headline_senti'], _ = update_sentiment(new_df['headline'])
                new_df['summary_senti'], _ = update_sentiment(new_df['summary'])
            DL.toDB(new_df, 'new_df.csv')

        new_df = DL.loadDB('new_df.csv', parse_dates=['publish_date_and_time'])
        logger.info(new_df)

        sentiment_df = pd.concat([new_df[sentiment_df.columns], sentiment_df], axis=0).sort_values(
            'publish_date_and_time', ascending=True). \
            drop_duplicates(['publish_date_and_time', 'uid'], keep='first').reset_index(drop=True)
        sentiment_df['rating_prev'] = sentiment_df.groupby(['ticker'])['rating_curr'].apply(lambda x: x.shift())

        sentiment_df = sentiment_df.sort_values(['publish_date_and_time'], ascending=True).reset_index(drop=True)
        sentiment_df['tp_curr'] = sentiment_df.groupby(['ticker'])['tp_curr'].apply(lambda x: x.fillna(method='ffill'))

        for col in ['tp_curr', 'tp_chg_pct', 'abs(tp_chg)']:
            sentiment_df[col] = sentiment_df[col].astype(float)
        sentiment_df['tp_prev'] = sentiment_df.groupby(['ticker'])['tp_curr'].apply(lambda x: x.shift())
        sentiment_df['tp_chg'] = sentiment_df.groupby(['ticker'])['tp_curr'].apply(lambda x: x - x.shift())
        sentiment_df['abs(tp_chg)'] = abs(sentiment_df['tp_chg'])
        sentiment_df['tp_chg_pct'] = sentiment_df['tp_chg'] / sentiment_df['tp_curr']

        sentiment_df['rating_curr'] = sentiment_df.groupby(['ticker'])['rating_curr'].apply(
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
        DL.toDB(sentiment_df, f'NBC sentiment.csv')
        return sentiment_df

if __name__ == '__main__':
    GSD = GSDatabase()
    GSD.GS_update_sentiment(update=True)


