# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:48:20 2021

@author: jayyang
"""
import time
import sys
import random
import pandas as pd

from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By

from uti import Logger

logger = Logger()


class UrlManager:
    @classmethod
    def start_browser(cls):
        chrome_options = Options()
        chrome_options.add_argument('--incognito')
        logger.info('Starting Browser')
        browser = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
        return browser

    @classmethod
    def restart_browser(cls, browser):
        try:
            logger.info('Closing Browser')
            browser.close()
        except Exception as e:
            logger.error(e)
        browser = cls.start_browser()
        return browser

    @classmethod
    def save_html(cls, html, path):
        with open(path, 'wb') as f:
            f.write(html.encode())

    @classmethod
    def open_html(cls, path):
        with open(path, 'rb') as f:
            return f.read()

    @classmethod
    def get_url(cls, browser, url, idle=5):
        print("%s: Accessing url..." % datetime.strftime(datetime.now(), "%H:%M:%S"), end=' ')
        try:
            browser.get(url)
            logger.info(f"Success. Idle for {idle} seconds...")
            time.sleep(idle)
        except Exception as e1:
            logger.error('error on line {}: {}'.format(sys.exc_info()[-1].tb_lineno, e1))
            logger.info(f"%s: Idle for {idle} seconds..." % datetime.strftime(datetime.now(), "%H:%M:%S"))
            time.sleep(idle)
            browser = cls().restart_browser(browser)
            browser.get(url)
            time.sleep(20)

    @classmethod
    def navigate(cls, browser, **kwarg):
        # xpath - browser.find_element_by_xpath(xpath)
        # link_text - browser.find_element_by_link_text(link_text)
        # id - browser.find_element_by_id()
        # x_offset
        # y_offset webdriver.ActionChains(browser).move_by_offset(x_offset, y_offset).perform()
        # click - result.click()
        # idle - time.sleep(idle)
        mapping = {'XPATH': By.XPATH, 'CLASS': By.CLASS_NAME, 'LINK_TEXT': By.LINK_TEXT,
                   'ID': By.ID, 'TAG': By.TAG_NAME}

        if 'log' in kwarg.keys():
            # print("%s: %s" % (datetime.strftime(datetime.now(), "%H:%M:%S"), kwarg['log']), end=' ')
            logger.info('%s %s' % (datetime.strftime(datetime.now(), "%H:%M:%S"), kwarg['log']))

        if 'result' not in kwarg.keys():
            result = None
        else:
            # Try pop?
            result = kwarg['result']

        for key, value in kwarg.items():
            if key == 'result':
                continue
            elif key.upper() in mapping.keys():
                if result is None:
                    result = browser.find_element(mapping[key.upper()], value)
                else:
                    result = result.find_element(mapping[key.upper()], value)

            elif key.lower().startswith('move'):
                if result is not None:
                    webdriver.ActionChains(browser).move_to_element(result).perform()

            elif key == 'x_offset':
                webdriver.ActionChains(browser).move_by_offset(kwarg['x_offset'], kwarg['y_offset']).perform()

        if 'click' in kwarg.keys():
            if result is not None:
                try:
                    result.click()
                except Exception as e:
                    browser.execute_script("arguments[0].click()", result)
                    logger.error(e)
            else:
                webdriver.ActionChains(browser).click().perform()

        # time.sleep(random.randint(1,2))
        if 'idle' in kwarg.keys():
            time.sleep(kwarg['idle'])

        return result

    @classmethod
    def get_ranking_page(cls, browser):
        logger.info('%s Accessing 榜单...' % datetime.strftime(datetime.now(), "%H:%M:%S"))
        UrlManager.navigate(browser, xpath="//li[@class='common-dropdown']", link_text='榜单', move=True, click=True,
                            idle=1)

    @classmethod
    def get_iosranking_page(cls, browser):
        logger.info('%s Accessing 苹果榜单排名...' % datetime.strftime(datetime.now(), "%H:%M:%S"))
        UrlManager.navigate(browser, xpath="//ul[@class='menu-news-main']", link_text='苹果榜单排名', move=True, click=True,
                            idle=1)

    @classmethod
    def get_iosgameranking_page(cls, browser):
        logger.info('%s Accessing 游戏畅销榜...' % datetime.strftime(datetime.now(), "%H:%M:%S"))
        UrlManager.navigate(browser, class_='menu_level_down.menu_level_down_cate', move=True, click=True, idle=1)
        UrlManager.navigate(browser, x_offset=10, y_offset=10, link_text='游戏总榜', move=True, click=True, idle=1)
        UrlManager.navigate(browser, class_='menu_level_down', move=True, click=True, idle=1)
        UrlManager.navigate(browser, link_text='畅销', move=True, click=True, idle=1)
        # time.sleep(random.randint(15, 20))

    @classmethod
    def refresh_date(cls, date, browser):
        logger.info('%s Refreshing Dates...' % datetime.strftime(datetime.now(), "%H:%M:%S"))
        # print('Refreshing Dates...')
        y, m, d = map(lambda x: str(int(x)), date.split('-'))

        UrlManager.navigate(browser, class_='menu_level_down-toggle.date-range-picker', move=True, click=True, idle=2)
        UrlManager.navigate(browser, x_offset=10, y_offset=10, idle=1)

        table = UrlManager.navigate(browser, _id='ui-datepicker-div')

        year_toggle = UrlManager.navigate(browser, result=table, class_='ui-datepicker-year', move=True, click=True,
                                          idle=1)
        UrlManager.navigate(browser, result=year_toggle,
                            xpath="//select[@class='ui-datepicker-year']/option[@value='%s']" % y, idle=1)

        month_toggle = UrlManager.navigate(browser, result=table, class_='ui-datepicker-month', move=True, click=True,
                                           idle=1)
        UrlManager.navigate(browser, result=month_toggle,
                            xpath="//select[@class='ui-datepicker-month']/option[@value='%s']" % str(int(m) - 1),
                            click=True, idle=1)

        UrlManager.navigate(browser, class_='ui-datepicker-calendar', tag='tbody', link_text=d, click=True)
        time.sleep(random.randint(5, 10))


def remove_pop(browser):
    result = browser.find_element_by_class_name("popupCloseIcon")
    if result is not None:
        UM.navigate(browser, class_="popupCloseIcon", move=True, click=True)


def crawl(browser):
    soup = BeautifulSoup(browser.page_source, "html.parser")
    table = soup.find_all(id="results_box")[0]
    rows = table.findChildren('tr')
    df = pd.DataFrame(columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volumn'])
    unit_table = {'K': 1000, 'M': 1000000}
    for row in rows[1:-1]:
        data = []

        row_data = row.text.split('\n')[1:-1]
        date = row_data[0]
        prices = row_data[1:-1]
        volume = row_data[-1]
        data.append(date)
        for cell in prices:

            if len(cell) > 0:
                try:
                    cell = cell.replace(',', '')
                    cell = float(cell)
                    data.append(cell)
                except Exception as e:
                    logger.error(e)
        data.append(volume.split(' ')[0])
        df.loc[len(df)] = data
    df = df.set_index('Date')
    return df


def crawl_investing(browser, tickers):
    log_list = []
    completed = []

    for tick in tickers:
        if tick in completed:
            continue
        logger.info(tick)
        try:
            UM.navigate(browser, class_="searchText", click=True)
            browser.find_element_by_class_name("searchText").send_keys(tick)
            time.sleep(random.randint(1, 3))
            UM.navigate(browser, class_="row", move=True, click=True)
            time.sleep(random.randint(1, 3))
        except:
            pass
        try:
            remove_pop(browser)
        except:
            pass

        logger.info('%s Clicking Historical Data %s' % (datetime.strftime(datetime.now(), "%H:%M:%S"), tick))
        UM.navigate(browser, link_text="Historical Data", move=True, click=True, idle=2)
        try:
            df = crawl(browser)
            # df.to_csv(os.Path.join(Path, f"{tick}.csv"))
            logger.info("Success!")
            completed.append(tick)
        except:
            logger.error(f"{tick} not crawled.")
            log_list.append(tick)

    return log_list


if __name__ == '__main__':
    UM = UrlManager()
    browser = UM.start_browser()
