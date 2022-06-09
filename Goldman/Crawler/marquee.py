import re
import os
import glob
import time
import numpy as np
import pandas as pd
from datetime import datetime
from random import randint
from bs4 import BeautifulSoup

from uti import Logger, UrlManager, DataLoader, By
from Database.settings import REPORT_TYPE_DICT

DL = DataLoader()
UM = UrlManager()
logger = Logger()
DATABASE_PATH = DL.database_path


def gs_get_page_time(text):
    """
        This function cleans the time format of Goldman report publication.
    """
    text, zone = text[:-2], text[-2:]
    mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
               'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    d, m, y, H, M = re.split('[ :]', text)
    m = mapping[m]
    d, m, y, H, M = map(int, [d, m, y, H, M])
    if zone.lower() == 'pm' and H != 12:
        H += 12
    elif zone.lower() == 'am' and H == 12:
        H -= 12

    return datetime(y, m, d, H, M)


def gs_get_page_data(row, output_type='list'):
    """
        This function cleans information from the html table of each page of Equity Research reports.
    """
    writeline = {}
    # Get dates
    _time = gs_get_page_time(row.find_all('div', class_="SearchResults__dateInTitleCol")[0].text)
    _time = datetime.strftime(_time, '%Y/%m/%d %H:%M:%S')
    _date = _time.split(' ')[0]

    _metaText = row.find_all('div', class_="SearchResults__metaText")[0]
    _source = _metaText.text.split(" | ")[0]
    _asset_class = _metaText.text.split(" | ")[1].split(' -  ')[0]

    # Get headline, source, asset_class, analysts, summary
    _headline = row.find_all('a', class_="SearchResults__headline")[0]
    try:
        if ':' not in _headline.text and '(' not in _headline.text:
            _ticker = ''
            _asset_class = 'Macro'
        elif '(' in _headline.text and ':' in _headline.text and _headline.text.find('(') < _headline.text.find(':'):
            try:
                _ticker = re.findall(r'\((.*?)\)[ /:]', _headline.text)[0]
                # _ticker = re.replace('_', _ticker)
            except Exception as e:
                logger.error(f'Please check {_headline}', e)
                _ticker = ''
                _asset_class = 'Macro'
        elif ':' in _headline.text:
            _ticker = _headline.text.split(':')[0]
            _asset_class = 'Macro'
        else:
            _ticker = ''
        _ticker = _ticker.replace(', ', '/')
        _ticker = re.split('[ |/]', _ticker)[0]

        if len(_metaText.text.split(" | ")[1].split(' -  ')) != 2:
            _analysts = ' '
        else:
            _analysts = _metaText.text.split(" | ")[1].split(' -  ')[1]
        _analysts = _analysts.replace(', CFA', '').replace(', Ph.D.', '')
        _analysts = _analysts.replace(',', '|')

        try:
            _summary = row.find_all('p', class_="SearchResults__colExtract")[0].text
            _summary = _summary.replace(',', '')
        except Exception as e:
            logger.error(e)
            logger.info(_headline.text)
            _summary = ''

        writeline = {'Date': _date, 'Time': _time, 'Ticker': _ticker.replace(',', ' '), 'Source': _source.replace(',', ' '), 'Asset Class': _asset_class.replace(',', ' '),
                     'Headline': _headline.text.replace(',', ' '), 'Summary': _summary.replace(',', ' '), 'Analysts': _analysts}

        if output_type != 'list':
            return writeline
        return list(writeline.values())

    except Exception as e:
        logger.error(e)
        logger.info(_headline.text)


def gs_flip_page(browser):
    """
        This function navigates the browser to flip to the next page of Equity Research reports.
    """
    try:
        # UM.navigate(browser, XPATH=f"//ul/li/a[contains(text(), {str(page)})]", click=True, idle=randint(3, 5))
        UM.navigate(browser, XPATH="//ul/li/a[@class='SearchResults__paginationNext']", click=True, idle=randint(3, 5))
    except Exception as e:
        logger.error(e)


def gs_get_page_table_and_time(browser, report_type_path):
    """
        This function retrieves the latest report published on the current Goldman Equity Research platform,
        and compare with the latest report existing in the database.
    """
    try:
        table = browser.find_element(By.XPATH, '//table')
    except Exception as e:
        logger.error(e)
        time.sleep(randint(3, 5))
        table = browser.find_element(By.XPATH, '//table')
    table = BeautifulSoup(table.get_attribute('innerHTML'), 'html.parser')
    latest_report_time = gs_get_page_time(table.find_all('div', class_="SearchResults__dateInTitleCol")[0].text)
    latest_report_time_str = datetime.strftime(latest_report_time, "%Y%m%d%H%M")
    logger.info(latest_report_time)
    logger.info(latest_report_time_str)

    if not DL.checkDB(f'{DATABASE_PATH}/{report_type_path}/{latest_report_time_str[:8]}'):
        try:
            DL.create_folder(f'{DATABASE_PATH}/{report_type_path}/{latest_report_time_str[:8]}')
            DL.create_folder(
                f'{DATABASE_PATH}/{report_type_path}/{latest_report_time_str[:8]}/{latest_report_time_str}.json')
        except Exception as e:
            logger.error(e)
    UM.save_html(table,
                 f'{DATABASE_PATH}/{report_type_path}/{latest_report_time_str[:8]}/{latest_report_time_str}.json')
    logger.info('Reports up to ' + datetime.strftime(latest_report_time, '%Y-%m-%d %H:%M:%S') + ' saved.')

    return table, latest_report_time, latest_report_time_str


def gs_save_page(browser, crawl_older_dates=False, report_type_path='Goldman Reports'):
    """
        This function saves the page html table to local database, stored in
        Database/Goldman/Goldman Reports by date.
    """
    if not crawl_older_dates:
        latest_record = max([os.path.basename(x) for x in glob.glob(f'{DATABASE_PATH}/{report_type_path}/*')]).rstrip(
            '.json')
        latest_record = max([os.path.basename(x) for x in
                             glob.glob(f'{DATABASE_PATH}/{report_type_path}/{latest_record}/*.json')]).rstrip('.json')

    table, latest_report_time, latest_report_time_str = gs_get_page_table_and_time(browser, report_type_path)

    new_df = np.array([])
    while True:
        try:
            if crawl_older_dates:
                if latest_report_time <= datetime(2018, 8, 1):
                    return
                latest_record = min(
                    os.listdir(f'{DATABASE_PATH}/{report_type_path}/{latest_report_time_str[:8]}')).rstrip('.json')
                if latest_report_time_str <= latest_record:
                    UM.save_html(table,
                                 f'{DATABASE_PATH}/{report_type_path}/{latest_report_time_str[:8]}/{latest_report_time_str}.json')
                    logger.info(
                        'Reports up to ' + datetime.strftime(latest_report_time, '%Y-%m-%d %H:%M:%S') + ' saved.')

                # Flip page
                gs_flip_page(browser)
                table, latest_report_time, latest_report_time_str = gs_get_page_table_and_time(browser, report_type_path)
            else:
                if latest_report_time_str > latest_record:

                    UM.save_html(table, f'{DATABASE_PATH}/{report_type_path}/{latest_report_time_str[:8]}/{latest_report_time_str}.json')
                    logger.info(
                        'Reports up to ' + datetime.strftime(latest_report_time, '%Y-%m-%d %H:%M:%S') + ' saved.')

                    soup = BeautifulSoup(str(table), 'html.parser')
                    links = browser.find_elements(By.CLASS_NAME, 'SearchResults__headline')
                    trs = soup.find_all('tr')[1:]

                    for i in range(len(trs)):
                        row = trs[i]
                        link = links[i]

                        newline = gs_get_page_data(row, 'list')
                        logger.info(newline[2])

                        latest_report_time_str = datetime.strftime(datetime.strptime(newline[1],  '%Y/%m/%d %H:%M:%S'), "%Y%m%d%H%M")
                        if latest_report_time_str <= latest_record:
                            break

                        # Click link, switch to the new window.
                        UM.navigate(browser, result=link, click=True, idle=randint(3, 5))
                        browser.switch_to.window(browser.window_handles[1])

                        # Grab and determine Report Type based on hierarchy
                        tag_element = browser.find_element(By.XPATH, f"//ul[@class='tag-list']")
                        tag_soup = BeautifulSoup(tag_element.get_attribute('innerHTML'), 'html.parser')

                        titles_html = tag_soup.find_all(title=True)
                        tags_str = [x['title'] for x in titles_html]

                        report_type = 'ad-hoc'
                        for tag, rt in REPORT_TYPE_DICT.items():
                            if tag in tags_str:
                                report_type = rt
                                break
                        logger.info(report_type)
                        newline.append(report_type)
                        new_df = np.append(new_df, newline)

                        browser.close()
                        browser.switch_to.window(browser.window_handles[0])
                        time.sleep(0.000001)

                    # Flip page
                    gs_flip_page(browser)
                    table, latest_report_time, latest_report_time_str = gs_get_page_table_and_time(browser, report_type_path)
                    logger.info('Latest_report_time ' + latest_report_time_str)
                    logger.info('Latest_record ' + latest_record)
                else:
                    break

        except Exception as e:
            logger.info(latest_report_time)
            logger.info(latest_report_time_str)
            logger.error(e)
            time.sleep(randint(1, 3))


    # Update database
    new_df = new_df.reshape(-1, 9)
    new_df = pd.DataFrame(new_df,
                          columns=['Date', 'Time', 'Ticker', 'Source', 'Asset Class', 'Headline',
                                   'Summary', 'Analysts', 'Report Type'])
    new_df['Time'] = new_df['Time'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    logger.info(new_df)


    GS_raw = DL.loadDB('GS_raw.csv', parse_dates=['Date', 'Time'])
    GS_raw = pd.concat([new_df, GS_raw], axis=0)
    GS_raw.drop_duplicates(inplace=True)
    GS_raw = GS_raw.sort_values(['Time'], ascending=False)
    DL.toDB(GS_raw, 'GS_raw.csv')
