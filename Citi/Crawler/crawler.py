import time
from datetime import datetime
from random import randint

from Crawler.outlook import outlook_initialize, outlook_get_marquee_link
from Crawler.marquee import gs_save_page
from Database.settings import REPORT_TYPE_MAPPING
from uti import UrlManager, Logger, By

UM = UrlManager()
logger = Logger()


def GS_repeat_get_url(browser, messages=[]):
    logger.info('Accessing url...')
    urls = outlook_get_marquee_link(messages)
    assert len(urls) > 0, 'URL is empty, please check the Email messages.'

    while True:
        url = urls[randint(0, len(urls)-1)]
        UM.get_url(browser, url)
        time.sleep(randint(1, 3))
        try:
            # Step 1: Get to Search Page
            userinput = input('Please check the website. Continue? (y/n)')
            while userinput.lower() not in ['y', 'n']:
                userinput = input('Input not recognized. Continue? (y/n)')
            if userinput.lower() == 'y':
                break

            # UM.navigate(browser, XPATH="//ul[@class='aurora-shell-header-menu-bar-items']/li[2]", move=True, idle=2, log='Accessing search page')
            # UM.navigate(browser, XPATH="//li/a[@class='current'][@title='Equity Research]", move=True, click=True)
            #             idle=randint(3, 5),
            #             log='Clicking Equity Research')
        except Exception as e:
            logger.error(e)

def GS_crawler(crawl_older_dates=False):
    if crawl_older_dates:
        logger.info('Job: GS_crawler | Mode: crawl older dates')
    else:
        logger.info('Job: GS_crawler | Mode: daily update')

    # Step 1: Get valid url
    GS_messages = outlook_initialize()
    browser = UM.start_browser()
    GS_repeat_get_url(browser, GS_messages)

    userinput = 'y'
    while userinput.lower() == 'y':
        try:
            userinput = input(
                'Please update the website. \n1. Limit To: Exclude Models\n2. Sub-sources/Assets: Equity Research). \n3. Click Date toggle. \nContinue? (y/n)')
            while userinput.lower() not in ['y', 'n']:
                userinput = input('Input not recognized. Continue? (y/n)')
            if userinput.lower() == 'n':
                return

            # UM.navigate(browser, x_offset=10, y_offset=100, click=True, idle=3)

            # Step 2: View More
            # UM.navigate(browser, XPATH="//a[contains(text(), 'View More')]", move=True, click=True, idle=randint(3, 5), log='Clicking view more')

            # Step 3: Sort the reports starting with the newest, and save html page by page
            # sortcontainer = browser.find_element(By.XPATH, "//div[@class='SearchResults__sortContainer']")
            # sortcontainer.click()

            # UM.navigate(browser, XPATH="//option[contains(text(), 'Date')]", click=True, idle=2, log='Sorting the reports by date')
            # sortcontainer.click()

            if crawl_older_dates:
                userinput = input('Please update the dates in GS platform. Continue? (y/n)')
                while userinput.lower() not in ['y', 'n']:
                    userinput = input('Input not recognized. Continue? (y/n)')
                if userinput.lower() == 'n':
                    return

            # Step 4: save pages.
            gs_save_page(browser, crawl_older_dates=crawl_older_dates)
            print("%s: %s" % (datetime.strftime(datetime.now(), "%H:%M:%S"), 'All reports downloaded.'))


        except Exception as e:
            logger.error(e)

        userinput = input('Re-crawl the website? [y/n]')

def GS_crawler_report_type(crawl_older_dates=True):
    # Step 1: Get valid url
    GS_messages = outlook_initialize()
    browser = UM.start_browser()
    GS_repeat_get_url(browser, GS_messages)

    if crawl_older_dates:
        logger.info('Job: GS_crawler_report_type | Mode: crawl older dates')
    else:
        logger.info('Job: GS_crawler_report_type | Mode: daily update')

    # while True:
    try:
        UM.navigate(browser, x_offset=10, y_offset=100, click=True, idle=3)

        # Step 2: View More
        UM.navigate(browser, XPATH="//a[contains(text(), 'View More')]", move=True, click=True, idle=randint(3, 5), log='Clicking view more')

        # Step 3: Sort the reports starting with the newest, and save html page by page
        sortcontainer = browser.find_element(By.XPATH, "//div[@class='SearchResults__sortContainer']")
        sortcontainer.click()
        UM.navigate(browser, XPATH="//option[contains(text(), 'Date')]", click=True, idle=2, log='Sorting the reports by date')
        sortcontainer.click()

        print(REPORT_TYPE_MAPPING)
        print()
        print('0: Earning\'s Review (Subjects: Earnings) \n \
        1: Rating Change (Actions: Rating Change/Rating Downgrade/Rating Upgrade) \n \
        2: Target Price Change (Actions: 2.1 Price Target Increase/2.2 Decrease) \n \
        3: Estimate Change (Actions: EPS Estimate Change) \n \
        4: Earning\'s Preview (Subjects: Earnings Preview) \n')

        userinput = input('Please input report type:')
        while userinput not in REPORT_TYPE_MAPPING.keys():
            userinput = input('Please insert again, valid inputs: 0, 1, 2, 3, 4')
        report_type = REPORT_TYPE_MAPPING[userinput]
        logger.info(f'Job: GS_crawler_report_type | Mode: daily update | Report type: {report_type}')
        userinput = input('Please update the report type tags in GS platform. Continue? (y/n)')

        if crawl_older_dates:
            userinput = input('Please update the dates in GS platform. Continue? (y/n)')
            while userinput.lower() not in ['y', 'n']:
                userinput = input('Input not recognized. Continue? (y/n)')
            if userinput.lower() == 'n':
                return

        # Step 4: save pages.
        gs_save_page(browser, crawl_older_dates=crawl_older_dates, report_type_path=f'Report Types/{report_type}')
        print("%s: %s" % (datetime.strftime(datetime.now(), "%H:%M:%S"), 'All reports downloaded.'))

        # Step 5: update database with new data, and sentiment.
        # GS_update_database()

        userinput = input(f'{report_type} completed. Continue?')
        if userinput.lower() not in ['y', 'n']:
            userinput = input('Input not recognized. Continue? (y/n)')
        if userinput.lower() == 'n':
            return

    except Exception as e:
        logger.error(e)

