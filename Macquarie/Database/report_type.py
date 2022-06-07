import os
import pandas as pd
import sys
dir_name = os.path.dirname(__file__)
print(dir_name)
if os.path.abspath(f'{dir_name}/..') not in sys.path:
    sys.path.append(os.path.abspath(f'{dir_name}/..'))
    print(dir_name)

from database import GSDatabase
from uti import timeit, DataLoader, Logger
from datetime import datetime
from settings import REPORT_TYPE_MAPPING
from Crawler import gs_get_page_data, gs_get_report_type

from Path import DATABASE_PATH

GSD = GSDatabase()
DL = DataLoader()
logger = Logger()

# REPORT_TYPE_MAPPING = {'0': 'Earning\'s Review', '1': 'Rating Change',
#                        '2': 'Target Price Change', '2.1': 'Target Price Increase', '2.2': 'Target Price Decrease',
#                        '3': 'Estimate Change',
#                        '4': 'Earning\'s Preview',  # mutually exclusive with Earning's Review
#                        }

def report_type_df():
    for report_type in REPORT_TYPE_MAPPING.values():
        if report_type in os.listdir(f'{DATABASE_PATH}/Report Types'):
            logger.info(report_type)
            # report_type = 'Earning\'s Review'

            GSD.GS_rewrite_database(os.path.join(DATABASE_PATH, f'Report Types/{report_type}'))
            df = pd.read_csv(os.path.join(DATABASE_PATH, f'Report Types/{report_type}/output.csv'))
            df['Report Type'] = report_type
            DL.toDB(df, f'{DATABASE_PATH}/Report Types/{report_type}/{report_type}.csv')



def compile_report_type_df():
    # Hierarchy: Earning's Review, Rating Change, Price Target increase/decrease >= 10%, EPS estimate change, ad-hoc
    # Initiation
    # total_df = pd.DataFrame()
    # dates_range = pd.date_range(datetime(2018, 8, 1), datetime(2022, 2, 5))
    # for report_type in REPORT_TYPE_MAPPING.values():
    #     if DL.checkDB(os.Path.join(DATABASE_PATH, f'Report Types/{report_type}.csv')):
    #         df = DL.loadDB(f'Report Types/{report_type}.csv', parse_dates=['Date', 'Time'])
    #         df = df[df['Date'].isin(dates_range)]
    #
    #         if len(total_df) == 0:
    #             total_df = df.copy()
    #         else:
    #             total_df = pd.merge(total_df, df, on='Headline', how='outer', )
    sentiment_df = DL.loadDB('GS_sentiment.csv', parse_dates=['Date', 'Time'])
    dates_range = pd.date_range(datetime(2018, 8, 1), datetime(2022, 2, 5))
    sentiment_df = sentiment_df[sentiment_df['Date'].isin(dates_range)]
    sentiment_df['Report Type'] = 'ad-hoc'
    for report_type in ['Earning\'s Review', 'Rating Change', 'Target Price Increase', 'Target Price Decrease',
                        'Estimate Change', 'Initiation'][::-1]:
        logger.info(report_type)
        if DL.checkDB(os.path.join(DATABASE_PATH, f'Report Types/{report_type}/{report_type}.csv')):
            df = DL.loadDB(f'Report Types/{report_type}/{report_type}.csv', parse_dates=['Date', 'Time'])
            df = df[df['Date'].isin(dates_range)]

            criteria_time = sentiment_df['Time'].isin(df['Time'])
            criteria_headline = sentiment_df['Headline'].isin(df['Headline'])
            indices = sentiment_df[criteria_time&criteria_headline].index
            sentiment_df.loc[indices, 'Report Type'] = report_type

    DL.toDB(sentiment_df, 'test_sentiment.csv')





if __name__ == '__main__':
    compile_report_type_df()
    pass

