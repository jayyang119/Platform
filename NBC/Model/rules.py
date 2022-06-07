import numpy as np
import pandas as pd
from Broker import get_pnl
from uti import DataLoader
from Backtest.visualize import plot_matrix
from Model import DataCleaner
from Backtest.settings import get_expectancy

DL = DataLoader()
DC = DataCleaner()


def benchmark_rule(df_):

    df = df_[['headline_senti', 'summary_senti']].copy()
    # if 'side' not in df.columns:
    df['side'] = 'neutral'

    long_mask = ((df['headline_senti'] == 'positive') & (df['summary_senti'] == 'positive'))
    short_mask = (df['headline_senti'] == 'negative') | (df['summary_senti'] == 'negative')

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def benchmark_rule2(df_):

    #
    # for col in ['tp_curr', 'tp_prev']:
    #     if col not in df_.columns:
    #         df_[col] = ''
    df = df_[['tp_curr', 'tp_prev', 'side']].copy()
    long_mask = (df['tp_prev'] < df['tp_curr'])
    short_mask = (df['tp_prev'] > df['tp_curr'])

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def benchmark_rule3(df_):
    rating_num = {'Outperform': 1, 'Sector Performance': 0, 'Underperform': -1}

    df = df_[['rating_prev', 'rating_curr', 'side']].copy()
    df['rating_prev_num'] = df_['rating_prev'].map(rating_num)
    df['rating_curr_num'] = df_['rating_curr'].map(rating_num)
    # df['rating_chg_num'] = df['rating_curr_num'] - df['rating_prev_num']

    long_mask = (df['rating_curr_num'] - df['rating_prev_num'] > 0)
    short_mask = (df['rating_curr_num'] - df['rating_prev_num'] < 0)

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']

# def benchmark_rule4(df_):
#     df = df_[['tickers', 'side']].copy()
#     multi_ticker_mask = (df_['tickers'].fillna('').str.contains(','))
#     df['side'] = np.where(multi_ticker_mask, 'neutral', df['side'])
#
#     return df['side']
