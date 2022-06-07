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
    if 'side' not in df_.columns:
        df_['side'] = 'neutral'
    df = df_[['headline_senti', 'summary_senti', 'side']].copy()

    long_mask = ((df['headline_senti'] == 'positive') & (df['summary_senti'] == 'positive'))
    short_mask = (df['headline_senti'] == 'negative') | (df['summary_senti'] == 'negative')

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def benchmark_rule2(df_):
    if 'side' not in df_.columns:
        df_['side'] = 'neutral'

    for col in ['tp_curr', 'tp_prev', 'RC_upgrade', 'RC_downgrade']:
        if col not in df_.columns:
            df_[col] = ''

    df = df_[['tp_curr', 'tp_prev', 'RC_upgrade', 'RC_downgrade', 'side']].copy()

    long_mask = (df['tp_prev'] < df['tp_curr'])
    short_mask = (df['tp_prev'] > df['tp_curr'])

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def benchmark_rule3(df_):
    # if 'side' not in df_.columns:
    #     df_['side'] = 'neutral'
    # for col in ['RC_upgrade', 'RC_downgrade']:
    #     if col not in df_.columns:
    #         df_[col] = ''
    # df = df_[['RC_upgrade', 'RC_downgrade', 'side']].copy()
    #
    # long_mask = ((df['RC_upgrade'] is True) & (not df['RC_downgrade'] is True))
    # short_mask = ((not df['RC_upgrade'] is True) & (df['RC_downgrade'] is True))
    #
    # df['side'] = np.where(long_mask, 'long', df['side'])
    # df['side'] = np.where(short_mask, 'short', df['side'])
    #
    # return df['side']
    rating_num = {'buy': 1, 'neutral': 0, 'sell': -1}

    df = df_[['rating_prev', 'rating_curr', 'side']].copy()
    df['rating_prev_num'] = df_['rating_prev'].map(rating_num)
    df['rating_curr_num'] = df_['rating_curr'].map(rating_num)
    # df['rating_chg_num'] = df['rating_curr_num'] - df['rating_prev_num']

    long_mask = (df['rating_curr_num'] - df['rating_prev_num'] > 0)
    short_mask = (df['rating_curr_num'] - df['rating_prev_num'] < 0)

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def benchmark_rule4(df_):
    df = df_[['tickers', 'side']].copy()
    multi_ticker_mask = (df_['tickers'].fillna('').str.contains(','))
    df['side'] = np.where(multi_ticker_mask, 'neutral', df['side'])

    return df['side']
