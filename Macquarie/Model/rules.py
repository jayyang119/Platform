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
    df = df_[['Headline sentiment', 'Summary sentiment', 'side']].copy()

    long_mask = ((df['Headline sentiment'] == 'positive') & (df['Summary sentiment'] == 'positive'))
    short_mask = (df['Headline sentiment'] == 'negative') | (df['Summary sentiment'] == 'negative')

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def benchmark_rule2(df_):
    if 'side' not in df_.columns:
        df_['side'] = 'neutral'

    for col in ['TPS', 'TPS_prev', 'RC_upgrade', 'RC_downgrade']:
        if col not in df_.columns:
            df_[col] = ''

    df = df_[['TPS', 'TPS_prev', 'RC_upgrade', 'RC_downgrade', 'side']].copy()

    long_mask = (df['TPS_prev'] < df['TPS'])
    short_mask = (df['TPS_prev'] > df['TPS'])

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def benchmark_rule3(df_):
    if 'side' not in df_.columns:
        df_['side'] = 'neutral'
    for col in ['RC_upgrade', 'RC_downgrade']:
        if col not in df_.columns:
            df_[col] = ''
    df = df_[['RC_upgrade', 'RC_downgrade', 'side']].copy()

    long_mask = ((df['RC_upgrade'] == 'Y') & (df['RC_downgrade'] != 'Y'))
    short_mask = ((df['RC_upgrade'] != 'Y') & (df['RC_downgrade'] == 'Y'))

    df['side'] = np.where(long_mask, 'long', df['side'])
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']