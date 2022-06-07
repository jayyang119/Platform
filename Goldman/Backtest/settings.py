# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 09:42:29 2021

@author: jayyang
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from collections import defaultdict


def preprocessing(df):
    df = df.copy(deep=True)
    df['Side'] = df['Side'].apply(lambda x: x.lower())
    if 'Side2' not in df.columns:
        df['Side2'] = df.loc[:]['Side'].apply(lambda x: 1 if x == 'long' else -1)
    return df


def sum_R(df, column='d0_r'):
    # Calculate the Sum of R, given a dataset of trades.
    return df[column].sum()


def hit_ratio(df, column='d0_r'):
    # Calculate the Hit ratio, given a dataset of trades.
    trades_win = df[df[column] > 0]
    if len(df) == 0:
        return 0
    hit_r = len(trades_win) / len(df)
    return hit_r


def benchmark_side(df, column='d0_r'):
    # Given a dataset of trades, split into long and short trades, and calculate hit ratios for both sides.
    long_trades_df = df[df['side'].isin(['long', 'positive'])]
    short_trades_df = df[df['side'].isin(['short', 'negative'])]

    long_hit_ratio = hit_ratio(long_trades_df, column)
    print("Long trades hit ratio %.4f" % long_hit_ratio)

    short_hit_ratio = hit_ratio(short_trades_df, column)
    print("Short trades hit ratio %.4f" % short_hit_ratio)

    return long_hit_ratio, short_hit_ratio

def get_expectancy(df, column, inputs=[], group_by=[]):
    # Given a dataset of trades, groupby based a given set of criteria, and output the expectancy of each group.
    df = df[~df['side'].isin(['neutral'])].copy()
    if len(inputs) > 0 and len(group_by) > 0:
        expectancy = df[inputs].groupby(group_by)[column].agg({
            ('Hit ratio', lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0),
            ('Loss ratio', lambda x: (x < 0).sum() / len(x) if len(x) > 0 else 0),
            ('Avg win size', lambda x: x[x > 0].mean()),
            ('Avg losing size', lambda x: x[x < 0].mean()),
            ('Count', 'count')}).fillna(0)
    else:
        expectancy = df[column].agg({
            'Hit ratio': lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0,
            'Loss ratio': lambda x: (x < 0).sum() / len(x) if len(x) > 0 else 0,
            'Avg win size': lambda x: x[x > 0].mean(),
            'Avg losing size': lambda x: x[x < 0].mean(),
            'No. of trades': lambda x: x.sum()})
    try:
        expectancy['Expectancy'] = expectancy['Hit ratio'] * expectancy['Avg win size'] + \
                                   expectancy['Loss ratio'] * expectancy['Avg losing size']
    except:
        print(expectancy)
    return expectancy

def show_values_on_bars(axs, precision=0):
    # Add the value annotation on top of the bar plot.
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = f'%.{precision}f' % p.get_height()
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def plot_EDA(data, labels, title='Trades by sides', precision=0):
    # Plot the result of Exploratory Data Analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.35
    try:
        x = np.arange(len(data))
        for i in x:
            ax.bar([i], data[i], width, label=labels[i])

        show_values_on_bars(ax, precision)
        ax.set_title(title)
        ax.set_xticks(x)

        ax.set_xticklabels(labels)
    except:
        ax.bar([0], data, width, label=labels)

        show_values_on_bars(ax, precision)
        ax.set_title(title)
    ax.legend(loc='center')
    plt.ylabel('Count')
    return fig


def plot_EDA_country(df, hue=None, x='exch_location', y='No. of trades', title='', precision=0, rotate=False):
    # Plot the result of Exploratory Data Analysis, with sub-classificaiton.
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.title(title)
    if hue is None:
        sns.barplot(x=x, y=y, data=df)
    else:
        sns.barplot(x=x, y=y, hue=hue, data=df)

    if rotate:
        plt.xticks(rotation=45)
    show_values_on_bars(ax, precision=precision)
    return fig


def get_price(price_df, parse_dates, name='Close'):
    # Return a price for easier mapping, default as Close Price
    intraday_prices = defaultdict(dict)
    for ticker, row in price_df.iterrows():
        entry_date = row[parse_dates[0]]

        entry_date_prices = defaultdict(list)
        for i in parse_dates:
            date = row[i]
            date_col = '/'.join(i.split('/')[:-1] + [name])
            close = row[date_col]
            entry_date_prices['price'].append(close)
            entry_date_prices['dates'].append(date)
        intraday_prices[ticker][entry_date] = entry_date_prices
    return intraday_prices


def get_exit_data(df, CLOSE_DATA, OPEN_DATA, dates_mapping_index, holding_period=None, stop_intraday=True):
    # Stop loss algorithm
    # Compute the Exit Price and Exit Date of holding period ranging from 1 day to maximum holding_period, 
    # based on the a series of trading days with Close price, Open price and STOP LOSS PRICE
    # Output 1: exit_database, which contains a 2-d array specifying the Exit Prices and Exit Dates for each stock during the tradespan.
    # Output 2: stopped_record, whether the trade is hold to maturity or stopped intraday.
    stop_loss_data = df.loc[:]['STOP LOSS PRICE'].values
    side_data = df.loc[:]['side2'].values

    if holding_period is not None:
        holding_periods = [holding_period for i in range(len(stop_loss_data))]
        close_data = CLOSE_DATA[:, 1]
        open_data = OPEN_DATA[:, 1]
        date_data = CLOSE_DATA[:, 0]

    else:
        holding_periods = [dates_mapping_index[_ticker] for _ticker in df.index]
        close_data = [x[1] for x in CLOSE_DATA]
        open_data = [x[1] for x in OPEN_DATA]
        date_data = [x[0] for x in CLOSE_DATA]

    exit_database = []
    stopped_record = []

    for i in range(len(stop_loss_data)):
        stopped = 0
        date_i = []
        exit_i = []

        for j in range(holding_periods[i]):
            if stopped:
                exit_i.append(exit_i[-1])
                date_i.append(date_i[-1])
            else:
                # Long position
                if side_data[i] == 1:
                    # Long position - stopped at Open price
                    if open_data[i][j] < stop_loss_data[i]:
                        exit_i.append(open_data[i][j])
                        date_i.append(date_data[i][j])
                        stopped = 1
                    else:
                        # Long position - stopped at STOP LOSS PRICE during the trading day, Realized R = -1
                        if stop_loss_data[i] > close_data[i][j]:
                            if stop_intraday is True:
                                exit_i.append(stop_loss_data[i])  # Allowed to stop at open price
                                stopped = 1
                            else:
                                exit_i.append(close_data[i][j])  # Not allowed to stop at open price
                            date_i.append(date_data[i][j])

                        # Long position - close position at Close price
                        else:
                            exit_i.append(close_data[i][j])
                            date_i.append(date_data[i][j])
                # Short position
                else:
                    # Short position - stopped at Open price
                    if open_data[i][j] > stop_loss_data[i]:
                        exit_i.append(open_data[i][j])
                        date_i.append(date_data[i][j])
                        stopped = 1
                    else:
                        # Short position - stopped at STOP LOSS PRICE during the trading day, Realized R = -1
                        if stop_loss_data[i] < close_data[i][j]:
                            if stop_intraday is True:
                                exit_i.append(stop_loss_data[i])  # Allowed to stop at open price
                                stopped = 1
                            else:
                                exit_i.append(close_data[i][j])  # Not allowed to stop at open price
                            date_i.append(date_data[i][j])
                        else:
                            # Short position - close position at Close price
                            exit_i.append(close_data[i][j])
                            date_i.append(date_data[i][j])

        exit_database.append([exit_i, date_i])
        stopped_record.append(stopped)
    return exit_database, stopped_record


def original_exit_index(trades_df, close_price_database):
    # Used in backtesting the original trading records, output the index of the holding dates for each trade.
    mapping_index = {}
    for ticker, row in trades_df.iterrows():
        dates = close_price_database[ticker][row['Entry Date']]['dates']

        if row['Exit Date'] in dates:
            mapping_index[ticker] = dates.index(row['Exit Date']) + 1
        else:
            mapping_index[ticker] = len(dates)
    return mapping_index


def refresh_df(trades_df, open_price_database, close_price_database, dates_mapping_index, atr_x=None,
               holding_period=None):
    # Backtest mechanism, refresh the trade_df with new Exit Price, Exit Date, Realized R, Realized gains for a given set of ATR(x), holding_period
    df = trades_df.copy(deep=True)

    if atr_x is not None:
        df['ATR(x)'] = atr_x
        df['ATR Used'] = df.loc[:]['ATR'] * atr_x
        df['STOP LOSS PRICE'] = df.loc[:]['Entry Price'] - df.loc[:]['side2'] * df.loc[:]['ATR Used']

    if holding_period is None:
        # Original holding period
        # dates_mapping_index = original_exit_index(trades_df, close_price_database)
        CLOSE_DATA = [[close_price_database[_ticker][_row['Entry Date']]['dates'][:dates_mapping_index[_ticker]],
                       close_price_database[_ticker][_row['Entry Date']]['price'][:dates_mapping_index[_ticker]]]
                      for _ticker, _row in df.iterrows()]

        OPEN_DATA = [[open_price_database[_ticker][_row['Entry Date']]['dates'][:dates_mapping_index[_ticker]],
                      open_price_database[_ticker][_row['Entry Date']]['price'][:dates_mapping_index[_ticker]]]
                     for _ticker, _row in df.iterrows()]

        EXIT_DATA, STOPPED_RECORD = get_exit_data(df, CLOSE_DATA, OPEN_DATA, dates_mapping_index)

        df['Exit Price'] = [x[0][-1] for x in EXIT_DATA]
        df['Exit Date'] = [x[1][-1] for x in EXIT_DATA]
        df['Stopped'] = STOPPED_RECORD

    else:
        CLOSE_DATA = np.array([[close_price_database[_ticker][_row['Entry Date']]['dates'][:holding_period],
                                close_price_database[_ticker][_row['Entry Date']]['price'][:holding_period]]
                               for _ticker, _row in df.iterrows()])

        OPEN_DATA = np.array([[open_price_database[_ticker][_row['Entry Date']]['dates'][:holding_period],
                               open_price_database[_ticker][_row['Entry Date']]['price'][:holding_period]]
                              for _ticker, _row in df.iterrows()])

        EXIT_DATA, STOPPED_RECORD = get_exit_data(df, CLOSE_DATA, OPEN_DATA, dates_mapping_index,
                                                  holding_period=holding_period)

        df['Exit Price'] = np.array(EXIT_DATA)[:, 0, holding_period - 1]
        df['Exit Date'] = np.array(EXIT_DATA)[:, 1, holding_period - 1]
        df['Stopped'] = STOPPED_RECORD

    df['Realized R'] = (df['Exit Price'] - df['Entry Price']) / df['ATR Used'] * df['side2']
    if 'Realized gains' in df.columns and 'Position(R)' in df.columns:
        df['Realized gains'] = (df['Exit Price'] - df['Entry Price']) * df['Position(R)'] * df['side2']
    return df


def evaluate_trades(df):
    # Evaluate trades based on Sum of R, Hit ratio, Expectancy.
    # Total:
    df = df.copy(deep=True)
    df = preprocessing(df)

    # Sum of R
    s_r = sum_R(df)
    print('Sum of R: %.2f' % s_r)

    # hit ratio
    h_r = hit_ratio(df)
    print("Hit ratio: %.4f" % h_r)

    # expectancy
    e = get_expectancy(df)
    print('Expectancy: %.4f' % e[0])

    return s_r, h_r, e[0]


def backtest_parameter(df, open_price_database, close_price_database, dates_mapping_index, parameter="atr"):
    # Backtest ATR / Holding period
    def backtest_atr(df):
        df = df.copy(deep=True)
        results = [0]

        for atr_x in np.arange(1, 40):
            results.append(
                sum_R(refresh_df(df, open_price_database, close_price_database, dates_mapping_index, atr_x=atr_x)))

        return results

    def backtest_period(df):
        df = df.copy(deep=True)
        results = []

        for holding_period in range(1, 6):
            results.append(sum_R(refresh_df(df, open_price_database, close_price_database, dates_mapping_index,
                                            holding_period=holding_period)))

        return results

    # Sum of R against ATR(x)
    # df = df.copy(deep=True)

    if parameter == "atr":
        results = backtest_atr(df)
        title = "Returns / stop-loss"
        x_label = "ATR"
    else:
        results = backtest_period(df)
        title = "Returns / holding period"
        x_ticks = range(1, 6)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), dpi=150)
    ax.plot(results, c='green', label='Return')
    ax.fill_between(range(len(results)), 0, results, facecolor='green', alpha=0.3)
    ax.yaxis.set_ticks_position('right')
    ax.set_ylabel('Risk adjusted return (R)')

    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
    # plt.grid(False)
    plt.title(title)
    plt.legend()
    # plt.show()
    return fig


def trades_matrix_plot(matrix, title='', precision=2):
    # Basic matrix plot, with 1,2,3, for ATR as y-axis, and only 1-5 Days holding Period as x-axis
    plt.figure(figsize=(16, 6), dpi=150)
    plt.title(title, fontsize=14)
    sns.heatmap(get_df_plot(matrix), linewidths=.5, cmap='BuPu', annot=True, fmt=f'.{precision}f')
    # plt.show()


def trades_to_matrix(trades_df, open_price_database, close_price_database, dates_mapping_index):
    # Transform trades_df to matrix for better calculation and plotting
    output_matrix_total = list()
    output_matrix_long = list()
    output_matrix_short = list()
    for atr in range(1, 4):
        _row_total = []
        _row_long = []
        _row_short = []
        for holding_period in range(1, 6):
            print(f"Strategy | ATR={atr} | Holds {holding_period} days")
            df = refresh_df(trades_df, open_price_database, close_price_database, dates_mapping_index, atr,
                            holding_period)
            long_trades_df = df[df['side'] == 'long']
            short_trades_df = df[df['side'] == 'short']
            # evaluate_trades(df)
            _row_total.append(list(evaluate_trades(df)))
            _row_long.append(list(evaluate_trades(long_trades_df)))
            _row_short.append(list(evaluate_trades(short_trades_df)))
        output_matrix_total.append(_row_total)
        output_matrix_long.append(_row_long)
        output_matrix_short.append(_row_short)

    return np.array(output_matrix_total), np.array(output_matrix_long), np.array(output_matrix_short)


def get_df_plot(m):
    # Return dataframe for plotting
    x_tickers = ['Hold %d day' % _i for _i in range(1, 6)]
    y_tickers = ['ATR(%d)' % _i for _i in range(1, 4)]
    return pd.DataFrame(m, index=y_tickers, columns=x_tickers)


def trades_comparison_plot(matrix1, matrix2, matrix3, title='', precision=2):
    # Sub plots for total trades, long trades, short trades
    y_tickers = ['ATR(%d)' % _i for _i in range(1, 4)]
    x_tickers = ['Hold %d day' % _i for _i in range(1, 6)]

    vmin = min([x.min() for x in [matrix1, matrix2, matrix3]])
    vmax = max([x.max() for x in [matrix1, matrix2, matrix3]])

    plt.figure(figsize=(16, 9), dpi=150)
    plt.suptitle(title, fontsize=14)

    plt.subplot(221)
    plt.title('Long Trades')
    sns.heatmap(get_df_plot(matrix1), linewidths=.5, cmap='BuPu', annot=True, fmt=f'.{precision}f',
                vmin=vmin, vmax=vmax, cbar=False)

    plt.subplot(222)
    plt.title('Short Trades')
    sns.heatmap(get_df_plot(matrix2), linewidths=.5, cmap='BuPu', annot=True, fmt=f'.{precision}f',
                vmin=vmin, vmax=vmax, cbar=False)

    plt.subplot(212)
    plt.title('Total Trades (Benchmark)')
    sns.heatmap(get_df_plot(matrix3), linewidths=.5, cmap='BuPu', annot=True, fmt=f'.{precision}f',
                vmin=vmin, vmax=vmax, cbar_kws=dict(use_gridspec=False, location='bottom', shrink=0.4))


def trades_evaluation_plot(matrix1, matrix2, matrix3, title='Strategy Performance                       '):
    # Evaluate trades in 3 sub plots in 3 evaluating criteria.
    plt.figure(figsize=(16, 10), dpi=150)
    plt.suptitle(title, fontsize=14)

    plt.subplot(311)
    plt.title('Sum of R')
    sns.heatmap(get_df_plot(matrix1), linewidths=.5, cmap='BuPu', annot=True, fmt='.2f', xticklabels=False)

    plt.subplot(312)
    plt.title('Hit ratio')
    sns.heatmap(get_df_plot(matrix2), linewidths=.5, cmap='BuPu', annot=True, fmt='.4f', xticklabels=False)

    plt.subplot(313)
    plt.title('Expectancy')
    sns.heatmap(get_df_plot(matrix3), linewidths=.5, cmap='BuPu', annot=True, fmt='.2f')


def plot_r_distribution(df, title=''):
    # Plot R distribution for a given set of trades dataset.
    long_trades_r = df[df['side'] == 'long']['Realized R']
    short_trades_r = df[df['side'] == 'short']['Realized R']

    r_max = max(long_trades_r.max(), short_trades_r.max())
    r_min = max(long_trades_r.min(), short_trades_r.min())

    bins = np.linspace(r_min, r_max, 50)
    x_range = np.arange(r_min, r_max, 0.0001)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), dpi=150)
    plt.style.use('seaborn-paper')
    plt.title(title)
    plt.hist([short_trades_r, long_trades_r], bins, label=['Short', 'Long'])

    ymin, ymax = ax.get_ylim()
    # plt.ylim(ymin, ymax)
    ax.fill_between(x_range, ymin, ymax, where=x_range <= 0, facecolor='red', alpha=0.3)
    ax.fill_between(x_range, ymin, ymax, where=x_range >= 0, facecolor='green', alpha=0.3)
    plt.text(1, ymax - ymax / 35, 'Profit', fontsize=12)
    plt.text(-1, ymax - ymax / 35, 'Losses', fontsize=12)

    plt.xticks(np.arange(r_min, r_max, 0.2), rotation=90)
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(axis='x')
    plt.legend(loc='upper right')
    # # plt.show()
    return fig


def get_optimal_paramter(df):
    # Return the optimal atr and optimal period given a certain criteria
    optimal_atr, optimal_holding_period = [x[0] + 1 for x in np.where(df == np.amax(df))]
    return optimal_atr, optimal_holding_period
