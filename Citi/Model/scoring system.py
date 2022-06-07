# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:46:38 2022

@author: JayYang
"""
import pandas as pd
import numpy as np
from random import sample

from Broker import get_pnl
from uti import DataLoader, Logger
from Model.settings import DataCleaner
from Backtest.settings import get_expectancy
from Backtest import BacktestEngine, plot_matrix
from Model.rules import benchmark_filter, region_case_study
from library import Dataset
import itertools
from itertools import product
from Model.LR import LR
import matplotlib.pyplot as plt

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path
DC = DataCleaner()
Engine = BacktestEngine()

REGION_MAPPING_DICT = {'Korea & Japan': 'Asia',
                       'Hong Kong': 'Asia', 'Taiwan & Singapore': 'Asia',
                       'Southeast Asia': 'Asia',
                       'Asia exotic': 'Asia',
                       'South Africa': 'South Africa',
                       'Americas': 'Americas',
                       'China': 'Asia'}

MARKET_MAPPING_DICT = {'Japan': 'Korea & Japan', 'South Korea': 'Korea & Japan',
                        'Hong Kong': 'Hong Kong',
                        'Taiwan': 'Taiwan & Singapore', 'Singapore': 'Taiwan & Singapore',
                        'Malaysia': 'Southeast Asia', 'Indonesia': 'Southeast Asia', 'Philipines': 'Southeast Asia', 'Thailand': 'Southeast Asia', 'Philippines': 'Southeast Asia',
                        'United States': 'Americas', 'Canada': 'Americas',
                        'South Africa': 'South Africa'}


def benchmark_expectancy(sort_by='exch_region', train_data=None, test_data=None):
    if train_data is None and test_data is None:
        train_data, test_data = DC.get_benchmark_test_data()
        if 'side' not in train_data.columns:
            train_data['side'] = ''
        if 'side' not in test_data.columns:
            test_data['side'] = ''
        train_data = benchmark_filter(train_data)
        train_data = get_pnl(train_data)

    elif train_data is None:
        training_data, testing_data = DC.get_benchmark_test_data()
        train_data = pd.concat([training_data, testing_data], axis=0)

    # test_data, elements_name = get_daily_trade(train_data, test_data)
    # model = LR(train_data[elements_name], train_data[['d0_r']])
    # model.train()
    # model.evaluate()
    # intercept, ols_weights = model.get_params()
    #
    # test_data['score'] = np.sum(ols_weights * test_data[elements_name], axis=1) + intercept[0]
    # test_data.loc[test_data['score'] < 0, 'score'] = 0


    # Blind long short expectancy
    blind_long = DL.loadBT('blind long')
    blind_short = DL.loadBT('blind short')

    blind_long['exch_region2'] = blind_long['exch_region']
    blind_short['exch_region2'] = blind_short['exch_region']
    test_data['exch_region2'] = test_data['exch_region']

    blind_long['exch_location'] = blind_long['exch_location'].replace(MARKET_MAPPING_DICT)
    blind_short['exch_location'] = blind_short['exch_location'].replace(MARKET_MAPPING_DICT)
    test_data['exch_location'] = test_data['exch_location'].replace(MARKET_MAPPING_DICT)

    blind_long['exch_region'] = blind_long['exch_location'].replace(REGION_MAPPING_DICT)
    blind_short['exch_region'] = blind_short['exch_location'].replace(REGION_MAPPING_DICT)
    test_data['exch_region'] = test_data['exch_location'].replace(REGION_MAPPING_DICT)

    blind_long.loc[blind_long['exch_region2'] == 'Europe', 'exch_region'] = 'Europe'
    blind_short.loc[blind_short['exch_region2'] == 'Europe', 'exch_region'] = 'Europe'
    test_data.loc[test_data['exch_region2'] == 'Europe', 'exch_region'] = 'Europe'

    blind_long['Sector'] = blind_long['Sector'].fillna('').apply(lambda x: x.split(',')[0])
    blind_short['Sector'] = blind_short['Sector'].fillna('').apply(lambda x: x.split(',')[0])
    test_data['Sector'] = test_data['Sector'].fillna('').apply(lambda x: x.split(',')[0])

    ele_group = ['Headline sentiment', 'Summary sentiment', 'Report Type']
    blind_long_expectancy = get_expectancy(blind_long, group_by=ele_group)
    blind_short_expectancy = get_expectancy(blind_short, group_by=ele_group)

    unique_values = []
    for sub_ele in ele_group:
       unique_values.append(blind_long[sub_ele].unique())

    combinations = itertools.product(*unique_values)
    case_unfound = set(combinations).difference(set(blind_long_expectancy.index))
    case_unfound_df = pd.DataFrame(0, index=case_unfound, columns=blind_long_expectancy.columns)

    blind_long_expectancy = pd.concat([blind_long_expectancy, case_unfound_df], axis=0)
    blind_short_expectancy = pd.concat([blind_short_expectancy, case_unfound_df], axis=0)


    # test_data['Expectancy (blind long)'] = [blind_long_expectancy.loc[ele]['Expectancy'] for ele in test_data[ele_group].iterrows()]
    # test_data['Expectancy (blind short)'] = [blind_short_expectancy.loc[ele]['Expectancy'] for ele in test_data[ele_group].iterrows()]
    #
    test_data['Expectancy (blind long)'] = [blind_long_expectancy.loc[(x['Headline sentiment'], x['Summary sentiment'], x['Report Type'])]['Expectancy']
                                            for _, x in test_data[['Headline sentiment', 'Summary sentiment', 'Report Type']].iterrows()]
    #
    test_data['Expectancy (blind short)'] = [blind_short_expectancy.loc[(x['Headline sentiment'], x['Summary sentiment'], x['Report Type'])]['Expectancy']
                                             for _, x in test_data[['Headline sentiment', 'Summary sentiment', 'Report Type']].iterrows()]

    logger.info(test_data)
    return test_data



def sub_ele_index(ele, row):
    result = []
    for sub_ele in ele:
        result.append(row[sub_ele])
    return tuple(result)


def get_daily_trade(_train_data, _test_data, score_weights=[], intercept=[0]):
    exclusion_list = ['Australia', 'Turkey', 'Czech Republic', 'Poland', 'Greece', 'Americas']
    training_data = _train_data[~_train_data['exch_location'].isin(exclusion_list)].reset_index(drop=True).copy()
    testing_data = _test_data[~_test_data['exch_location'].isin(exclusion_list)].reset_index(drop=True).copy()

    all_data = pd.concat([training_data, testing_data], axis=0).copy()

    elements = [
        #                 ['Ticker'],
        ['Head analyst'],
        ['exch_region'],
        ['Report Type'],
        ['Sector', 'market_cap_grp'],
    ]
    elements_name = []

    if len(score_weights) == 0:
        score_weights = [0.2] * len(elements)
    for ele in elements:
        expectancy = get_expectancy(training_data, group_by=ele)

        unique_values = []
        if len(ele) > 1:
            for sub_ele in ele:
                unique_values.append(all_data[sub_ele].unique())
            combinations = product(*unique_values)

        else:
            combinations = all_data[ele[0]].unique()

        case_unfound = set(combinations).difference(set(expectancy.index))
        case_unfound_df = pd.DataFrame(0, index=case_unfound, columns=expectancy.columns)
        expectancy = pd.concat([expectancy, case_unfound_df], axis=0)

        print(expectancy)

        if len(ele) > 1:
            ele_name = '_'.join(ele) + '_score'
            training_data[f'{ele_name}'] = [expectancy.loc[sub_ele_index(ele, row)]['Expectancy'] for _, row in
                                            training_data[ele].iterrows()]
            testing_data[f'{ele_name}'] = [expectancy.loc[sub_ele_index(ele, row)]['Expectancy'] for _, row in
                                           testing_data[ele].iterrows()]
        else:
            ele_name = ele[0] + '_score'
            training_data[f'{ele_name}'] = training_data[ele[0]].map(dict(expectancy['Expectancy'])).fillna(0)
            testing_data[f'{ele_name}'] = testing_data[ele[0]].map(dict(expectancy['Expectancy'])).fillna(0)
        print(ele_name)
        elements_name.append(ele_name)
        print(elements_name)

    training_data['score'] = np.sum(score_weights * training_data[elements_name], axis=1) + intercept[0]
    testing_data['score'] = np.sum(score_weights * testing_data[elements_name], axis=1) + intercept[0]

    #     test_data.loc[test_data['score'] < 0, 'side'] = ''
    #     test_data.loc[test_data['score'] < 0, 'score'] = 0

    return training_data, testing_data, elements_name


def benchmark_rule(df_):
    df = df_[['Headline sentiment', 'Summary sentiment']].copy()

    #     long_mask = ((df['Headline sentiment'] == 'positive') & (df['Summary sentiment'] != 'negative')) | ((df['Summary sentiment'] == 'positive') & (df['Headline sentiment'] != 'negative'))
    long_mask = ((df['Headline sentiment'] == 'positive') & (df['Summary sentiment'] == 'positive'))
    short_mask = (df['Headline sentiment'] == 'negative') | (df['Summary sentiment'] == 'negative')

    df['side'] = np.where(long_mask, 'long', 'neutral')
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']

def benchmark_rule2(df_):
    df = df_[['TPS', 'TPS_prev', 'RC_upgrade', 'RC_downgrade']]

    long_mask = (df['TPS_prev'] < df['TPS'])
    short_mask = (df['TPS_prev'] > df['TPS'])

    df['side'] = np.where(long_mask, 'long', 'neutral')
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']

def benchmark_rule3(df_):
    df = df_[['RC_upgrade', 'RC_downgrade']]

    long_mask = ((df['RC_upgrade'] == 'Y') & (df['RC_downgrade'] != 'Y'))
    short_mask = ((df['RC_upgrade'] != 'Y') & (df['RC_downgrade'] == 'Y'))

    df['side'] = np.where(long_mask, 'long', 'neutral')
    df['side'] = np.where(short_mask, 'short', df['side'])

    return df['side']


def simulation(range_of_test=range(10), from_local=False, exclude=[]):
    train_data, test_data = DC.get_benchmark_test_data()
    train_data['exch_region2'] = train_data['exch_region']
    test_data['exch_region2'] = test_data['exch_region']

    train_data['exch_location'] = train_data['exch_location'].replace(MARKET_MAPPING_DICT)
    test_data['exch_location'] = test_data['exch_location'].replace(MARKET_MAPPING_DICT)

    train_data['exch_region'] = train_data['exch_location'].replace(REGION_MAPPING_DICT)
    test_data['exch_region'] = test_data['exch_location'].replace(REGION_MAPPING_DICT)

    train_data.loc[train_data['exch_region2'] == 'Europe', 'exch_region'] = 'Europe'
    test_data.loc[test_data['exch_region2'] == 'Europe', 'exch_region'] = 'Europe'

    train_data['Sector'] = train_data['Sector'].fillna('').apply(lambda x: x.split(',')[0])
    test_data['Sector'] = test_data['Sector'].fillna('').apply(lambda x: x.split(',')[0])

    dfall_5DR = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    different_test_R = {}
    print(f"Simulating {max(range_of_test)} times.")
    for k in range_of_test:
        logger.info(f"{k} trial simulation")

        trading_R_details = []
        if not from_local:
            testing_date = sample(set(dfall_5DR['Date']), 252)

            training_data = dfall_5DR.loc[~dfall_5DR['Date'].isin(testing_date)].reset_index(drop=True).copy(deep=True)
            testing_data = dfall_5DR.loc[dfall_5DR['Date'].isin(testing_date)].reset_index(drop=True).copy(deep=True)

            _, output, elements_name = get_daily_trade(training_data, testing_data)
            output['score'] = np.mean(output[elements_name], axis=1)

            output = output.sort_values('Date', ascending=True)
            final_trades = Engine.portfolio_management(output, random_pick=False, rank_by='score')
            DL.toBT(final_trades, f'Simulation/{k} trial')

        else:
            if DL.checkDB(f'Backtest/Simulation/{k} trial.csv'):
                final_trades = DL.loadBT(f'Simulation/{k} trial')
                if len(final_trades) == 0:
                    continue
                final_trades = final_trades[~final_trades['exch_region'].isin(exclude)]
            else:
                continue

        today_R = final_trades.set_index('Date')['d0_r']
        R_in_this_test = final_trades.groupby('Date')['d0_r'].sum()
        num_trades_in_this_test = final_trades.groupby('Date')['No. of trades'].sum()
        trading_R_details = trading_R_details + today_R.values.tolist()

        different_test_R[f"{k} trial test"] = [R_in_this_test, num_trades_in_this_test, trading_R_details]
    return different_test_R


def simulation_datecleaning(sim_result):
    equity_curve, num_trades, hit_ratio, expectancy = [], [], [], []
    for i in sim_result.keys():
        R_data_details = pd.Series(sim_result[i][2])

        equity_curve.append(pd.Series(sim_result[i][0]).cumsum())
        num_trades.append(len(R_data_details))
        hit_ratio.append(sum(R_data_details > 0) / len(R_data_details))
        expectancy.append(sum(R_data_details) / len(R_data_details))
    return equity_curve, num_trades, hit_ratio, expectancy


def simulation_visualization(equity_curve, num_trades, hit_ratio, expectancy, scoring_type=""):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
    plt.grid()
    fig.patch.set_facecolor('xkcd:gray')

    for equity in equity_curve:
        axes[0][0].plot(range(len(equity)), equity, linewidth=1)
    # axes[0][0].set_xticks(equity.index, range(len(equity)))

    pd.Series(num_trades).plot(kind="hist", bins=40, ax=axes[0][1], cmap="coolwarm_r", density=True, alpha=0.6)
    pd.Series(num_trades).plot(kind="kde", ax=axes[0][1], cmap="coolwarm_r", linewidth=2.5)

    pd.Series(hit_ratio).plot(kind="hist", bins=40, ax=axes[1][0], cmap="pink", density=True, alpha=0.6)
    pd.Series(hit_ratio).plot(kind="kde", ax=axes[1][0], cmap="pink", linewidth=2.5)

    pd.Series(expectancy).plot(kind="hist", bins=40, ax=axes[1][1], cmap="Accent", density=True, alpha=0.6)
    pd.Series(expectancy).plot(kind="kde", ax=axes[1][1], cmap="Accent", linewidth=2.5)

    axes[0][0].set_title(scoring_type + "Scoring System R Curve Simulation", fontsize=25)
    axes[0][1].set_xlabel("Number of Trades", fontsize=20)
    axes[0][1].set_title("Distribution of Number of Trades", fontsize=25)
    axes[1][0].set_xlabel("Hit Ratio", fontsize=20)
    axes[1][0].set_title("Distribution of Hit Ratio", fontsize=25)
    axes[1][1].set_xlabel("Expectancy", fontsize=20)
    axes[1][1].set_title("Distribution of Expectancy", fontsize=25)
    plt.show()



if __name__ == '__main__':

    column = 'd0_r'

    # strategy = 'Benchmark strategy (long pos short neg)'
    # strategy = 'Benchmark strategy (region specific rules (PM))'
    # strategy = f'Benchmark strategy (region specific rules after fees (PM))'
    # _, test_data = DC.get_benchmark_test_data()
    # test_data['side'] = test_data['Headline sentiment'].replace({'positive': 'long', 'negative': 'short'})
    # test_data = get_pnl(test_data)
    # test_data = DL.loadBT(strategy)
    # DS = Dataset(test_data)
    # DS.backtest()
    # results_after_fee = DS.df.copy(deep=True)
    # results_after_fee['d0_r'] = results_after_fee['d0_r'] - results_after_fee['fee_in_r']
    # results_after_fee['d1_r'] = results_after_fee['d1_r'] - results_after_fee['fee_in_r']
    # results_after_fee['d2_r'] = results_after_fee['d2_r'] - results_after_fee['fee_in_r']



    # price_df = DL.loadDB('price_df.csv')
    # # price_df = price_df[price_df['release_period'] != 'Within']
    # price_df['exch_location'] = price_df['exch_location'].replace(MARKET_MAPPING_DICT)
    # print(price_df['exch_location'].unique())
    # price_df.loc[price_df['exch_region']=='Europe', 'exch_location'] = 'Europe'
    # results = price_df[price_df['exch_location'].isin(['Korea & Japan', 'Taiwan', 'Hong Kong', 'Europe', 'Southeast Asia',
    #                                                    'South Africa', 'Americas'])].reset_index(drop=True)
    #

    all_data = DL.loadDB('tpc_df_combined.csv', parse_dates=['Date', 'Time'])
    all_data['TPS'] = all_data.groupby(['Ticker', 'Date'])['TPS_new'].apply(lambda x: x.fillna(method='ffill'))
    all_data['TPS_prev'] = all_data.groupby(['Ticker', 'Date'])['TPS_new'].apply(lambda x: x.shift(1))

    all_data = all_data.sort_values('Date', ascending=True)
    train_data = all_data.iloc[:len(all_data) * 0.75].copy()
    test_data = all_data.iloc[len(all_data) * 0.75:].copy()

    DL.toDB(train_data, 'Backtest/Benchmark data/train.csv')
    DL.toDB(test_data, 'Backtest/Benchmark data/train.csv')


    # Benchmark: long double positive, short negative
    # strategy = 'benchmark'
    # double_positive_mask = (results['Headline sentiment'] == 'positive') & (results['Summary sentiment'] == 'positive')
    # short_mask = (results['Headline sentiment'] == 'negative') | (results['Summary sentiment'] == 'negative')
    # results.loc[double_positive_mask, 'side'] = 'long'
    # results.loc[short_mask, 'side'] = 'short'
    # results = results.loc[double_positive_mask | short_mask].reset_index(drop=True)
    # results = get_pnl(results)
    # DL.toBT(results, strategy)
    # plot_matrix(strategy)

    # vis = visual(strategy)
    # vis.visual_job()

    # results = DL.loadBT(strategy)
    # results['Expectancy'] = 0.0
    # strategy = 'benchmark (PM)'
    # pnl_df = Engine.portfolio_management(results, random_pick=True)
    # DL.toBT(pnl_df, strategy)
    # plot_matrix(strategy)

    # vis = visual(strategy)
    # vis.visual_job()



    # strategy = 'Benchmark strategy (region specific rules (PM))'
    # results = DL.loadBT('Benchmark strategy (region specific rules)')
    # pnl_df = DL.loadBT(strategy)
    # pnl_df = Engine.portfolio_management(results)
    # pnl_df = pnl_df[~pnl_df['exch_region'].isin(['Americas'])]
    # strategy = 'Benchmark strategy (region specific rules exl Am (PM))'
    # DL.toBT(pnl_df, strategy)
    # plot_matrix(strategy)
    # vis = visual(strategy)
    # vis.visual_job()
    #
    # strategy = f'Benchmark strategy (region specific rules after fees (PM))'
    # strategy = 'Benchmark strategy (region specific rules exl Am (PM))'
    # pnl_df = DL.loadBT(strategy)
    # pnl_df_after_fee = pnl_df[~pnl_df['exch_region'].isin(['Americas'])]
    # DS = Dataset(pnl_df)
    # DS.backtest()
    # pnl_df_after_fee = DS.df.copy(deep=True)
    # pnl_df_after_fee['d0_r'] = pnl_df_after_fee['d0_r'] - pnl_df_after_fee['fee_in_r']
    # pnl_df_after_fee['d1_r'] = pnl_df_after_fee['d1_r'] - pnl_df_after_fee['fee_in_r']
    # pnl_df_after_fee['d2_r'] = pnl_df_after_fee['d2_r'] - pnl_df_after_fee['fee_in_r']
    # strategy = f'Benchmark strategy (region specific rules exl Am after fees (PM))'
    # DL.toBT(pnl_df_after_fee, strategy)
    # vis = visual(strategy)
    # vis.visual_job()
    # plot_matrix(strategy)


    # simulation_result = simulation(range(1, 101), from_local=False, exclude=['Americas'])
    # equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b = simulation_datecleaning(simulation_result)
    # simulation_visualization(equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b)

    # results = benchmark_strategy(sort_by='exch_region')
    # pnl_df = Engine.portfolio_management(results)
    # strategy = 'Headline strategy (scoring on side, exch_region, report type)'
    # DL.toDB(pnl_df, f'Backtest/{strategy}.csv')
    #
    # vis = visual(strategy)
    # vis.visual_job()

    # plot_matrix()