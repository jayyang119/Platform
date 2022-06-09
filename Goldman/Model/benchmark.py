# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:46:38 2022

@author: JayYang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from random import sample
from Broker import get_pnl
from uti import DataLoader, Logger
from Backtest import visual
from Model.settings import DataCleaner
from Backtest.settings import get_expectancy
from Backtest import backtest_engine, plot_matrix
from Model.settings import benchmark_filter
from Model.LR import LR
from library import Dataset
from itertools import product

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path
DC = DataCleaner()
Engine = backtest_engine()
lr = LR()


def benchmark_expectancy(sort_by='exch_location', train_data=None, test_data=None):
    """
        This function calculates the historical expectancy of train_data, and maps to test_data as scores,
        given sort_by as the factor.
    """
    column = 'd0_r'
    if train_data is None and test_data is None:
        train_data, test_data = DC.get_benchmark_test_data()
        if 'side' not in train_data.columns:
            train_data['side'] = ''
        if 'side' not in test_data.columns:
            test_data['side'] = ''
        train_data = benchmark_filter(train_data)
        train_data = get_pnl(train_data)

    elif train_data is None:
        train_data = DL.loadDB('Backtest/Benchmark data/benchmark.csv')

    blind_data = test_data.copy()
    test_data = benchmark_filter(test_data, bt=False)
    blind_data = pd.concat([test_data, blind_data], axis=0).drop_duplicates(subset=['Time', 'Ticker'], keep=False)

    train_expectancy = get_expectancy(train_data, column,
                                      inputs=['No. of trades', column, sort_by, 'side', 'Report Type'],
                                      group_by=[sort_by, 'side', 'Report Type'])
    for by_i in test_data[sort_by].unique():
        for side in ['long', 'short']:
            for report_type in test_data['Report Type'].unique():
                if (by_i, side, report_type) not in train_expectancy.index:
                    train_expectancy.loc[(by_i, side, report_type)] = [0] * len(train_expectancy.columns)

    blind_data['Expectancy'] = 0
    test_data['Expectancy'] = [train_expectancy.loc[(x[sort_by], x['side'], x['Report Type'])]['Expectancy']
                               for _, x in test_data[[sort_by, 'side', 'Report Type']].iterrows()]
    test_data = pd.concat([test_data, blind_data], axis=0).drop_duplicates()

    # Blind long short expectancy
    blind_long = DL.loadDB(f'{DATABASE_PATH}/Backtest/Benchmark data/blind_long.csv')
    blind_short = DL.loadDB(f'{DATABASE_PATH}/Backtest/Benchmark data/blind_short.csv')

    blind_long_expectancy = get_expectancy(blind_long, column,
                                           inputs=['No. of trades', column, sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type'],
                                           group_by=[sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type'])
    blind_short_expectancy = get_expectancy(blind_short, column,
                                            inputs=['No. of trades', column, sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type'],
                                            group_by=[sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type'])

    for by_i in blind_data[sort_by].unique():
        for head_senti in blind_data['Headline sentiment'].unique():
            for sum_senti in blind_data['Summary sentiment'].unique():
                for report_type in blind_data['Report Type'].unique():
                    if (by_i, head_senti, sum_senti, report_type) not in blind_long_expectancy.index:
                        blind_long_expectancy.loc[(by_i, head_senti, sum_senti, report_type)] = [0] * len(blind_long_expectancy.columns)
                    if (by_i, head_senti, sum_senti, report_type) not in blind_short_expectancy.index:
                        blind_short_expectancy.loc[(by_i, head_senti, sum_senti, report_type)] = [0] * len(blind_short_expectancy.columns)

    test_data['Expectancy (blind long)'] = [blind_long_expectancy.loc[(x[sort_by], x['Headline sentiment'], x['Summary sentiment'], x['Report Type'])]['Expectancy']
                                            for _, x in test_data[[sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type']].iterrows()]

    test_data['Expectancy (blind short)'] = [blind_short_expectancy.loc[(x[sort_by], x['Headline sentiment'], x['Summary sentiment'], x['Report Type'])]['Expectancy']
                                             for _, x in test_data[[sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type']].iterrows()]

    logger.info(test_data)
    return test_data


region_mapping = {'Europe': 'Europe', 'Korea & Japan': 'Asia',
                  'Southeast Asia': 'Southeast Asia', 'Hong Kong': 'Asia',
                  'Taiwan': 'Asia',
                  'South Africa': 'South Africa'}

MARKET_GROUPING_DICT = {'Japan': 'Korea & Japan', 'South Korea': 'Korea & Japan', 'Philipines': 'Southeast Asia',
                        'Singapore': 'Southeast Asia', 'Malaysia': 'Southeast Asia', 'Indonesia': 'Southeast Asia',
                        'Thailand': 'Southeast Asia', 'Taiwan': 'Taiwan', 'Europe': 'Europe', 'Hong Kong': 'Hong Kong',
                        'United States': 'Americas', 'Canada': 'Americas', 'South Africa': 'South Africa'}


def sub_ele_index(ele, row):
    """
        This function returns a tuple of multi-column index for mapping expectancy in test data.
    """
    result = []
    for sub_ele in ele:
        result.append(row[sub_ele])
    return tuple(result)


def get_daily_trade(train_data, test_data, score_weights=[], intercept=0):
    """
        Given train_data, this function calculates the historical expectancy and maps to test_data.
    """
    train_data['exch_region'] = train_data['exch_region'].replace(region_mapping)
    train_data = train_data[train_data['exch_region'].isin(region_mapping.values())]
    train_data = train_data[
        ~train_data['exch_location'].isin(['China', 'Ireland', 'Switzerland', 'Finland', 'Greece'])].reset_index(
        drop=True)

    test_data['exch_region'] = test_data['exch_region'].replace(region_mapping)
    test_data = test_data[test_data['exch_region'].isin(region_mapping.values())]
    test_data = test_data[
        ~test_data['exch_location'].isin(['China', 'Ireland', 'Switzerland', 'Finland', 'Greece'])].reset_index(
        drop=True)

    all_data = pd.concat([train_data, test_data], axis=0).copy()

    elements = [
        ['exch_location', 'Headline sentiment', 'Summary sentiment'],
        ['Report Type', 'Headline sentiment', 'Summary sentiment'],
        ['Sector', 'Headline sentiment', 'Summary sentiment'],
        ['market_cap_grp', 'Headline sentiment', 'Summary sentiment']
    ]
    elements_name = []

    if len(score_weights) == 0:
        score_weights = [0.2] * len(elements)
    for ele in elements:
        expectancy = get_expectancy(train_data, group_by=ele)

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
            test_data[f'{ele_name}'] = [expectancy.loc[sub_ele_index(ele, row)]['Expectancy'] for _, row in
                                        test_data[ele].iterrows()]
        else:
            ele_name = ele[0] + '_score'
            test_data[f'{ele_name}'] = test_data[ele[0]].map(
                dict(expectancy['Expectancy'])).fillna(0)
        print(ele_name)
        elements_name.append(ele_name)
        print(elements_name)
    test_data['score'] = np.sum(intercept + score_weights * test_data[elements_name], axis=1)
    test_data['delta_r'] = (test_data['d0_close'] - test_data['d0_open']) / test_data['atr_used']

    #     test_data.loc[test_data['score']>0, 'score'] = 0

    return test_data, elements_name


def simulation(range_of_test=range(10), from_local=False, exclude=[]):
    """
        Monte carlo simulation of a set a benchmark rules from get_daily_trade.
    """
    train_data, test_data = DC.get_benchmark_test_data()
    train_data['exch_location2'] = train_data['exch_location'].copy()
    test_data['exch_location2'] = test_data['exch_location'].copy()

    train_data.loc[train_data['exch_region'] == 'Europe', 'exch_location2'] = 'Europe'
    test_data.loc[test_data['exch_region'] == 'Europe', 'exch_location2'] = 'Europe'
    train_data['exch_region'] = train_data['exch_location2'].map(MARKET_GROUPING_DICT).fillna('Europe')
    test_data['exch_region'] = test_data['exch_location2'].map(MARKET_GROUPING_DICT).fillna('Europe')

    dfall_5DR = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    different_test_R = {}
    print(f"Simulating {max(range_of_test)} times.")
    for k in range_of_test:
        logger.info(f"{k} trial simulation")

        trading_R_details = []
        if not from_local:
            testing_date = sample(set(dfall_5DR['Date']), 252)

            training_data = dfall_5DR.loc[~dfall_5DR['Date'].isin(testing_date)].copy(deep=True)
            testing_data = dfall_5DR.loc[dfall_5DR['Date'].isin(testing_date)].copy(deep=True)

            daily_trade, elements_name = get_daily_trade(training_data, testing_data)
            model = lr(daily_trade[elements_name], daily_trade[['d0_r']])
            model.train()
            model.evaluate()
            # intercept = -0.03506858
            # ols_weights = [0.86667071, 0.03735833, 0.16950042]
            intercept, ols_weights = model.get_params()
            output['score'] = np.sum(ols_weights * output[elements_name], axis=1) + intercept[0]

            output = output.sort_values('Date')

            final_trades = Engine.portfolio_management(output)
            DL.toBT(final_trades, f'Simulation/{k} trial')
        else:
            if DL.checkDB(f'Backtest/Simulation/{k} trial.csv'):
                final_trades = DL.loadBT(f'Simulation/{k} trial')
                if len(final_trades) == 0:
                    continue
                final_trades = final_trades[~final_trades['exch_region'].isin(exclude)]
            else:
                continue

        today_R = final_trades.set_index('Date')['R0']
        R_in_this_test = final_trades.groupby('Date')['R0'].sum()
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
        axes[0][0].plot(equity.index, equity, linewidth=1)
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

    # train_data, test_data = DC.get_benchmark_test_data()
    # results = pd.concat([train_data, test_data], axis=0)
    # results['side'] = 'long'
    # results = get_pnl(results)

    results = DL.loadDB('Backtest/Benchmark data/blind_long.csv')
    DL.toBT(results, 'simple long')
    plot_matrix('simple long', by='exch_location')

    results = DL.loadDB('Backtest/Benchmark data/blind_short.csv')
    DL.toBT(results, 'simple short')
    plot_matrix('simple short', by='exch_location')

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


    # simulation_result = simulation(range(1, 200), from_local=True, exclude=['Americas'])
    # equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b = simulation_datecleaning(simulation_result)
    # simulation_visualization(equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b)

    # results = benchmark_strategy(sort_by='exch_region')
    # pnl_df = Engine.portfolio_management(results)
    # strategy = 'Headline strategy (scoring on side, exch_region, report type)'
    # DL.toDB(pnl_df, f'Backtest/{strategy}.csv')

    # vis = visual(strategy)
    # vis.visual_job()

    # plot_matrix()