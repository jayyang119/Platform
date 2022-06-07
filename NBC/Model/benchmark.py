# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:46:38 2022

@author: JayYang
"""
import pandas as pd
import numpy as np
from random import sample
from Broker import get_pnl
from Model.rules import benchmark_rule, benchmark_rule2, benchmark_rule3
from uti import DataLoader, Logger
from Model.settings import DataCleaner
from Backtest.settings import get_expectancy
from Backtest import BacktestEngine
import itertools
from Model.LR import LR
from Backtest import visual

import matplotlib.pyplot as plt

from Backtest.visualize import plot_matrix

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path
DC = DataCleaner()
Engine = BacktestEngine()


def benchmark_expectancy(train_data=None, test_data=None):
    ######################## Scoring System ###################################
    if train_data is None and test_data is None:  # Default training data, testing data
        train_data, test_data = DC.get_benchmark_test_data()

    elif train_data is None:  # Testing data is defined, use all the historical price data to output scores
        training_data, testing_data = DC.get_benchmark_test_data()
        train_data = pd.concat([training_data, testing_data], axis=0)

    ####################### Blind long short expectancy ##################################
    blind_long = DL.loadBT('Benchmark data/blind long')
    blind_short = DL.loadBT('Benchmark data/blind short')

    blind_long = DC.preprocess_trade_df(blind_long)
    blind_short = DC.preprocess_trade_df(blind_short)

    ele_group = ['headline_senti', 'summary_senti', 'report_type']
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

    test_data['Expectancy (blind long)'] = [blind_long_expectancy.loc[(x['headline_senti'], x['summary_senti'], x['report_type'])]['Expectancy']
                                            for _, x in test_data[['headline_senti', 'summary_senti', 'report_type']].iterrows()]
    test_data['Expectancy (blind short)'] = [blind_short_expectancy.loc[(x['headline_senti'], x['summary_senti'], x['report_type'])]['Expectancy']
                                             for _, x in test_data[['headline_senti', 'summary_senti', 'report_type']].iterrows()]
    #######################################################################################



    # if not DL.checkDB('Backtest/Benchmark data/train temp.csv') and train_data is None:
    #     training_data = DL.loadDB('Backtest/Benchmark data/train.csv', parse_dates=['Date', 'Time'])
    #     testing_data = DL.loadDB('Backtest/Benchmark data/test.csv', parse_dates=['Date', 'Time'])
    #
    #     train_data = pd.concat([training_data, testing_data], axis=0)
    #
    #     train_data['side'] = benchmark_rule(train_data)
    #     train_data['side'] = benchmark_rule2(train_data)
    #     train_data['side'] = benchmark_rule3(train_data)
    #     train_data = get_pnl(train_data)
    #     DL.toDB(train_data, 'Backtest/Benchmark data/train temp.csv')
    # elif train_data is None:
    #     train_data = DL.loadDB('Backtest/Benchmark data/train temp.csv')

    # Apply Rules
    test_data['side'] = benchmark_rule(test_data)
    test_data['side'] = benchmark_rule2(test_data)
    test_data['side'] = benchmark_rule3(test_data)
    # test_data['side'] = test_data['side'].replace('neutral', '')
    train_data, test_data, elements_name = get_daily_trade(train_data, test_data)  # Calculate factor scores

    model = LR(train_data[elements_name], train_data[['d0_r']])
    model.train()
    model.evaluate()
    intercept, ols_weights = model.get_params()

    # Scoring: in Americas use OLS, in Asia and Europe use simple average.
    us_mask = (test_data['exch_region'] == 'Americas')
    test_data['ols_score'] = np.sum(ols_weights * test_data[elements_name], axis=1) + intercept[0]
    test_data['mean_score'] = np.mean(test_data[elements_name], axis=1)
    test_data['score'] = np.where(us_mask, test_data['ols_score'], test_data['mean_score'])
    test_data.loc[test_data['score'] <= 0, 'side'] = ''
    test_data.loc[test_data['score'] <= 0, 'score'] = 0
    test_data.loc[~test_data['side'].isin(['long', 'short']), 'score'] = 0

    test_data['d0_exp'] = np.where(us_mask, test_data['ols_score'], test_data['mean_score'])
    ###########################################################################

    ############################ Other Scores #################################
    ele_group = ['analyst_pri']
    analyst_long_expectancy = get_expectancy(blind_long, group_by=ele_group)
    analyst_short_expectancy = get_expectancy(blind_short, group_by=ele_group)

    unique_values = []
    for sub_ele in ele_group:
        unique_values.append(pd.concat([blind_long, test_data], axis=0)[sub_ele].unique())
    if len(unique_values) > 1:
        combinations = itertools.product(*unique_values)
    else:
        combinations = unique_values[0]

    case_absent = set(combinations).difference(set(analyst_long_expectancy.index))
    case_absent_df = pd.DataFrame(0, index=case_absent, columns=analyst_long_expectancy.columns)

    analyst_long_expectancy = pd.concat([analyst_long_expectancy, case_absent_df], axis=0)
    analyst_short_expectancy = pd.concat([analyst_short_expectancy, case_absent_df], axis=0)

    test_data['top_analyst_long'] = [
        analyst_long_expectancy.loc[x['analyst_pri']]['Expectancy'] for _, x in test_data[['analyst_pri']].iterrows()]
    test_data['top_analyst_short'] = [
        analyst_short_expectancy.loc[x['analyst_pri']]['Expectancy'] for _, x in test_data[['analyst_pri']].iterrows()]
    ###########################################################################
    # logger.info(test_data)
    return test_data


def sub_ele_index(ele, row):
    result = []
    for sub_ele in ele:
        result.append(row[sub_ele])
    return tuple(result)


def get_daily_trade(_train_data, _test_data, score_weights=[], intercept=[0]):
    exclusion_list = []
    training_data = _train_data[~_train_data['exch_location'].isin(exclusion_list)].reset_index(drop=True).copy()
    testing_data = _test_data[~_test_data['exch_location'].isin(exclusion_list)].reset_index(drop=True).copy()

    all_data = pd.concat([training_data, testing_data], axis=0).copy()

    elements = [
        ['analyst_pri'],
        ['ticker'],
        ['release_period'],
        ['report_type'],
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
            combinations = itertools.product(*unique_values)
        else:
            combinations = all_data[ele[0]].unique()

        case_absent = set(combinations).difference(set(expectancy.index))
        case_absent_df = pd.DataFrame(0, index=case_absent, columns=expectancy.columns)
        expectancy = pd.concat([expectancy, case_absent_df], axis=0)

        if len(ele) > 1:
            ele_name = '_'.join(ele) + '_score'
        else:
            ele_name = ele[0] + '_score'
        training_data[f'{ele_name}'] = [expectancy.loc[sub_ele_index(ele, row)]['Expectancy'] for _, row in
                                        training_data[ele].iterrows()]
        testing_data[f'{ele_name}'] = [expectancy.loc[sub_ele_index(ele, row)]['Expectancy'] for _, row in
                                       testing_data[ele].iterrows()]
        print(ele_name)
        elements_name.append(ele_name)
        print(elements_name)

    training_data['score'] = np.sum(score_weights * training_data[elements_name], axis=1) + intercept[0]
    testing_data['score'] = np.sum(score_weights * testing_data[elements_name], axis=1) + intercept[0]

    return training_data, testing_data, elements_name


def simulation(range_of_test=range(10), from_local=False, exclude=[]):
    # Preprocess
    # train_data = DL.loadDB('Backtest/Benchmark data/X train.csv', parse_dates=['publish_date_and_time'])
    # test_data = DL.loadDB('Backtest/Benchmark data/test.csv', parse_dates=['publish_date_and_time'])
    train_data, test_data = DC.get_benchmark_test_data()

    dfall_5DR = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    # blind_long = dfall_5DR.copy()
    # blind_long['side'] = 'long'
    # blind_long = get_pnl(blind_long)
    # # dfall_5DR = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    #
    # blind_short = dfall_5DR.copy()
    # blind_short['side'] = 'short'
    # blind_short = get_pnl(blind_short)
    #
    # dfall_5DR = pd.concat([blind_long, blind_short], axis=0).reset_index(drop=True)

    different_test_R = {}
    print(f"Simulating {max(range_of_test)} times.")
    for k in range_of_test:
        logger.info(f"{k} trial simulation")

        trading_R_details = []
        if not from_local:
            testing_date = sample(set(dfall_5DR['publish_date_and_time']), 252)

            # training_data = dfall_5DR.loc[~dfall_5DR['Date'].isin(testing_date)].reset_index(drop=True).copy(deep=True)
            # testing_data = dfall_5DR.loc[dfall_5DR['Date'].isin(testing_date)].reset_index(drop=True).copy(deep=True)
            #
            # ele_group = ['headline_senti', 'summary_senti', 'report_type']
            # blind_long_expectancy = get_expectancy(training_data[training_data['side']=='long'], group_by=ele_group)
            # blind_short_expectancy = get_expectancy(training_data[training_data['side']=='short'], group_by=ele_group)
            #
            # unique_values = []
            # for sub_ele in ele_group:
            #     unique_values.append(training_data[sub_ele].unique())
            #
            # combinations = itertools.product(*unique_values)
            # case_absent = set(combinations).difference(set(blind_long_expectancy.index))
            # case_absent_df = pd.DataFrame(0, index=case_absent, columns=blind_long_expectancy.columns)
            #
            # blind_long_expectancy = pd.concat([blind_long_expectancy, case_absent_df], axis=0)
            # blind_short_expectancy = pd.concat([blind_short_expectancy, case_absent_df], axis=0)
            #
            # testing_data_long = testing_data[testing_data['side'] == 'long'].copy().reset_index(drop=True)
            # testing_data_short = testing_data[testing_data['side'] == 'short'].copy().reset_index(drop=True)
            #
            # testing_data_long['Expectancy'] = [
            #     blind_long_expectancy.loc[(x['headline_senti'], x['summary_senti'], x['report_type'])][
            #         'Expectancy']
            #     for _, x in testing_data_long[['headline_senti', 'summary_senti', 'report_type']].iterrows()]
            #
            # testing_data_short['Expectancy'] = [
            #     blind_short_expectancy.loc[(x['headline_senti'], x['summary_senti'], x['report_type'])][
            #         'Expectancy']
            #     for _, x in testing_data_short[['headline_senti', 'summary_senti', 'report_type']].iterrows()]
            #
            # output = pd.concat([testing_data_long, testing_data_short], axis=0).reset_index(drop=True)

            training_data = dfall_5DR.loc[~dfall_5DR['publish_date_and_time'].isin(testing_date)].reset_index(drop=True).copy(deep=True)
            testing_data = dfall_5DR.loc[dfall_5DR['publish_date_and_time'].isin(testing_date)].reset_index(drop=True).copy(deep=True)
            training_data, output, elements_name = get_daily_trade(training_data, testing_data)
            #
            model = LR(training_data[elements_name], training_data[['d0_r']])
            model.train()
            model.evaluate()
            intercept, ols_weights = model.get_params()

            output['ols_score'] = np.sum(ols_weights * output[elements_name], axis=1) + intercept[0]
            # output['mean_score'] = np.mean(output[elements_name], axis=1)
            # output['score'] = np.where(us_mask, output['ols_score'], output['mean_score'])

            output = output.sort_values('publish_date_and_time', ascending=True)
            # final_trades = Engine.portfolio_management(output, random_pick=False, rank_by='score')
            final_trades = Engine.portfolio_management(output, random_pick=False, rank_by='ols_score')
            DL.toBT(final_trades, f'Simulation/{k} trial')

        else:
            if DL.checkDB(f'Backtest/Simulation/{k} trial.csv'):
                final_trades = DL.loadBT(f'Simulation/{k} trial')
                if len(final_trades) == 0:
                    continue
                final_trades = final_trades[~final_trades['exch_region'].isin(exclude)]
            else:
                continue

        today_R = final_trades.set_index('publish_date_and_time')['d0_r']
        R_in_this_test = final_trades.groupby('publish_date_and_time')['d0_r'].sum()
        final_trades['No. of trades'] = 1
        num_trades_in_this_test = final_trades.groupby('publish_date_and_time')['No. of trades'].sum()
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

    return fig



if __name__ == '__main__':
    train_data, test_data = DC.get_benchmark_test_data()
    all_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    plot_matrix('blind long')
    plot_matrix('blind short')

    # strategy = 'benchmark'
    # DL.toBT(all_data, strategy)
    # vis = visual(strategy)
    # vis.visual_job()
    # plot_matrix(strategy)
    # #
    # strategy = 'benchmark(pm random)'
    # after_pm = Engine.portfolio_management(all_data, random_pick=True)
    # DL.toBT(after_pm, strategy)
    # vis = visual(strategy)
    # vis.visual_job()
    # plot_matrix(strategy)
    #
    # #
    # strategy = 'benchmark(pm scoring)'
    # after_scoring = benchmark_expectancy()
    # after_scoring = Engine.portfolio_management(after_scoring, rank_by='d0_exp')
    # DL.toBT(after_scoring, strategy)
    # vis = visual(strategy)
    # vis.visual_job()
    # plot_matrix(strategy)
    # plot_matrix('benchmark(pm)')
    # all_data['side'] = benchmark_rule(all_data)
    # print(all_data['side'].value_counts())
    # all_data['side'] = benchmark_rule2(all_data)
    # print(all_data['side'].value_counts())
    # all_data['side'] = benchmark_rule3(all_data)
    # print(all_data['side'].value_counts())



    # print(train_data)
    # print(test_data)

    # strategy = 'Benchmark strategy (long pos short neg)'
    # strategy = 'Benchmark strategy (region specific rules (PM))'
    # strategy = f'Benchmark strategy (region specific rules after fees (PM))'
    # _, test_data = DC.get_benchmark_test_data()
    # test_data['side'] = test_data['headline_senti'].replace({'positive': 'long', 'negative': 'short'})
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

    # train_data, test_data = DC.get_benchmark_test_data()
    # results = pd.concat([train_data, test_data], axis=0)
    # #
    # strategy = 'blind long'
    # # results = DL.loadBT(strategy)
    # results['side'] = 'long'
    # results = get_pnl(results)
    # DL.toBT(results, strategy)
    # # plot_matrix(strategy)
    # # # #
    # strategy = 'blind short'
    # # # results = DL.loadBT(strategy)
    # results['side'] = 'short'
    # results = get_pnl(results)
    # DL.toBT(results, strategy)
    # # plot_matrix(strategy)


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

    # from datetime import datetime
    # train_data = DL.loadDB('Backtest/Benchmark data/train temp.csv', parse_dates=['Date', 'Time'])
    # date_range = pd.date_range('20220101', '20220331')
    # test_data = train_data[train_data['Date'].isin(date_range)]
    # train_data = train_data[train_data['Date'] < datetime(2022, 1, 1)]
    #
    # test_data = benchmark_expectancy(train_data, test_data)
    #
    # DL.toDB(test_data, 'Citi data Jan-Mar.csv')

    # simulation_result = simulation(range(1, 500), from_local=True)
    # equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b = simulation_datecleaning(simulation_result)
    # fig = simulation_visualization(equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b)
    # fig.savefig(f'{DL.database_path}/Backtest/Simulation/Simulation ols.png')
    #
    # region_list = ['Americas', 'Europe', 'Asia', 'South Africa']
    # for region in region_list:
    #     exclusion_list = [x for x in region_list if x != region]
    #     simulation_result = simulation(range(1, 501), from_local=True, exclude=exclusion_list)
    #     equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b = simulation_datecleaning(simulation_result)
    #     fig = simulation_visualization(equity_curve_b, num_trades_b, hit_ratio_b, expectancy_b)
    #     fig.savefig(f'{DL.database_path}/Backtest/Simulation/Simulation {region} (blind).png')

    # results = benchmark_strategy(sort_by='exch_region')
    # pnl_df = Engine.portfolio_management(results)
    # strategy = 'Headline strategy (scoring on side, exch_region, report_type)'
    # DL.toDB(pnl_df, f'Backtest/{strategy}.csv')
    #
    # vis = visual(strategy)
    # vis.visual_job()

    # plot_matrix()
