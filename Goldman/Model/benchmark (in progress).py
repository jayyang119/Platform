# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:46:38 2022

@author: JayYang
"""
import pandas as pd
import numpy as np
import os
from random import sample

from Broker import get_pnl
from uti import DataLoader, Logger
from Backtest import visual
from datetime import datetime
from Model.settings import DataCleaner
from Backtest.settings import (get_expectancy, hit_ratio, benchmark_side,
                               plot_EDA, plot_EDA_country, )
from library import Dataset
from Backtest import backtest_engine
from Backtest.simulation_functions import simulation_datecleaning, simulation_visualization

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path
DC = DataCleaner()
Engine = backtest_engine()


def benchmark_rules(by_x='exch_location', train_data=None, test_data=None):
    column = 'd0_r'
    if train_data is None and test_data is None:
        train_data, test_data = DC.get_benchmark_test_data()
    # train_data['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)
    # test_data['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)
    if 'side' not in train_data.columns:
        train_data['side'] = ''
    if 'side' not in test_data.columns:
        test_data['side'] = ''
    # test_data = test_data[test_data['release_period']!='Within'].reset_index(drop=True)

    # Rules
    # Asia
    def asia_df(df):
        asia = df[df['exch_region'] == 'Asia'].reset_index(drop=True)  # A copy
        # Long trades
        jp_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') & \
                   (asia['Report Type'] == 'Target Price Increase') & (asia['exch_location'] == 'Japan')
        jp_long2 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') & \
                   (asia['Report Type'] == 'ad-hoc') & (asia['exch_location'] == 'Japan')

        hk_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') &\
                   (asia['Report Type'].isin(['Earning\'s Review', 'Rating Change', 'Target Price Increase', 'ad-hoc'])) &\
                   (asia['exch_location'] == 'Hong Kong')

        au_long0 = ((asia['Headline sentiment'] == 'positive') | (asia['Summary sentiment'] == 'positive')) &\
                   (asia['Summary sentiment'] != 'negative') &\
                   (asia['Report Type'].isin(['Earning\'s Review'])) & (asia['exch_location'] == 'Australia')
        au_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') &\
                   (asia['Report Type'].isin(['Rating Change'])) & (asia['exch_location'] == 'Australia')
        au_long2 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
                   (asia['Report Type'].isin(['ad-hoc'])) & (asia['exch_location'] == 'Australia')

        cn_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
                   (asia['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Rating Change'])) & (asia['exch_location'] == 'China')
        cn_long2 = ((asia['Headline sentiment'] == 'positive') | (asia['Summary sentiment'] == 'positive')) &\
                   (asia['Report Type'].isin(['Target Price Increase'])) & (asia['exch_location'] == 'China')

        sk_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
                   (asia['Report Type'].isin(['Rating Change'])) & (asia['exch_location'] == 'South Korea')

        tw_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') &\
                   (asia['Report Type'].isin(['Earning\'s Review', 'Rating Change'])) & (asia['exch_location'] == 'Taiwan')

        # Short trades
        jp_short1 = (asia['Headline sentiment'] == 'negative') & (asia['Report Type'] == 'ad-hoc') & (asia['exch_location'] == 'Japan')

        hk_short1 = (asia['Headline sentiment'] == 'negative') &\
                    (asia['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Rating Change', 'Target Price Decrease'])) & \
                    (asia['exch_location'] == 'Hong Kong')
        hk_short2 = (asia['Summary sentiment'] == 'negative') & \
                    (asia['Report Type'].isin(['Estimate Change', 'Rating Change', 'ad-hoc'])) & \
                    (asia['exch_location'] == 'Hong Kong')

        au_short1 = (asia['Headline sentiment'] == 'negative') & (asia['Report Type'].isin(['ad-hoc'])) &\
                    (asia['exch_location'] == 'Australia')
        au_short2 = ((asia['Headline sentiment'] == 'negative') | (asia['Summary sentiment'] == 'negative')) &\
                    (asia['Report Type'].isin(['Estimate Review', 'Rating Change'])) & (asia['exch_location'] == 'Australia')
        au_short3 = (asia['Report Type'].isin(['Target Price Decrease'])) & (asia['exch_location'] == 'Australia')
        au_short4 = (asia['Summary sentiment'] == 'negative') & (asia['Report Type'].isin(['Earning\'s Review'])) & \
                    (asia['exch_location'] == 'Australia')

        cn_short1 = (asia['Report Type'].isin(['Target Price Decrease'])) & (asia['exch_location'] == 'China')

        tw_short1 = (asia['Headline sentiment'] == 'negative') &\
                    (asia['Report Type'].isin(['Rating Change', 'Earning\'s Review'])) & \
                    (asia['exch_location'] == 'Taiwan')

        sk_short1 = (asia['Headline sentiment'] == 'negative') &\
                    (asia['Report Type'].isin(['Rating Change', 'Estimate Change', 'ad-hoc', 'Earning\'s Review', 'Target Price Decrease'])) &\
                    (asia['exch_location'] == 'South Korea')

        tk_short1 = (asia['Headline sentiment'] == 'negative') &\
                    (asia['Report Type'].isin(['ad-hoc', 'Earning\'s Review'])) & (asia['exch_location'] == 'Turkey')

        asia_long_trades_index = jp_long1 | jp_long2 | hk_long1 | au_long0 | au_long1 | au_long2 |\
                                 cn_long1 | cn_long2 | sk_long1 | tw_long1
        asia_short_trades_index = jp_short1 | hk_short1 | hk_short2 | au_short1 | au_short2 | au_short3 | au_short4 | cn_short1 |\
                                  sk_short1 | tk_short1 | tw_short1
        # asia_trades_index = asia_long_trades_index + asia_short_trades_index
        asia.loc[asia_long_trades_index, 'side'] = 'long'
        asia.loc[asia_short_trades_index, 'side'] = 'short'

        asia = asia.loc[asia_long_trades_index | asia_short_trades_index].reset_index(drop=True)
        # asia.loc[~(asia_long_trades_index | asia_short_trades_index), 'side'] = \
        #     asia.loc[~(asia_long_trades_index | asia_short_trades_index), 'Headline sentiment'].replace({'positive': 'long', 'negative': 'short'})

        asia = get_pnl(asia)
        return asia

    # Europe
    def eu_df(df):
        europe = df[df['exch_region'] == 'Europe'].reset_index(drop=True)
        europe = europe[~europe['exch_location'].isin(['Finland', 'Denmark', 'Greece', 'Austria'])].reset_index(
            drop=True)

        fc_long1 = (europe['Headline sentiment'] == 'positive') & \
                   (europe['Report Type'].isin(['Estimate Change', 'Target Price Increase'])) &\
                   (europe['exch_location'] == 'France')

        gm_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Report Type'].isin(['ad-hoc'])) &\
                   (europe['exch_location'] == 'Germany')
        gm_long2 = (europe['Report Type'].isin(['Target Price Increase'])) &\
                   (europe['exch_location'] == 'Germany')
        gm_long3 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') &\
                   (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change'])) & (europe['exch_location'] == 'Germany')

        uk_long1 = (europe['Headline sentiment'] == 'positive') & \
                   (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change'])) & (europe['exch_location'] == 'United Kingdom')
        uk_long2 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') &\
                   (europe['Report Type'].isin(['Rating Change', 'Target Price Increase', 'ad-hoc'])) &\
                   (europe['exch_location'] == 'United Kingdom')

        pg_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') &\
                   (europe['Report Type'].isin(['Target Price Increase', 'ad-hoc'])) & (europe['exch_location'] == 'Portugal')

        it_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') &\
                   (europe['Report Type'].isin(['ad-hoc', 'Rating Change'])) &\
                   (europe['exch_location'] == 'Italy')
        it_long2 = (europe['Headline sentiment'] == 'positive') & (europe['Report Type'].isin(['Earning\'s Review'])) &\
                   (europe['exch_location'] == 'Italy')

        sw_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') &\
                   (europe['Report Type'].isin(['Estimate Change', 'Rating Change'])) & (europe['exch_location'] == 'Switzerland')

        dm_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') &\
                   (europe['Report Type'].isin(['Earning\'s Review', 'Rating Change'])) & (europe['exch_location'] == 'Denmark')

        as_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
                   (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'ad-hoc'])) & (europe['exch_location'] == 'Austria')

        fl_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') &\
                   (europe['Report Type'].isin(['Earning\'s Review', 'Target Price Increase'])) & (europe['exch_location'] == 'Finland')

        europe_long_trades_index = fc_long1 | gm_long1 | gm_long2 | gm_long3 | uk_long1 | uk_long2 | \
                                   pg_long1 | sw_long1 | dm_long1 | as_long1 | fl_long1 | it_long1 | it_long2

        fc_short1 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) &\
                    (europe['Report Type'].isin(['Earning\'s Review', 'ad-hoc'])) & (europe['exch_location'] == 'France')
        fc_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Report Type'].isin(['Rating Change'])) & (europe['exch_location'] == 'France')
        fc_short3 = (europe['Report Type'].isin(['Target Price Decrease'])) & (europe['exch_location'] == 'France')

        gm_short1 = (europe['Summary sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change'])) & (europe['exch_location'] == 'Germany')
        gm_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Germany')
        gm_short3 = (europe['Report Type'].isin( ['Target Price Decrease'])) & (europe['exch_location'] == 'Germany')


        bg_short1 = (europe['Headline sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Rating Change', 'Target Price Decrease'])) &\
                    (europe['exch_location'] == 'Belgium')
        bg_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Belgium')

        uk_short1 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] != 'positive') &\
                    (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'United Kingdom')
        uk_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'United Kingdom')

        sp_short1 = (europe['Headline sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['ad-hoc', 'Rating Change'])) &\
                    (europe['exch_location'] == 'Spain')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'
        sp_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'Spain')
        sp_short3 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) & \
                    (europe['Report Type'].isin(['Estimate Change'])) & (europe['exch_location'] == 'Spain')

        it_short1 = (europe['Headline sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['ad-hoc', 'Rating Change'])) &\
                    (europe['exch_location'] == 'Italy')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'

        sw_short1 = (europe['Headline sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Target Price Decrease'])) &\
                    (europe['exch_location'] == 'Switzerland')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'
        sw_short2 = (europe['Summary sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Switzerland')


        sd_short1 = (europe['Summary sentiment'] == 'negative') &\
                    (europe['Report Type'].isin(['Rating Change'])) &\
                    (europe['exch_location'] == 'Sweden')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'
        sd_short2 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) & \
                    (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'Sweden')

        fl_short1 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                    (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Rating Change', 'Target Price Decrease'])) &\
                    (europe['exch_location'] == 'Finland')

        nw_short1 = (europe['Summary sentiment'] == 'negative') & \
                    (europe['Report Type'].isin(['Earning\'s Review'])) & \
                    (europe['exch_location'] == 'Norway')

        europe_short_trades_index = fc_short1 | fc_short2 | fc_short3 | gm_short1 | gm_short2 | gm_short3 | bg_short1 | \
                                    bg_short2 | uk_short1 | uk_short2 | sp_short1 | sp_short2 | sp_short3 | it_short1 | \
                                    sw_short1 | sw_short2 | sd_short1 | sd_short2 | fl_short1 | nw_short1

        europe.loc[europe_long_trades_index, 'side'] = 'long'
        europe.loc[europe_short_trades_index, 'side'] = 'short'
        # europe.loc[~(europe_long_trades_index | europe_short_trades_index), 'side'] = \
        #     europe.loc[~(europe_long_trades_index | europe_short_trades_index), 'Headline sentiment'].replace(
        #         {'positive': 'long', 'negative': 'short'})
        europe = europe.loc[europe_long_trades_index | europe_short_trades_index].reset_index(drop=True)
        europe = get_pnl(europe)
        return europe

    # US
    def am_df(df):
        am = df[df['exch_region'] == 'Americas'].reset_index(drop=True)

        us_long1 = list(np.where(#(am['Headline sentiment'] == 'positive') &
                                 #(am['Summary sentiment'] == 'positive') &
                                 (am['Report Type'].isin(['Target Price Increase'])) &
                                 (am['exch_location'] == 'United States'))[0])
        us_long2 = list(np.where((am['Headline sentiment'] == 'positive') &
                                 (am['Report Type'].isin(['ad-hoc'])) &
                                 (am['exch_location'] == 'United States'))[0])

        ca_long1 = list(np.where((am['Headline sentiment'] == 'positive') &
                                  (am['Report Type'].isin(['ad-hoc'])) &
                                  (am['exch_location'] == 'Canada'))[0])

        # Short trades
        us_short1 = list(np.where((am['Headline sentiment'] == 'negative') &
                                  (am['Report Type'].isin(['Earning\'s Review', 'Target Price Decrease', 'ad-hoc'])) &
                                  (am['exch_location'] == 'United States'))[0])
        us_short2 = list(np.where((am['Headline sentiment'] == 'negative') &
                                  (am['Summary sentiment'] != 'positive') &
                                  (am['Report Type'].isin(['Rating Change'])) &
                                  (am['exch_location'] == 'United States'))[0])

        ca_short1 = list(np.where((am['Headline sentiment'] == 'negative') &
                                  (am['Report Type'].isin(['Earning\'s Review'])) &
                                  (am['exch_location'] == 'Canada'))[0])

        am_long_trades_index = us_long1 + us_long2 + ca_long1
        am_short_trades_index = us_short1 + us_short2 + ca_short1
        am_trades_index = am_long_trades_index + am_short_trades_index
        am_trades = am.iloc[am_trades_index].copy(deep=True)

        am_trades.loc[am_long_trades_index, 'side'] = 'long'
        am_trades.loc[am_short_trades_index, 'side'] = 'short'
        am_trades = am_trades.drop_duplicates().reset_index(drop=True)

        am_trades = get_pnl(am_trades)
        return am_trades

    def get_all_trades_expectancy():
        asia_train_data = asia_df(train_data)
        am_train_data = am_df(train_data)
        euro_train_data = eu_df(train_data)

        asia_test_data = asia_df(test_data)
        am_test_data = am_df(test_data)
        euro_test_data = eu_df(test_data)

        all_trades_train_data = pd.concat([asia_train_data, am_train_data, euro_train_data], axis=0).reset_index(drop=True)
        all_trades_test_data = pd.concat([asia_test_data, am_test_data, euro_test_data], axis=0).reset_index(drop=True)

        all_trades_train_data = get_pnl(all_trades_train_data)


        # Expectancy 1: by market and report type
        expectancy_by_market_and_report_type = get_expectancy(all_trades_train_data, column,
                                                              inputs=['No. of trades', column, 'exch_location', 'side', 'Report Type'],
                                                              group_by=['exch_location', 'side', 'Report Type'])

        for by_i in all_trades_test_data['exch_location'].unique():
            for side in ['long', 'short']:
                for report_type in all_trades_test_data['Report Type'].unique():
                    if (by_i, side, report_type) not in expectancy_by_market_and_report_type.index:
                        expectancy_by_market_and_report_type.loc[(by_i, side, report_type), :] = 0.0

        all_trades_test_data['Expectancy1'] = [expectancy_by_market_and_report_type.loc[(x['exch_location'], x['side'], x['Report Type'])]['Expectancy']
                                               for _, x in all_trades_test_data[['exch_location', 'side', 'Report Type']].iterrows()]

        # Expectancy 2: by ticker
        expectancy_by_ticker = get_expectancy(all_trades_train_data, column,
                                              inputs=['No. of trades', column, 'Ticker', 'side'],
                                              group_by=['Ticker', 'side'])
        for by_i in all_trades_test_data['Ticker'].unique():
            for side in ['long', 'short']:
                if (by_i, side) not in expectancy_by_ticker.index:
                    expectancy_by_ticker.loc[(by_i, side), :] = 0.0

        all_trades_test_data['Expectancy2'] = [expectancy_by_ticker.loc[(x['Ticker'], x['side'])]['Expectancy']
                                               for _, x in all_trades_test_data[['Ticker', 'side']].iterrows()]

        # Expectancy 3: by Analyst
        expectancy_by_analyst = get_expectancy(all_trades_train_data, column,
                                               inputs=['No. of trades', column, 'Head analyst', 'side'],
                                               group_by=['Head analyst', 'side'])
        for by_i in all_trades_test_data['Head analyst'].unique():
            for side in ['long', 'short']:
                if (by_i, side) not in expectancy_by_analyst.index:
                    expectancy_by_analyst.loc[(by_i, side), :] = 0.0

        all_trades_test_data['Expectancy3'] = [expectancy_by_analyst.loc[(x['Head analyst'], x['side'])]['Expectancy']
                                               for _, x in all_trades_test_data[['Head analyst', 'side']].iterrows()]

        # Expectancy 4: by Region, release_period
        # expectancy_by_region_and_release_period = benchmark_expectancy(all_trades_train_data, column,
        #                                                                inputs=['No. of trades', column, 'exch_region', 'side', 'release_period'],
        #                                                                group_by=['exch_region', 'side', 'release_period'])
        #
        # for by_i in all_trades_test_data['exch_region'].unique():
        #     for side in ['long', 'short']:
        #         for release_period in all_trades_test_data['release_period'].unique():
        #             if (by_i, side, release_period) not in expectancy_by_region_and_release_period.index:
        #                 expectancy_by_region_and_release_period.loc[(by_i, side, release_period), :] = 0.0
        #
        # all_trades_test_data['Expectancy4'] = [
        #     expectancy_by_region_and_release_period.loc[(x['exch_region'], x['side'], x['release_period'])]['Expectancy']
        #     for _, x in all_trades_test_data[['exch_region', 'side', 'release_period']].iterrows()]

        # Intercept: [-0.00835556]
        # Coefficients: [[0.55768331 - 0.03300604  0.06690937]]

        all_trades_test_data['Expectancy'] = - 0.00835556 \
                                             + 0.55768331 * all_trades_test_data['Expectancy1'] \
                                             - 0.03300604 * all_trades_test_data['Expectancy2'] \
                                             + 0.06690937 * all_trades_test_data['Expectancy3']
        return all_trades_test_data

    test_data = get_all_trades_expectancy()
    logger.info(test_data)
    return test_data


def simulation(range_of_test=range(10), from_local=False, exclude=[]):
    train_data, test_data = DC.get_benchmark_test_data()
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

            output = benchmark_rules(train_data=training_data, test_data=testing_data)
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


if __name__ == '__main__':

    column = 'd0_r'


    def plot_matrix(strategy='Headline strategy (scoring on side, market, report type)'):
        trade_df = DL.loadBT(strategy)
        trade_df['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)
        writer = pd.ExcelWriter(os.path.join(DATABASE_PATH, f'Backtest/Matrix({strategy}).xlsx'),
                                engine='xlsxwriter')

        for side in trade_df['side'].unique():
            if side == 'neutral':
                continue
            df = trade_df[trade_df['side'] == side].reset_index(drop=True)
            df = get_pnl(df)
            # df = DataCleaner.preprocess_trade_df(df)
            # DL.toDB(pnl_df, f'Backtest/Matrix({strategy}).csv')

            # by = 'exch_region'  # or 'exch_location'
            by = 'exch_location'
            for by_i in df[by].unique():
                logger.info(by_i)
                df_i = df[df[by] == by_i]
                k = get_expectancy(df_i, column,
                                   inputs=['No. of trades', column, 'Report Type', 'Headline sentiment',
                                                 'Summary sentiment'],
                                   group_by=['Report Type', 'Headline sentiment', 'Summary sentiment'],
                                   )
                k = k[['Hit ratio', 'Expectancy', 'Count']]
                k.unstack().to_excel(writer, sheet_name=f'{by_i} {side}')
        writer.save()


    def benchmark_strategy(sort_by='exch_region'):
        train_data, test_data = DC.get_benchmark_test_data()
        # train_data['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)
        # test_data['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)
        # Scoring system based on R0 market, side, report type
        expectancy_sort_by = get_expectancy(train_data, column,
                                            inputs=['No. of trades', column, sort_by, 'side', 'Report Type'],
                                            group_by=[sort_by, 'side', 'Report Type'])

        for by_i in test_data[sort_by].unique():
            for side in ['long', 'short']:
                for report_type in test_data['Report Type'].unique():
                    if (by_i, side, report_type) not in expectancy_sort_by.index:
                        # ind = (market, side, report_type)
                        expectancy_sort_by.loc[(by_i, side, report_type)] = [0] * len(expectancy_sort_by.columns)

        expectancy_mapping = [expectancy_sort_by.loc[(x[sort_by], x['side'], x['Report Type'])]['Expectancy']
                              for _, x in test_data[[sort_by, 'side', 'Report Type']].iterrows()]
        test_data.loc[:, 'Expectancy'] = expectancy_mapping
        return test_data


    def region_case_study(region='Asia', side='positive'):
        train_data, test_data = DC.get_benchmark_test_data()
        train_data = train_data[train_data['exch_region'] == region].reset_index(drop=True)
        test_data = test_data[test_data['exch_region'] == region].reset_index(drop=True)


        train_data['side'] = side
        # train_data = get_pnl(train_data)
        pnl_df = get_pnl(train_data)

        expectancy_sort_by = get_expectancy(train_data, column,
                                            inputs=['No. of trades', column, 'Report Type', 'Headline sentiment', 'Summary sentiment', 'exch_location'],
                                            group_by=['Report Type', 'Headline sentiment', 'Summary sentiment', 'exch_location'])

        for hs in ['positive', 'negative', 'neutral']:
            for ss in ['positive', 'negative', 'neutral']:
                for market in test_data['exch_location'].unique():
                    for report_type in test_data['Report Type'].unique():
                        ind = (report_type, hs, ss, market)
                        if ind not in expectancy_sort_by.index:
                            expectancy_sort_by.loc[ind, :] = 0.00

        expectancy_mapping = [expectancy_sort_by.loc[(x['Report Type'], x['Headline sentiment'], x['Summary sentiment'], x['exch_location'])]['Expectancy']
                              for _, x in test_data[['Report Type', 'Headline sentiment', 'Summary sentiment', 'exch_location']].iterrows()]
        test_data.loc[:, 'Expectancy'] = expectancy_mapping
        #
        # test_data['side'] = pd.cut(test_data['Expectancy'],
        #                            bins=[-np.inf, -0.000001, 0.000001, np.inf],
        #                            labels=['negative', 'neutral', 'positive'])
        # test_data['Expectancy'] = abs(test_data['Expectancy'])
        #
        # pnl_df = get_pnl(test_data)
        DL.toBT(pnl_df, f'{region}_{side}_pnl')
        plot_matrix(f'{region}_{side}_pnl')



    # region = 'Americas'
    # strategy = f'{region}_pnl(PM)'
    # region_case_study(region, 'positive')
    # region_case_study(region, 'negative')
    # vis = visual(strategy)
    # vis.visual_job()
    # plot_matrix('Americas_positive_pnl')
    # plot_matrix('Americas_negative_pnl')
    # plot_matrix('Americas_pnl(PM)')

    # region = 'Asia'
    # region_case_study(region, 'positive')
    # region_case_study(region, 'negative')

    strategy = 'Benchmark strategy (scoring system)'
    results = benchmark_rules()
    DL.toBT(results, strategy)
    # vis = visual(strategy)
    # vis.visual_job()
    # plot_matrix(strategy)

    strategy = f'Benchmark strategy (scoring system after fees)'
    DS = Dataset(results)
    DS.backtest()
    results_after_fee = DS.df.copy(deep=True)
    results_after_fee['d0_r'] = results_after_fee['d0_r'] - results_after_fee['fee_in_r']
    results_after_fee['d1_r'] = results_after_fee['d1_r'] - results_after_fee['fee_in_r']
    results_after_fee['d2_r'] = results_after_fee['d2_r'] - results_after_fee['fee_in_r']
    DL.toBT(results_after_fee, strategy)
    vis = visual(strategy)
    vis.visual_job()
    # plot_matrix(strategy)


    # strategy = 'Benchmark strategy (scoring system (PM))'
    # results = DL.loadBT('Benchmark strategy (scoring system)')
    # pnl_df = Engine.portfolio_management(results)
    # DL.toBT(pnl_df, strategy)
    # plot_matrix(strategy)
    # vis = visual(strategy)
    # vis.visual_job()
    #
    # strategy = f'Benchmark strategy (scoring system after fees (PM))'
    # DS = Dataset(pnl_df)
    # DS.backtest()
    # pnl_df_after_fee = DS.df.copy(deep=True)
    # pnl_df_after_fee['d0_r'] = pnl_df_after_fee['d0_r'] - pnl_df_after_fee['fee_in_r']
    # pnl_df_after_fee['d1_r'] = pnl_df_after_fee['d1_r'] - pnl_df_after_fee['fee_in_r']
    # pnl_df_after_fee['d2_r'] = pnl_df_after_fee['d2_r'] - pnl_df_after_fee['fee_in_r']
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
    #
    # vis = visual(strategy)
    # vis.visual_job()

    # plot_matrix()