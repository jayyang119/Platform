import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from uti import DataLoader, Logger
from Broker import get_pnl
from Backtest.settings import (get_expectancy, hit_ratio, benchmark_side,
                               plot_EDA, plot_EDA_country, )

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path
BACKTEST_PATH = os.path.join(DATABASE_PATH, 'Backtest')

class visual:
    def __init__(self, strategy='Headline strategy'):
        self.trades_df = DL.loadDB(f'Backtest/{strategy}.csv', parse_dates=['publish_date_and_time', 'd0_date'])
        self.strategy = strategy
        DL.create_folder(f'{BACKTEST_PATH}/{strategy}')

    def preprocess(self):
        self.trades_df['side'] = self.trades_df[['side']].replace({'positive': 'long', 'negative': 'short'})
        self.trades_df['No. of trades'] = 1
        # self.trades_df = self.trades_df.set_index('publish_date_and_time')
        self.trades_df = self.trades_df.set_index('d0_date')
        self.trades_df = self.trades_df.sort_index(ascending=True)
        print(self.trades_df[['side']].value_counts())

    def plot_cum_r(self, df, region=''):
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.title(f'Cumulative R({self.strategy}) {region}')
        plt.plot(df['d0_r'].cumsum(), label='Hold 1 day')
        plt.plot(df['d1_r'].cumsum(), label='Hold 2 days')
        plt.plot(df['d2_r'].cumsum(), label='Hold 3 days')
        print(f'{region} Sum of d0_r', df['d0_r'].sum())
        print(f'{region} Sum of d1_r', df['d1_r'].sum())
        print(f'{region} Sum of d2_r', df['d2_r'].sum())
        plt.legend(loc='best')
        # plt.show()
        if len(region) == 0:
            fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/cumulative r.png')
        else:
            fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/cumulative r({region}).png')

    def plot_r_dist(self, column='d0_r'):
        fig, axs = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
        # plt.set_('R distribution{strategy}')
        axs.hist(self.trades_df['d0_r'], label='Hold 1 day', bins=100)
        #         # axs[1].hist(df['d1_r'], label=('Hold 2 days'), bins=100)
        #         # axs[2].hist(df['d2_r'], label=('Hold 3 days'), bins=100)
        axs.title.set_text(f'R distribution {self.strategy} - {column}')
        # axs[1].title.set_text('R distribution - Hold 2 days')
        # axs[2].title.set_text('R distribution - Hold 3 days')
        axs.set_xlim([-1.5, max(self.trades_df[column])])
        axs.set_xticks(np.arange(-1, int(max(self.trades_df[column]))))
        plt.tight_layout()
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/{column} distribution.png')

    def plot_hit_ratio(self, column='d0_r'):
        total_hit_ratio = hit_ratio(self.trades_df, column=column)
        print(f'Total hit ratio {column}', total_hit_ratio)

        long_hit_ratio, short_hit_ratio = benchmark_side(self.trades_df, column)
        fig = plot_EDA([long_hit_ratio, total_hit_ratio, short_hit_ratio],
                        labels=['Long hit ratio', 'Total hit ratio', 'Short hit ratio'],
                        title=f'Hit ratio plot (Hold 1 day) - {self.strategy} {column}', precision=4)
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/hit ratio {column}.png')

    def plot_side_expectancy(self, column='d0_r'):
        # ele_group = [sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type']
        # blind_long_expectancy = get_expectancy(blind_long, group_by=ele_group)
        # blind_short_expectancy = get_expectancy(blind_short, group_by=ele_group)

        expectancy_by_side = get_expectancy(self.trades_df, group_by=[], column=column)
        print('Total expectancy:', expectancy_by_side['Expectancy'])
        fig = plot_EDA(expectancy_by_side['Expectancy'], labels=['Total'],
                       precision=4, title=f'Total Expectancy - {self.strategy} {column}')
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/Total expectancy {column}.png')

        expectancy_by_side = get_expectancy(self.trades_df, group_by=['side'], column=column)
        print(expectancy_by_side['Expectancy'])
        fig = plot_EDA(expectancy_by_side.reset_index()['Expectancy'], labels=['Long', 'Short'],
                       precision=4, title=f'Expectancy by sides - {self.strategy} {column}')
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/expectancy by side {column}.png')

    def plot_market_r(self, column='d0_r'):
        R_by_market_side = self.trades_df[['No. of trades', column, 'exch_location', 'side']].groupby(['exch_location', 'side']).agg('sum')
        fig = plot_EDA_country(R_by_market_side.reset_index(), hue='side',
                               title=f'No. of trades by Country/side - {self.strategy} {column}', rotate=True)
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/No of trades by country and side.png')

        fig = plot_EDA_country(R_by_market_side.reset_index(), y=column, hue='side',
                               title=f'R by Country/side - {self.strategy} {column}', rotate=True)
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/R by market and side.png')

    def plot_market_expectancy(self, column='d0_r'):
        # ele_group = [sort_by, 'Headline sentiment', 'Summary sentiment', 'Report Type']
        # blind_long_expectancy = get_expectancy(blind_long, group_by=ele_group)
        # blind_short_expectancy = get_expectancy(blind_short, group_by=ele_group)

        expectancy_by_market = get_expectancy(self.trades_df, group_by=['exch_location'], column=column)
        fig = plot_EDA_country(expectancy_by_market.reset_index(), x='exch_location', y='Expectancy',
                               title=f'Expectancy by market - {self.strategy} {column}', precision=2, rotate=True)
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/expectancy by market.png')

        expectancy_by_market_side = get_expectancy(self.trades_df, group_by=['exch_location', 'side'], column=column)
        fig = plot_EDA_country(expectancy_by_market_side.reset_index(), x='exch_location', y='Expectancy',
                               hue='side', title=f'Expectancy by market - {self.strategy} {column}', precision=2, rotate=True)
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/expectancy by market and side.png')

    def plot_region_r(self):
        for region, df in self.trades_df.groupby('exch_region'):
            self.plot_cum_r(df, region)
        # column = 'd0_r'

    # def plot_pm_region_r(self):
    #     for region, df in self.trades_df.groupby('pm_region'):
    #         self.plot_cum_r(df, region)

    def plot_region_expectancy(self, column='d0_r'):
        expectancy_by_region = get_expectancy(self.trades_df, group_by=['exch_region'])
        fig = plot_EDA_country(expectancy_by_region.reset_index(), x='exch_region', y='Expectancy',
                               title=f'Expectancy by region - {self.strategy} {column}', precision=2, rotate=True)
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/expectancy by region.png')

        expectancy_by_region_side = get_expectancy(self.trades_df, group_by=['exch_region', 'side'])
        fig = plot_EDA_country(expectancy_by_region_side.reset_index(), x='exch_region', y='Expectancy',
                               hue='side', title=f'Expectancy by region - {self.strategy} {column}', precision=2, rotate=True)
        fig.savefig(f'{BACKTEST_PATH}/{self.strategy}/expectancy by region and side.png')

    def visual_job(self):
        self.preprocess()
        self.plot_cum_r(self.trades_df)
        self.plot_r_dist()
        self.plot_region_r()
        # self.plot_pm_region_r()
        print()
        self.plot_hit_ratio(column='d0_r')
        print()
        self.plot_side_expectancy(column='d0_r')
        print()
        self.plot_hit_ratio(column='d1_r')
        print()
        self.plot_side_expectancy(column='d1_r')
        print()
        self.plot_market_r()
        self.plot_market_expectancy()
        self.plot_region_expectancy()
        plt.close('all')


def get_matrix_metric(_df, by_i):
    count = len(_df)
    hit = hit_ratio(_df)
    expectancy = get_expectancy(_df, column='d0_r')['Expectancy']
    total_dict = {0: hit, 1: expectancy, 2: count}

    total = get_expectancy(_df, group_by=['exch_location', 'headline_senti', 'summary_senti'])[['Hit ratio', 'Expectancy', 'Count']].unstack()

    total_head = get_expectancy(_df, group_by=['exch_location', 'headline_senti'])[['Hit ratio', 'Expectancy', 'Count']].unstack()

    total_sum = get_expectancy(_df, group_by=['exch_location', 'summary_senti'])[['Hit ratio', 'Expectancy', 'Count']].unstack()

    outputs = []
    for j, metric_name in {0: 'Hit ratio', 1: 'Expectancy', 2: 'Count'}.items():
        multiple_width = len(total[metric_name].columns)
        multiple_height = len(total.loc[by_i])

        metric = total[total.columns[j * multiple_width:(j + 1) * multiple_width]].copy(deep=True)
        try:
            metric[(metric_name, 'All')] = total_head[total_head.columns[j * multiple_height:(j + 1) * multiple_height]].values[0]
        except Exception as e:
            logger.error(e)
            logger.info(metric)
            logger.info(total_head)
        try:
            metric.loc[(by_i, 'All'), :] = np.append(total_sum[total_sum.columns[j * multiple_width:(j + 1) * multiple_width]].values, [total_dict[j]])
        except Exception as e:
            logger.error(e)
            logger.info(metric)
            logger.info(total_sum)
        if len(metric) != 4:
            for senti in ['negative', 'neutral', 'positive']:
                if (by_i, senti) not in metric.index:
                    print((by_i, senti))
                    metric.loc[(by_i, senti), :] = 0.0
            try:
                metric = metric.loc[[(by_i, senti) for senti in ['negative', 'neutral', 'positive', 'All']]]
            except Exception as e:
                logger.error(e)
                logger.info(metric)

        if len(metric.columns) != 4:
            for senti in ['negative', 'neutral', 'positive']:
                if (metric_name, senti) not in metric.columns:
                    print((metric_name, senti))
                    metric[(metric_name, senti)] = 0.0
            try:
                metric = metric[[(metric_name, senti) for senti in ['negative', 'neutral', 'positive', 'All']]]
            except Exception as e:
                logger.error(e)
                logger.info(metric)
        outputs.append(metric)

    def drop_multi_index(df):
        df = df.reset_index(level=[0], drop=True).T.reset_index(level=[0], drop=True).T
        return df

    outputs = [drop_multi_index(_x) for _x in outputs]

    return outputs


def plot_matrix(strategy, by='exch_location'):
    trade_df = DL.loadBT(strategy)
    # trade_df['side'].replace({'positive': 'long', 'negative': 'short'}, inplace=True)
    trade_df['No. of trades'] = 1
    # writer = pd.ExcelWriter(os.path.join(DATABASE_PATH, f'Backtest/Matrix({strategy}).xlsx'), engine='xlsxwriter')
    writer = pd.ExcelWriter(os.path.join(DATABASE_PATH, f'Backtest/Matrix({strategy}).xlsx'), engine='openpyxl')

    for side in trade_df['side'].unique():
        df = trade_df[trade_df['side'] == side].reset_index(drop=True)

        # by = 'exch_region'  # or 'exch_location'
        from Crawler import REPORT_TYPE_MAPPING
        rt_list = list(REPORT_TYPE_MAPPING.values())
        # rt_list = ['Catalyst Watch', 'Earnings Review', 'Rating Change', 'Target Price Change', 'Estimate Change', 'M&A', 'ad-hoc']
        for by_i in df[by].unique():
            logger.info(by_i)
            df_i = df[df[by] == by_i].copy(deep=True)

            (k_hit, k_expectancy, k_count) = get_matrix_metric(df_i, by_i)
            logger.info(k_hit)
            logger.info(k_expectancy)
            logger.info(k_count)

            # if side == 'long':
            k_count.to_excel(writer, sheet_name=f'{by_i} {side}', startcol=3, startrow=5)
            k_expectancy.to_excel(writer, sheet_name=f'{by_i} {side}', startcol=3, startrow=13)
            k_hit.to_excel(writer, sheet_name=f'{by_i} {side}', startcol=3, startrow=21)


            for _count in range(len(rt_list)):
                report_type = rt_list[_count]
                df_rt = df_i[df_i['report_type'] == report_type]
                if len(df_rt) == 0:
                    continue
                (k_hit, k_expectancy, k_count) = get_matrix_metric(df_rt, by_i)
                # rt_name = ''.join([x[0].upper() for x in report_type.split(' ')])
                # if side == 'long':
                k_count.to_excel(writer, sheet_name=f'{by_i} {side}', startcol=3, startrow=5+(_count+1)*24)
                k_expectancy.to_excel(writer, sheet_name=f'{by_i} {side}', startcol=3, startrow=13+(_count+1)*24)
                k_hit.to_excel(writer, sheet_name=f'{by_i} {side}', startcol=3, startrow=21+(_count+1)*24)

    writer.save()


if __name__ == '__main__':
    # pnl_df = DL.loadDB('Backtest/Headline strategy (total).csv')
    # pnl_df = pnl_df[pnl_df['side'].isin(['long', 'short'])]
    # print(pnl_df['side'].describe())


    # for report_type in pnl_df['Report Type'].unique():
    #     for side in ['long', 'short']:
    column = 'd0_r'
    # for group, df in pnl_df.groupby(['Report Type', 'side']):
    #     output = get_expectancy(df, column, inputs=['No. of trades', column, 'exch_region'], group_by=['exch_region'])
    #     print(group, output)
    #
    #     DL.toDB(output, f'Backtest/Summary {group[0]} {group[1]}.csv')

    plot_matrix('blind long')
    plot_matrix('blind short')
    # trade_df = DL.loadDB('price_df.csv')
    #
    # strategy = 'benchmark'
    # strategy = 'scoring (PM)'
    # vis = visual(strategy)
    # vis.visual_job()
    # for strategy in ['short', 'long']:
    #     trade_df['side'] = strategy
    #
    #     writer = pd.ExcelWriter(os.path.join(DATABASE_PATH, f'Backtest/{strategy} strategy R.xlsx'),
    #                             engine='xlsxwriter')
    #     for region in pnl_df['exch_region'].unique():
    #         df = pnl_df[pnl_df['exch_region'] == region]
    #         k = get_expectancy(df, column,
    #                                  inputs=['No. of trades', column, 'Report Type', 'Headline sentiment',
    #                                          'Summary sentiment'],
    #                                  group_by=['Report Type', 'Headline sentiment', 'Summary sentiment'],
    #                                  )
    #         k = k[['Hit ratio', 'Expectancy', 'Count']]
    #         k.unstack().to_excel(writer, sheet_name=f'{region}')
    #     writer.save()
