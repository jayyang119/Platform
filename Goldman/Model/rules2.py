import numpy as np
import pandas as pd
from Broker import get_pnl
from Model import DataCleaner
from uti import DataLoader
from Backtest import plot_matrix, get_expectancy

DC = DataCleaner()
DL = DataLoader()

def asia_df(df):
    asia = df[df['exch_region'] == 'Asia'].reset_index(drop=True)  # A copy
    # Long trades
    jp_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') & \
               (asia['Report Type'].isin(['Earning\'s Review', 'Target Price Increase'])) & (asia['exch_location'] == 'Japan')
    jp_long2 = (asia['Headline sentiment'] == 'positive') & \
               (asia['Report Type'] == 'ad-hoc') & (asia['exch_location'] == 'Japan')

    hk_long1 = (asia['Headline sentiment'] == 'positive') | (asia['Summary sentiment'] == 'positive') & \
               (asia['Headline sentiment'] != 'negative') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Earning\'s Review', 'ad-hoc'])) & \
               (asia['exch_location'] == 'Hong Kong')   # 'Rating Change',
    hk_long2 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Rating Change'])) & (asia['exch_location'] == 'Hong Kong')

    au_long0 = ((asia['Headline sentiment'] == 'positive') | (asia['Summary sentiment'] == 'positive')) & \
               (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Earning\'s Review'])) & (asia['exch_location'] == 'Australia')
    au_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Rating Change'])) & (asia['exch_location'] == 'Australia')
    au_long2 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['ad-hoc'])) & (asia['exch_location'] == 'Australia')

    cn_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Rating Change'])) & (
                           asia['exch_location'] == 'China')
    cn_long2 = (asia['Headline sentiment'] == 'positive') | (asia['Summary sentiment'] == 'positive') & \
               (asia['Report Type'].isin(['Target Price Increase'])) & (asia['exch_location'] == 'China')

    sk_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Rating Change'])) & (asia['exch_location'] == 'South Korea')

    tw_long1 = (asia['Headline sentiment'] == 'positive') | (asia['Summary sentiment'] == 'positive') & \
               (asia['Headline sentiment'] != 'negative') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Earning\'s Review', 'Rating Change'])) & (asia['exch_location'] == 'Taiwan')
    tw_long2 = (asia['Report Type'].isin(['Target Price Increase'])) & (asia['exch_location'] == 'Taiwan')

    asia_long_trades_index = jp_long1 | jp_long2 | hk_long1 | hk_long2 | au_long0 | au_long1 | au_long2 | \
                             cn_long1 | cn_long2 | sk_long1 | tw_long1 | tw_long2

    # Short trades
    jp_short1 = (asia['Headline sentiment'] == 'negative') & (asia['Report Type'] == 'ad-hoc') & (asia['exch_location'] == 'Japan')
    jp_short2 = (asia['Headline sentiment'] == 'negative') & (asia['Summary sentiment'] == 'negative') & \
                (asia['Report Type'] == 'Earning\'s Review') & (asia['exch_location'] == 'Japan')

    hk_short1 = (asia['Headline sentiment'] == 'negative') | (asia['Summary sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['Earning\'s Review', 'ad-hoc', 'Estimate Change', 'Rating Change', 'Target Price Decrease'])) & \
                (asia['exch_location'] == 'Hong Kong')

    au_short1 = (asia['Headline sentiment'] == 'negative') & (asia['Report Type'].isin(['ad-hoc'])) & \
                (asia['exch_location'] == 'Australia')
    au_short2 = ((asia['Headline sentiment'] == 'negative') | (asia['Summary sentiment'] == 'negative')) & \
                (asia['Report Type'].isin(['Estimate Review', 'Rating Change'])) & (
                            asia['exch_location'] == 'Australia')
    au_short3 = (asia['Report Type'].isin(['Target Price Decrease'])) & (asia['exch_location'] == 'Australia')
    au_short4 = (asia['Summary sentiment'] == 'negative') & (asia['Report Type'].isin(['Earning\'s Review'])) & \
                (asia['exch_location'] == 'Australia')

    tw_short1 = (asia['Headline sentiment'] == 'negative') | (asia['Summary sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['Rating Change', 'Earning\'s Review', 'ad-hoc'])) & \
                (asia['exch_location'] == 'Taiwan')

    sk_short1 = (asia['Headline sentiment'] == 'negative') | (asia['Summary sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['ad-hoc', 'Earning\'s Review'])) & \
                (asia['exch_location'] == 'South Korea')
    sk_short2 = (asia['Headline sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['Rating Change', 'Estimate Change', 'Target Price Decrease'])) & \
                (asia['exch_location'] == 'South Korea')

    tk_short1 = (asia['Headline sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['ad-hoc', 'Earning\'s Review'])) & (asia['exch_location'] == 'Turkey')
    asia_short_trades_index = jp_short1 | jp_short2 | hk_short1 | au_short1 | au_short2 | au_short3 | au_short4 | \
                              sk_short1 | sk_short2 | tk_short1 | tw_short1
    asia.loc[asia_long_trades_index, 'side'] = 'long'
    asia.loc[asia_short_trades_index, 'side'] = 'short'

    asia = asia.loc[asia_long_trades_index | asia_short_trades_index].reset_index(drop=True)

    asia = get_pnl(asia)
    return asia

# Europe
def eu_df(df):
    europe = df[df['exch_region'] == 'Europe'].reset_index(drop=True)
    # europe = europe[~europe['exch_location'].isin(['Finland', 'Denmark', 'Greece', 'Austria'])].reset_index(
    #     drop=True)

    fc_long1 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Estimate Change', 'Target Price Increase'])) & \
               (europe['exch_location'] == 'France')
    fc_long2 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] != 'negative') & \
               (europe['Report Type'] == 'Earning\'s Review') & (europe['exch_location'] == 'France')
    fc_long3 = (europe['Headline sentiment'] != 'negative') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'] == 'ad-hoc') & (europe['exch_location'] == 'France')

    bg_long1 = (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Estimate Change'])) & \
               (europe['exch_location'] == 'Belgium')
    bg_long2 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Target Price Increase'])) & \
               (europe['exch_location'] == 'Belgium')

    gm_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Report Type'].isin(['ad-hoc'])) & \
               (europe['exch_location'] == 'Germany')
    gm_long2 = (europe['Report Type'].isin(['Target Price Increase'])) & \
               (europe['exch_location'] == 'Germany')
    gm_long3 = (europe['Summary sentiment'] == 'positive') & (europe['Report Type'].isin(['Estimate Change'])) & \
               (europe['exch_location'] == 'Germany')

    sp_long1 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'ad-hoc', 'Rating Change'])) &\
               (europe['exch_location'] == 'Spain')
    sp_long2 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] != 'negative') & \
               (europe['Report Type'].isin(['Estimate Change'])) & \
               (europe['exch_location'] == 'Spain')

    uk_long1 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Target Price Increase', 'ad-hoc'])) &\
               (europe['exch_location'] == 'United Kingdom')
    uk_long2 = (europe['Headline sentiment'] != 'negative') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Rating Change'])) & \
               (europe['exch_location'] == 'United Kingdom')

    pg_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Target Price Increase', 'ad-hoc'])) & (
                           europe['exch_location'] == 'Portugal')
    pg_long2 = (europe['Headline sentiment'] == 'positive') | (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review'])) & (
                           europe['exch_location'] == 'Portugal')
    pg_long3 = (europe['Headline sentiment'] != 'negative') | (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Rating Change'])) & (
                           europe['exch_location'] == 'Portugal')

    it_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['ad-hoc'])) & \
               (europe['exch_location'] == 'Italy')
    it_long2 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Rating Change', 'Target Price Increase'])) & \
               (europe['exch_location'] == 'Italy')

    sw_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Rating Change'])) & (
                           europe['exch_location'] == 'Switzerland')
    sw_long2 = (europe['Headline sentiment'] != 'negative') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Estimate Change'])) & (
                           europe['exch_location'] == 'Switzerland')

    dm_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Rating Change'])) & (
                           europe['exch_location'] == 'Denmark')

    as_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'ad-hoc'])) & (
                           europe['exch_location'] == 'Austria')

    fl_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Rating Change', 'Target Price Increase'])) & (
                           europe['exch_location'] == 'Finland')

    ir_long1 = (europe['Headline sentiment'] == 'positive') | (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review'])) & (
                           europe['exch_location'] == 'Ireland')
    ir_long2 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Estimate Change', 'ad-hoc'])) & (
                       europe['exch_location'] == 'Ireland')

    nw_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Rating Change', 'Estimate Change', 'ad-hoc'])) & (
                       europe['exch_location'] == 'Norway')
    nw_long2 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Target Price Increase'])) & (
                       europe['exch_location'] == 'Norway')

    gc_long1 = (europe['Headline sentiment'] == 'positive') | (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Greece')

    europe_long_trades_index = fc_long1 | fc_long2 | fc_long3 | bg_long1 | bg_long2 | sp_long1 | sp_long2 |\
                               gm_long1 | gm_long2 | gm_long3 | uk_long1 | uk_long2 | \
                               pg_long1 | pg_long2 | pg_long3 | sw_long1 | sw_long2 | dm_long1 | as_long1 | fl_long1 |\
                               it_long1 | it_long2 | ir_long1 | ir_long2 | nw_long1 | nw_long2 | gc_long1

    fc_short1 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) & \
                (europe['Report Type'].isin(['Earning\'s Review', 'ad-hoc'])) & (europe['exch_location'] == 'France')
    fc_short2 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] != 'positive') &\
                (europe['Report Type'].isin(['Rating Change'])) & (europe['exch_location'] == 'France')
    fc_short3 = (europe['Report Type'].isin(['Target Price Decrease'])) & (europe['exch_location'] == 'France')

    bg_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Rating Change', 'Target Price Decrease'])) & \
                (europe['exch_location'] == 'Belgium')
    bg_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Estimate Change'])) & (europe['exch_location'] == 'Belgium')
    bg_short3 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Headline sentiment'] != 'positive') | (europe['Summary sentiment'] != 'positive') & \
                (europe['Report Type'].isin(['Earning\'s Review', 'ad-hoc'])) & (europe['exch_location'] == 'Belgium')

    gm_short1 = (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Estimate Change'])) & (europe['exch_location'] == 'Germany')
    gm_short2 = (europe['Headline sentiment'] != 'positive') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Germany')
    gm_short3 = (europe['Report Type'].isin(['Target Price Decrease'])) & (europe['exch_location'] == 'Germany')
    gm_short4 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Summary sentiment'] != 'positive') & (europe['Report Type'].isin(['Earning\'s Review'])) &\
                (europe['exch_location'] == 'Germany')

    sp_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Rating Change'])) & (europe['exch_location'] == 'Spain')
    sp_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'Spain')
    sp_short3 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Headline sentiment'] != 'positive') & (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Spain')
    sp_short4 = (europe['Report Type'].isin(['Target Price Decrease'])) & (europe['exch_location'] == 'Spain')

    uk_short1 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc', 'Earning\'s Review'])) & (europe['exch_location'] == 'United Kingdom')
    uk_short2 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Headline sentiment'] != 'positive') & (europe['Report Type'].isin(['Estimate Change'])) & (europe['exch_location'] == 'United Kingdom')

    it_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc', 'Rating Change', 'Earning\'s Review'])) & \
                (europe['exch_location'] == 'Italy')

    pg_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Estimate Change', 'Target Price Decrease'])) & \
                (europe['exch_location'] == 'Portugal')
    pg_short2 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Headline sentiment'] != 'positive') & (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Portugal')

    sd_short1 = (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Rating Change', 'Estimate Change'])) & \
                (europe['exch_location'] == 'Sweden')
    sd_short2 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) & \
                (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'Sweden')

    sw_short1 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Headline sentiment'] != 'positive') & (europe['Report Type'].isin(['Earning\'s Review'])) & \
                (europe['exch_location'] == 'Switzerland')
    sw_short2 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc', 'Estimate Change', 'Rating Change'])) & (europe['exch_location'] == 'Switzerland')
    sw_short3 = (europe['Report Type'].isin(['Target Price Decrease'])) & (europe['exch_location'] == 'Switzerland')

    fl_short1 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Estimate Change', 'Rating Change', 'Target Price Decrease'])) & \
                (europe['exch_location'] == 'Finland')
    fl_short2 = (europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'Finland')

    ir_short1 = (europe['Headline sentiment'] != 'positive') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Estimate Change'])) & (europe['exch_location'] == 'Ireland')
    ir_short2 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Ireland')

    nw_short1 = (europe['Headline sentiment'] != 'positive') | (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change'])) & \
                (europe['exch_location'] == 'Norway')

    europe_short_trades_index = fc_short1 | fc_short2 | fc_short3 | gm_short1 | gm_short2 | gm_short3 | gm_short4 | \
                                bg_short1 | bg_short2 | bg_short3 | uk_short1 | uk_short2 | \
                                sp_short1 | sp_short2 | sp_short3 | sp_short4 | it_short1 | pg_short1 | pg_short2 | \
                                sw_short1 | sw_short2 | sw_short3 | sd_short1 | sd_short2 | fl_short1 | fl_short2 | \
                                ir_short1 | ir_short2 | nw_short1

    europe.loc[europe_long_trades_index, 'side'] = 'long'
    europe.loc[europe_short_trades_index, 'side'] = 'short'

    europe = europe.loc[europe_long_trades_index | europe_short_trades_index].reset_index(drop=True)
    europe = get_pnl(europe)
    return europe

# US
def am_df(df):
    am = df[df['exch_region'] == 'Americas'].reset_index(drop=True)

    us_long1 = list(np.where(  # (am['Headline sentiment'] == 'positive') &
        # (am['Summary sentiment'] == 'positive') &
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


def region_case_study(region='Asia', side='positive'):
    column = 'd0_r'
    train_data, test_data = DC.get_benchmark_test_data()
    train_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
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
                        expectancy_sort_by.loc[ind] = [0] * len(expectancy_sort_by.columns)

    expectancy_mapping = [expectancy_sort_by.loc[(x['Report Type'], x['Headline sentiment'], x['Summary sentiment'], x['exch_location'])]['Expectancy']
                          for _, x in test_data[['Report Type', 'Headline sentiment', 'Summary sentiment', 'exch_location']].iterrows()]
    test_data.loc[:, 'Expectancy'] = expectancy_mapping

    DL.toBT(pnl_df, f'{region}_{side}_pnl')
    plot_matrix(f'{region}_{side}_pnl')
    # pnl_df = Engine.portfolio_management(pnl_df)
    # DL.toBT(pnl_df, f'{region}_{side}_pnl(PM)')

    # strategy = f'{region}_pnl(PM)'
    # vis = visual(strategy)
    # vis.visual_job()

