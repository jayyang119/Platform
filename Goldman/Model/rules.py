import numpy as np
import pandas as pd
from Broker import get_pnl
from uti import DataLoader
from Backtest import plot_matrix

DL = DataLoader()


def asia_df(df):
    """
        Asia filtering rules for Goldman reports.
    """
    asia = df[df['exch_region'] == 'Asia'].reset_index(drop=True)  # A copy
    if len(asia) == 0:
        return pd.DataFrame()
    # Long trades
    jp_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') & \
               (asia['Report Type'] == 'Target Price Increase') & (asia['exch_location'] == 'Japan')
    jp_long2 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') & \
               (asia['Report Type'] == 'ad-hoc') & (asia['exch_location'] == 'Japan')

    hk_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') & \
               (asia['Report Type'].isin(['Earning\'s Review', 'Rating Change', 'Target Price Increase', 'ad-hoc'])) & \
               (asia['exch_location'] == 'Hong Kong')

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
    cn_long2 = ((asia['Headline sentiment'] == 'positive') | (asia['Summary sentiment'] == 'positive')) & \
               (asia['Report Type'].isin(['Target Price Increase'])) & (asia['exch_location'] == 'China')

    sk_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] != 'negative') & \
               (asia['Report Type'].isin(['Rating Change'])) & (asia['exch_location'] == 'South Korea')

    tw_long1 = (asia['Headline sentiment'] == 'positive') & (asia['Summary sentiment'] == 'positive') & \
               (asia['Report Type'].isin(['Earning\'s Review', 'Rating Change'])) & (asia['exch_location'] == 'Taiwan')

    # Short trades
    jp_short1 = (asia['Headline sentiment'] == 'negative') & (asia['Report Type'] == 'ad-hoc') & (
                asia['exch_location'] == 'Japan')

    hk_short1 = (asia['Headline sentiment'] == 'negative') & \
                (asia['Report Type'].isin(
                    ['Earning\'s Review', 'Estimate Change', 'Rating Change', 'Target Price Decrease'])) & \
                (asia['exch_location'] == 'Hong Kong')
    hk_short2 = (asia['Summary sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['Estimate Change', 'Rating Change', 'ad-hoc'])) & \
                (asia['exch_location'] == 'Hong Kong')

    au_short1 = (asia['Headline sentiment'] == 'negative') & (asia['Report Type'].isin(['ad-hoc'])) & \
                (asia['exch_location'] == 'Australia')
    au_short2 = ((asia['Headline sentiment'] == 'negative') | (asia['Summary sentiment'] == 'negative')) & \
                (asia['Report Type'].isin(['Estimate Review', 'Rating Change'])) & (
                            asia['exch_location'] == 'Australia')
    au_short3 = (asia['Report Type'].isin(['Target Price Decrease'])) & (asia['exch_location'] == 'Australia')
    au_short4 = (asia['Summary sentiment'] == 'negative') & (asia['Report Type'].isin(['Earning\'s Review'])) & \
                (asia['exch_location'] == 'Australia')

    cn_short1 = (asia['Report Type'].isin(['Target Price Decrease'])) & (asia['exch_location'] == 'China')

    tw_short1 = (asia['Headline sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['Rating Change', 'Earning\'s Review'])) & \
                (asia['exch_location'] == 'Taiwan')

    sk_short1 = (asia['Headline sentiment'] == 'negative') & \
                (asia['Report Type'].isin(
                    ['Rating Change', 'Estimate Change', 'ad-hoc', 'Earning\'s Review', 'Target Price Decrease'])) & \
                (asia['exch_location'] == 'South Korea')

    tk_short1 = (asia['Headline sentiment'] == 'negative') & \
                (asia['Report Type'].isin(['ad-hoc', 'Earning\'s Review'])) & (asia['exch_location'] == 'Turkey')

    asia_long_trades_index = jp_long1 | jp_long2 | hk_long1 | au_long0 | au_long1 | au_long2 | \
                             cn_long1 | cn_long2 | sk_long1 | tw_long1
    asia_short_trades_index = jp_short1 | hk_short1 | hk_short2 | au_short1 | au_short2 | au_short3 | au_short4 | cn_short1 | \
                              sk_short1 | tk_short1 | tw_short1
    asia.loc[asia_long_trades_index, 'side'] = 'long'
    asia.loc[asia_short_trades_index, 'side'] = 'short'

    asia = asia.loc[asia_long_trades_index | asia_short_trades_index].reset_index(drop=True)

    return asia

# Europe
def eu_df(df):
    """
        Europe filtering rules for Goldman reports.
    """
    europe = df[df['exch_region'] == 'Europe'].reset_index(drop=True)
    if len(europe) == 0:
        return pd.DataFrame()
    europe = europe[~europe['exch_location'].isin(['Finland', 'Denmark', 'Greece', 'Austria'])].reset_index(
        drop=True)

    fc_long1 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Estimate Change', 'Target Price Increase'])) & \
               (europe['exch_location'] == 'France')

    gm_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Report Type'].isin(['ad-hoc'])) & \
               (europe['exch_location'] == 'Germany')
    gm_long2 = (europe['Report Type'].isin(['Target Price Increase'])) & \
               (europe['exch_location'] == 'Germany')
    gm_long3 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change'])) & (
                           europe['exch_location'] == 'Germany')

    uk_long1 = (europe['Headline sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change'])) & (
                           europe['exch_location'] == 'United Kingdom')
    uk_long2 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Rating Change', 'Target Price Increase', 'ad-hoc'])) & \
               (europe['exch_location'] == 'United Kingdom')

    pg_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Target Price Increase', 'ad-hoc'])) & (
                           europe['exch_location'] == 'Portugal')

    it_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['ad-hoc', 'Rating Change'])) & \
               (europe['exch_location'] == 'Italy')
    it_long2 = (europe['Headline sentiment'] == 'positive') & (europe['Report Type'].isin(['Earning\'s Review'])) & \
               (europe['exch_location'] == 'Italy')

    sw_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Estimate Change', 'Rating Change'])) & (
                           europe['exch_location'] == 'Switzerland')

    dm_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Rating Change'])) & (
                           europe['exch_location'] == 'Denmark')

    as_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'ad-hoc'])) & (
                           europe['exch_location'] == 'Austria')

    fl_long1 = (europe['Headline sentiment'] == 'positive') & (europe['Summary sentiment'] == 'positive') & \
               (europe['Report Type'].isin(['Earning\'s Review', 'Target Price Increase'])) & (
                           europe['exch_location'] == 'Finland')

    europe_long_trades_index = fc_long1 | gm_long1 | gm_long2 | gm_long3 | uk_long1 | uk_long2 | \
                               pg_long1 | sw_long1 | dm_long1 | as_long1 | fl_long1 | it_long1 | it_long2

    fc_short1 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) & \
                (europe['Report Type'].isin(['Earning\'s Review', 'ad-hoc'])) & (europe['exch_location'] == 'France')
    fc_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Report Type'].isin(['Rating Change'])) & (
                europe['exch_location'] == 'France')
    fc_short3 = (europe['Report Type'].isin(['Target Price Decrease'])) & (europe['exch_location'] == 'France')

    gm_short1 = (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change'])) & (
                            europe['exch_location'] == 'Germany')
    gm_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Germany')
    gm_short3 = (europe['Report Type'].isin(['Target Price Decrease'])) & (europe['exch_location'] == 'Germany')

    bg_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(
                    ['Earning\'s Review', 'Estimate Change', 'Rating Change', 'Target Price Decrease'])) & \
                (europe['exch_location'] == 'Belgium')
    bg_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Belgium')

    uk_short1 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] != 'positive') & \
                (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'United Kingdom')
    uk_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'United Kingdom')

    sp_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc', 'Rating Change'])) & \
                (europe[
                     'exch_location'] == 'Spain')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'
    sp_short2 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'Spain')
    sp_short3 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) & \
                (europe['Report Type'].isin(['Estimate Change'])) & (europe['exch_location'] == 'Spain')

    it_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc', 'Rating Change'])) & \
                (europe[
                     'exch_location'] == 'Italy')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'

    sw_short1 = (europe['Headline sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review', 'Estimate Change', 'Target Price Decrease'])) & \
                (europe[
                     'exch_location'] == 'Switzerland')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'
    sw_short2 = (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['ad-hoc'])) & (europe['exch_location'] == 'Switzerland')

    sd_short1 = (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Rating Change'])) & \
                (europe[
                     'exch_location'] == 'Sweden')  # 'Earning\'s Review', 'Estimate Change', , 'Target Price Decrease'
    sd_short2 = ((europe['Headline sentiment'] == 'negative') | (europe['Summary sentiment'] == 'negative')) & \
                (europe['Report Type'].isin(['Earning\'s Review'])) & (europe['exch_location'] == 'Sweden')

    fl_short1 = (europe['Headline sentiment'] == 'negative') & (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(
                    ['Earning\'s Review', 'Estimate Change', 'Rating Change', 'Target Price Decrease'])) & \
                (europe['exch_location'] == 'Finland')

    nw_short1 = (europe['Summary sentiment'] == 'negative') & \
                (europe['Report Type'].isin(['Earning\'s Review'])) & \
                (europe['exch_location'] == 'Norway')

    europe_short_trades_index = fc_short1 | fc_short2 | fc_short3 | gm_short1 | gm_short2 | gm_short3 | bg_short1 | \
                                bg_short2 | uk_short1 | uk_short2 | sp_short1 | sp_short2 | sp_short3 | it_short1 | \
                                sw_short1 | sw_short2 | sd_short1 | sd_short2 | fl_short1 | nw_short1

    europe.loc[europe_long_trades_index, 'side'] = 'long'
    europe.loc[europe_short_trades_index, 'side'] = 'short'

    europe = europe.loc[europe_long_trades_index | europe_short_trades_index].reset_index(drop=True)

    return europe

# US
def am_df(df):
    """
        America filtering rules for Goldman reports.
    """
    am = df[df['exch_region'] == 'Americas'].reset_index(drop=True)
    if len(am) == 0:
        return pd.DataFrame()

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

    if len(am_trades.loc[am_long_trades_index]) > 0:
        am_trades.loc[am_long_trades_index, 'side'] = 'long'
    if len(am_trades.loc[am_short_trades_index]) > 0:
        am_trades.loc[am_short_trades_index, 'side'] = 'short'
    am_trades = am_trades.drop_duplicates().reset_index(drop=True)

    return am_trades


#
# def get_expectancy_based_on_sort_by(train_data, test_data, sort_by=[]):
#     inputs_list = ['No. of trades', 'd0_r', 'side', 'Report Type']
#     group_by_list = ['side', 'Report Type']
#
#     inputs_list.extend([sort_by])
#     group_by_list.extend([sort_by])
#
#     expectancy_sort_by = benchmark_expectancy(train_data, 'd0_r', inputs=inputs_list, group_by=group_by_list)
#
#     for by_i in test_data[sort_by].unique():
#         for side in ['long', 'short']:
#             for report_type in test_data['Report Type'].unique():
#                 if (by_i, side, report_type) not in expectancy_sort_by.index:
#                     expectancy_sort_by.loc[(by_i, side, report_type)] = [0] * len(expectancy_sort_by.columns)
#
#     test_data['Expectancy'] = [expectancy_sort_by.loc[(x[sort_by], x['side'], x['Report Type'])]['Expectancy']
#                                for _, x in test_data[[sort_by, 'side', 'Report Type']].iterrows()]
#     return
