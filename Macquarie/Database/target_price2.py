import pandas as pd
import re
import itertools
import numpy as np
from uti import DataLoader, Logger
from Broker import ricsregion
from Eikon import Eikon_update_price_enhanced
from Crawler import REPORT_TYPE_DICT香港天气历史

DL = DataLoader()
logger = Logger()


if __name__ == '__main__':
    price_df = DL.loadDB('price_df.csv', parse_dates=['Date', 'Time'])
    tpc_df8 = DL.loadDB('tpc_df8.csv', parse_dates=['Date', 'Time'])

    # tpc_df = price_df.loc[~price_df.index.isin(tpc_df8.index)]
    tpc_df = price_df[price_df['exch_location'] == 'United States']
    tpc_df = tpc_df[tpc_df['Report Types'].fillna('').str.contains('Target Price Change')]
    tpc_df_blank = tpc_df.copy()
    # tpc_df = tpc_df.sort_values('Time', ascending=True)

    # tpc_df = DL.loadDB('tpc_df8.csv', parse_dates=['Date', 'Time'])
    r = re.compile(r'\btarget price [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                                      
                   r'\bpt of [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp of [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price of [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target of [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bpt to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   
                   r'\bpt at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   
                   r'\bpt is now set at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp is now set at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price is now set at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target is now set at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   
                   r'\bpt rises to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp rises to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget rises to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice rises to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   
                   r'\bpt now at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp now at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price now at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target now at [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bpt is [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp is [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price is [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target is [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bpt are [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp are [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price are [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target are [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bwith a [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) tp|'
                   r'\bwith a [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) pt|'
                   r'\bwith a [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) target price|'
                   r'\bwith a [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) price target|'
                   
                   r'\b[a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) tp|'
                   r'\b[a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) pt|'
                   r'\b[a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) target price|'
                   r'\b[a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%) price target|'
                   
                   r'\bprice target revised [a-z]*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price .*?to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target .*?to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bpt .*?to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp .*?to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   
                   r'\bpt [a-z]*? from.*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp [a-z]*? from.*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price [a-z]*? from.*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target [a-z]*? from.*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   
                   r'\bpt revised [a-z]*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btp revised [a-z]*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price revised [a-z]*? to? [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'

                   , flags=re.I)

    # tpc_df_blank = tpc_df[tpc_df['Summary'].str.contains('%')].copy()
    tpc_df_blank['TPS'] = None
    count = 0
    years_tbd = map(str, range(2019, 2031))
    for i, row in tpc_df_blank.iterrows():
        summary = row['Summary']
        if type(summary) != str:
            summary = ''
        summary = summary.replace(',', '')
        search_result = r.findall(summary)
        search_result = [x for x in list(itertools.chain(*search_result)) if len(x) > 0]
        search_result = [x for x in search_result if x not in years_tbd]
        if len(search_result) > 0:
            print(summary)
            print(search_result)
            if len(np.unique(search_result)) > 1:
                tpc_df_blank.loc[i, 'TPS'] = str(search_result)
            else:
                tpc_df_blank.loc[i, 'TPS'] = str(search_result[0])
            count += 1
    print(count)
    DL.toDB(tpc_df_blank, 'tpc_df_us.csv')

    # r = re.compile(
    #                r'\bprice target to [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
    #                , flags=re.I)
    # summary = 'Deep sea trawling rights are up for renewal in 2020. The Viking Inshore Fishing matter (lost 60% of its long-term inshore fishing rights ) has left the deep sea trawling fishery worried about the allocation of long-term deep sea trawling rights in 2020. We have run a sensitivity analysis which suggests that, assuming I&J loses around 20% of its revenue and margins decline to 8.5%, there could be a 5% negative impact on our base-case valuation. Post the recent FY17 results, which were broadly in line, we make minor changes to our estimates. We now forecast FY18e EPS growth of 9%, putting AVI on an implied forward PE of 18x. We continue to argue that AVI needs to show stronger earnings growth to justify its premium valuation of c18x. We maintain our Neutral rating and increase our TP slightly to R102 p/sh as we roll forward our valuation models.'
    #
    # search_result = r.findall(summary)
    # search_result = [x for x in list(itertools.chain(*search_result)) if len(x) > 0]

    # tpc_df_blank = tpc_df[tpc_df['TP'].isna()].copy()

    # tpc_df_blank = tpc_df_blank.set_index(['Time', 'Summary'])
    # tpc_df_blank.loc[tpc_df_multi.index, 'TP'] = tpc_df_multi['TP']
    # DL.toDB(tpc_df_blank.reset_index(), 'tpc_df7.csv')

# tpc_df8['TPS2'] = tpc_df8.groupby(['Ticker', 'Date'])['TPS'].apply(lambda x: x.fillna(method='ffill'))
# tpc_df8 = tpc_df8.sort_values('Date', ascending=True)
# tpc_df8['TPS2'] = tpc_df8.groupby(['Ticker'])['TPS'].apply(lambda x: x.fillna(method='ffill'))
# tpc_df8 = tpc_df8.sort_values(['Ticker', 'Date'], ascending=True)
# tpc_df8['TPS2'] = tpc_df8.groupby(['Ticker'])['TPS'].apply(lambda x: x.fillna(method='ffill'))
# DL.toDB(tpc_df8, 'tpc_df_us_tocheck.csv')
