import pandas as pd
import re
import itertools
import numpy as np
from uti import DataLoader, Logger
from Broker import ricsregion
from Eikon import Eikon_update_price_enhanced
from Crawler import REPORT_TYPE_DICT

DL = DataLoader()
logger = Logger()

def tpc_scanner(summary_list: list) -> list:
    """
        Scan target price from the summaries of the reports that are classified as Target Price Change.

    """
    tpc_list = []
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
                   
                   r'\btp [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bpt [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\btarget price [a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'
                   r'\bprice target[a-z]{0,3}[ ]{0,1}[\$|\€\£\¥]{0,1}(\d*\.?\d+)(?<!%)|'

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

    count = 0
    years_tbd = map(str, range(2019, 2031))

    for summary in summary_list:
        summary = str(summary).replace(',', '')

        result = r.findall(summary)
        result = [x for x in list(itertools.chain(*result)) if len(x) > 0]
        result = [x for x in result if x not in years_tbd]
        if len(result) > 0:
            if len(np.unique(result)) > 1:
                # Store all the possible target prices for ARTIFICIAL intelligence to select
                tpc_list.append(str(result))
            elif result[0] is not None:
                tpc_list.append(str(result[0]))
            else:
                tpc_list.append('')
        else:
            tpc_list.append('')

    logger.info(f'Total number of {count} of TPC needs to modify with the help of ARTIFICIAL intelligence.')

    return tpc_list


def rating_scanner(summary_list: list) -> list:
    rating_list = []

    rt_pattern = re.compile(r'\b([a-z]{1,7}) rating', flags=re.I)
    for summary in summary_list:
        if type(summary) is not str:
            summary = ''

        summary = str(summary).replace(',', '').lower()
        if 'rating' in summary:
            # print('Has ratings')
            result = rt_pattern.findall(summary)
            result = list(np.unique([x for x in result if x != '' and x in ['neutral', 'buy', 'sell']]))

        else:
            result = []

        if len(result) > 1:
            print(summary)
            curr_rating = input(
                'Please manually check for current ratings, if unfound or multiple tickers, please press Enter:')
            rating_list.append(curr_rating)

        elif len(result) == 1:
            rating_list.append(result[0])
        else:
            rating_list.append('')

    return rating_list


def rc_scanner(summary_list: list) -> list:
    """
        Scan rating change from the summaries of the reports that are classified as Rating Change.

    """
    rc = []
    rc_list = []
    ug_pattern = re.compile(r'\bupgrad[a-z]{1,3} [a-z0-9]+ to(?<!%) ([a-z]{1,7})(?<!%)|'
                            r'\bupgrad[a-z]{1,3} to ([a-z]{1,7})(?<!%)', flags=re.I)
    dg_pattern = re.compile(r'\bdowngrad[a-z]{1,3} [a-z0-9]+ to(?<!%) ([a-z]{1,7})(?<!%)|'
                            r'\downgrad[a-z]{1,3} to ([a-z]{1,7})(?<!%)', flags=re.I)


    for summary in summary_list:
        if type(summary) is not str:
            summary = ''

        summary = str(summary).replace(',', '').lower()

        upgrade_result = "upgrad" in summary.lower()
        downgrade_result = "downgrad" in summary.lower()

        rc.append([upgrade_result, downgrade_result])

        if 'upgrad' in summary:
            if 'downgrad' not in summary:
                result = ug_pattern.findall(summary)
            else:
                # AI
                print(summary)

                curr_rating = input('Please manually check for current ratings, if unfound or multiple tickers, please press Enter:')
                # if curr_rating == '':
                #     curr_rating = ''

                rc_list.append(curr_rating)
                continue

        elif 'downgrad' in summary:
            result = dg_pattern.findall(summary)

        else:
            result = []

        print(result)
        if len(result) > 0:
            result = result[0]
            result = [x for x in result if x != '']
            if len(result) > 0:
                if result[0] in ['neutral', 'buy', 'sell']:
                    rc_list.append(result[0])
                    continue

        rc_list.append('')


    return rc, rc_list

if __name__ == '__main__':
    price_df = DL.loadDB('price_df.csv', parse_dates=['Date', 'Time'])
    tpc_df8 = DL.loadDB('tpc_df8.csv', parse_dates=['Date', 'Time'])

    tpc_df = price_df.loc[~price_df.index.isin(tpc_df8.index)]
    tpc_df = tpc_df[tpc_df['report_types'].fillna('').str.contains('Target Price Change')]
    tpc_df_blank = tpc_df.copy()


