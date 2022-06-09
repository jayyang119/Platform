import math
import pandas as pd
import numpy as np
from collections import defaultdict
from uti import DataLoader, Logger
from Broker import get_pnl

logger = Logger()
DL = DataLoader()


class BacktestEngine:
    """
    Backtest result with risk management
    - _daily_position: maximum trades per day per market.
    - _skew: maximum skewness within each market.
    - _region_position: maximum trades per day per region.
    - _min_region_position: minimum trades per day per region.
    """

    def __init__(self, daily_position=18, skew=0, region_position=6, min_region_position=0):
        self._daily_position = daily_position
        self._skew = skew
        self._region_position = region_position
        self._min_region_position = min_region_position
        logger.info(f'Engine current settings: \ndaily_position {self._daily_position} \nregion_position {self._region_position} \n'
                    f'skewness: {self._skew}')

    def set_parameters(self, **kwarg):
        for key, value in kwarg.items():
            if key == 'daily_position':
                self._daily_position = value
            elif key == 'skew':
                self._skew = value
            elif key == 'region_position':
                self._region_position = value
            elif key == 'min_region_position':
                self._min_region_position = value
            else:
                logger.error(f'Input error, please check your input:{key} {value}')
        print(
            f'Engine current settings: \ndaily_position {self._daily_position} \n region_position {self._region_position} \n'
            f'skewness: {self._skew}')


    def portfolio_management(self, _test_data, random_pick=False, rank_by='Expectancy'):
        deque = []
        # pm_region_position = {'Americas': 6, 'Europe': 6, 'Korea & Japan': 2, 'Southeast Asia': 2, 'Taiwan & Singapore': 2,
        #                       'Hong Kong': 4, 'South Africa': 2}
        # _test_data = test_data[test_data['pm_region'].isin(pm_region_position.keys())].reset_index(drop=True)

        for _, df in _test_data.groupby('Date'):
            # Debug
            # df = test_data[test_data['Date'] == datetime(2021, 4, 12)]
            #
            positive_expectancy = df[['pm_region', 'side', rank_by]].copy(deep=True)
            positive_expectancy = positive_expectancy[positive_expectancy['side'] != 'neutral']
            if not random_pick:
                positive_expectancy = positive_expectancy[positive_expectancy[rank_by] > 0]
            positive_expectancy = positive_expectancy.sort_values(rank_by, ascending=False)
            positive_expectancy['side2'] = positive_expectancy['side'].replace(
                {'long': 1, 'short': -1, 'positive': 1, 'negative': -1})

            region_queue = defaultdict(list)  # Keep count
            region_skew = defaultdict(int)  # Keep balance
            region_temp = defaultdict(list)
            daily_queue = []
            total_index = list(positive_expectancy.index)

            if self._min_region_position > 0:
                min_region_side_position = self._min_region_position // 2
                for region, region_df in positive_expectancy.groupby(['pm_region']):
                    region_df_long = region_df[region_df['side2'] > 0]
                    region_df_short = region_df[region_df['side2'] < 0]

                    available_trade_num = min(min(len(region_df_long), len(region_df_short)),
                                              min_region_side_position)

                    region_index_long = list(region_df_long.iloc[:available_trade_num].index)
                    region_index_short = list(region_df_short.iloc[:available_trade_num].index)
                    region_queue[region].extend(region_index_long)
                    region_queue[region].extend(region_index_short)

                    daily_queue.extend(region_queue[region])
                total_index = list(set(total_index) - set(daily_queue))

            while len(total_index) > 0 and len(daily_queue) < self._daily_position:
                i = total_index.pop(0)
                # region = positive_expectancy.loc[i]['exch_region']
                region = positive_expectancy.loc[i]['pm_region']
                side2 = positive_expectancy.loc[i]['side2']

                if len(region_queue[region]) >= self._daily_position:
                    continue

                if abs(region_skew[region]) < self._skew:
                    region_queue[region].append(i)
                    daily_queue.append(i)
                    region_skew[region] += side2

                elif region_skew[region] * side2 >= 0:  # Same sides
                    region_temp[region].append(i)
                    region_skew[region] += side2
                else:  # Different sides
                    if len(region_temp[region]) > 0:
                        i_temp = region_temp[region].pop(0)
                        region_queue[region].append(i_temp)
                        daily_queue.append(i_temp)
                        # region_skew[region] -= side2
                    # if len(region_queue[region]) >= self._region_position:
                    if len(region_queue[region]) >= self._region_position:
                        continue
                    region_skew[region] += side2
                    region_queue[region].append(i)
                    daily_queue.append(i)

            deque.extend(daily_queue)

        return _test_data.loc[deque]


    def backtest_job(self, test_data):
        test_data = get_pnl(test_data)
        results_df = self.portfolio_management(test_data)
        return results_df
