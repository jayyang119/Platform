import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from uti import DataLoader, Logger
from Broker import get_pnl

logger = Logger()
DL = DataLoader()


class backtest_engine:
    def __init__(self, daily_position=18, skew=0, region_position=6, min_region_position=0):
        self._daily_position = daily_position
        self._skew = skew
        self._region_position = region_position
        self._min_region_position = min_region_position
        print(f'Engine current settings: \ndaily_position {self._daily_position} \nregion_position {self._region_position} \n'
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

    """
    Backtest result with risk management
    - limit up to #daily_position of trades per day per market.
    """

    def portfolio_management(self, test_data):
        deque = []
        # df = test_data[test_data['Date']==datetime(2022, 1, 5)]
        for _, df in test_data.groupby('Date'):
            # Debug
            # df = test_data[test_data['Date'] == datetime(2021, 4, 12)]
            #
            positive_expectancy = df[['exch_location', 'exch_region', 'side', 'Expectancy']].copy(deep=True)
            positive_expectancy = positive_expectancy[positive_expectancy['Expectancy'] > 0]
            positive_expectancy = positive_expectancy.sort_values('Expectancy', ascending=False)
            positive_expectancy['side2'] = positive_expectancy['side'].replace(
                {'long': 1, 'short': -1, 'positive': 1, 'negative': -1})

            region_queue = defaultdict(list)  # Keep count
            region_skew = defaultdict(int)  # Keep balance
            region_temp = defaultdict(list)
            daily_queue = []
            total_index = list(positive_expectancy.index)

            if self._min_region_position > 0:
                min_region_side_position = self._min_region_position // 2
                for region, region_df in positive_expectancy.groupby(['exch_region']):
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
                region = positive_expectancy.loc[i]['exch_region']
                side2 = positive_expectancy.loc[i]['side2']

                if len(region_queue[region]) >= self._region_position:
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
                    if len(region_queue[region]) >= self._region_position:
                        continue
                    region_skew[region] += side2
                    region_queue[region].append(i)
                    daily_queue.append(i)

            deque.extend(daily_queue)
        return test_data.loc[deque]

    def portfolio_management2(self, test_data, sort_by="Expectancy"):
        total_trades = pd.DataFrame()
        # for _, df in test_data.groupby('Date'):
        for _, df in test_data.groupby('d0_date'):
            df = df[df[sort_by] > 0].reset_index(drop=True)
            df.sort_values(by=sort_by, ascending=False, inplace=True)

            if len(df) == 0:
                return df

            long_queue = df[df["side"] == "long"]
            short_queue = df[df["side"] == "short"]

            pending_trades = pd.DataFrame()
            min_length = min(len(long_queue), len(short_queue))

            if self._skew == 0:
                min_length_for_balance = int(min(min_length, math.floor(self._daily_position / 2)))
                pending_trades = pd.concat(
                    [
                        long_queue.iloc[:min_length_for_balance],
                        short_queue.iloc[:min_length_for_balance],
                    ]
                )

            elif self._skew > 0:
                for i in range(min_length + self._skew):
                    if i < len(long_queue):
                        pending_trades = pd.concat([pending_trades, long_queue.iloc[[i]]])
                    if i < len(short_queue):
                        pending_trades = pd.concat([pending_trades, short_queue.iloc[[i]]])
                    trades_skew = abs(np.where(pending_trades["side"] == "long", 1, -1).sum())
                    if trades_skew == self._skew:
                        break
                    if len(pending_trades) >= self._daily_position:
                        break
            for _, df2 in pending_trades.groupby('exch_region'):
                total_trades = pd.concat([total_trades, df2.iloc[:self._region_position]], axis=0)

        return total_trades

    def backtest_job(self, test_data):
        test_data = get_pnl(test_data)
        results_df = self.portfolio_management(test_data)
        return results_df
