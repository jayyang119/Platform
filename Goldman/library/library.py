# library.py
import pdb
import os
import pandas as pd
import numpy as np
import traceback
import json
import math
from collections import OrderedDict
from flatten_json import flatten_json
from library.finbert_senti import *
from Path import PLATFORM_PATH
from library.dictionary import *
from uti import Logger

logger = Logger()


class Dataset:
    def __init__(self, _df):
        self.df = _df

    def _drop_duplicate_entries(self):
        """
        This function drops the duplicate entries(rows).
        """
        num_duplicated = sum(self.df.duplicated())
        if num_duplicated > 0:
            self.df.drop_duplicates(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            print(f'Dropped {num_duplicated} duplicated entries.')

    def _clean_ticker(self, mode='stable'):
        """
        This function changes the ticker exchange code to composite code (if available).

        ---------------------------------------------------------------------
        
        Example: 
        
        - AAPL UN Equity -> AAPL US Equity

        Note: Germany stocks are changed to 'GY' (exchange code) instead of 'GR' (composite code)
        """

        if mode != 'eikon':
            splited_ticker = self.df['ticker'].str.split(' ', 1, expand=True)
            ticker_lst = splited_ticker[0]
            exch_code_lst = splited_ticker[1]

        else:
            ticker_lst = self.df['Ticker']
            exch_code_lst = self.df['Region']

        logger.info(exch_code_lst)

        mapped_exch_code_lst = exch_code_lst.map(ex_code_to_com_code)
        self.df['ticker'] = ticker_lst + ' ' + mapped_exch_code_lst

        if sum(pd.isnull(exch_code_lst)) > 0:
            print('Error: One or more ticker(s) does not have exchange code (i.e. ticker not in correct format).')

        for each in exch_code_lst.unique():
            if (each not in ex_code_to_com_code.keys()) & (each is not None):
                print(f'Error: Could not find exch_code \'{each}\' in dict(ex_code_to_com_code).')

    def _add_exch_location_col(self, mode='stable'):
        """
        This function adds `exch_location` column to the df (based on exch_code).

        ---------------------------------------------------------------------
        
        Example: 

        - MC FP Equity -> exch_location = 'France'
        exch_code_lst = self.df['ticker'].str.split(' ', expand=True)[1]
        """
        if mode != 'eikon':
            splited_ticker = self.df['ticker'].str.split(' ', 1, expand=True)
            ticker_lst = splited_ticker[0]
            exch_code_lst = splited_ticker[1]

        else:
            ticker_lst = self.df['Ticker']
            exch_code_lst = self.df['Region']

        mapped_exch_location = exch_code_lst.map(code_to_location_or_exch)
        self.df['exch_location'] = mapped_exch_location

        for each in exch_code_lst.unique():
            if (each not in code_to_location_or_exch.keys()):
                print(
                    f'Error: Could not find exchange location, exch_code \'{each}\' is not in dict(code_to_location_or_exch).')

    def _add_exch_region_col(self):
        """
        This function adds `exch_region` column to the df (based on `exch_location`).
        
        ---------------------------------------------------------------------
        
        Example: 
        
        - exch_location = 'France', exch_region = 'Europe'

        Note: Will return NaN if it is a country that we won't trade in.
        """
        exch_location_lst = self.df['exch_location']

        mapped_exch_region = exch_location_lst.map(location_details_dict)
        mapped_exch_region = [each['region'] if type(each) == type(dict()) else np.NaN for each in mapped_exch_region]
        self.df['exch_region'] = mapped_exch_region

        excl_lst = [each for each in exch_location_lst.unique() if each not in location_details_dict.keys()]
        if len(excl_lst) > 0:
            try:
                excl_lst.sort()
            except Exception as e:
                print(traceback.format_exc())

            print(f'{excl_lst} not in dict(location_details_dict).')

    def _drop_entries_with_non_tradable_location(self):
        """
        This function removes entries with exchange(location) that we won't trade, such as 'Chile', 'India', 'Mexico', etc.
        """
        df = self.df.copy()
        df = df[df['exch_location'].isin(location_details_dict.keys())]
        df.reset_index(drop=True, inplace=True)
        num_dropped = len(self.df) - len(df)
        self.df = df

        if num_dropped > 0:
            print(f'Dropped {num_dropped} entries from the above locations.')
            print(set(df['Ticker']).difference(set(self.df['Ticker'])))
        self.df = df

    def _add_local_date_and_time_and_release_period_col(self, tz_name, mode='stable'):
        """
        This function adds `date_local`, `time_local` and `release_period` to the df (based on `exch_location` and `publish_date_and_time`).
        
        ---------------------------------------------------------------------
        
        Example: 
        
        - ('United Kingdom', '2019-12-21 00:27:00') -> ('2019-12-20', '16:27:00', 'Within')
        """
        exchange_location = self.df['exch_location']
        if mode != 'eikon':
            publish_datetime = self.df['publish_date_and_time']
        else:
            publish_datetime = self.df['Time']

        pytz_time_zone = exchange_location.apply(lambda x: location_details_dict[x]['tz_name'])
        mkt_open_time = exchange_location.apply(lambda x: location_details_dict[x]['open_time'])
        mkt_close_time = exchange_location.apply(lambda x: location_details_dict[x]['close_time'])

        mkt_open_time = pd.to_datetime(mkt_open_time).dt.time
        mkt_close_time = pd.to_datetime(mkt_close_time).dt.time

        local_datetime_dict = {}
        for each_timezone in pytz_time_zone.unique():
            datetime_lst = pd.to_datetime(pd.Series(publish_datetime[pytz_time_zone == each_timezone]))

            # define timezone for publish datatime
            each_original_datetime = datetime_lst.dt.tz_localize(tz_name)

            # convert publish datetime to local datetime (will adjust DST automatically)
            each_local_datetime = each_original_datetime.dt.tz_convert(each_timezone)

            # remove the timezone
            each_local_datetime = each_local_datetime.dt.tz_localize(None)

            temp_dict = dict(zip(datetime_lst.index, each_local_datetime))
            local_datetime_dict.update(temp_dict)

        local_datetime_dict = OrderedDict(sorted(local_datetime_dict.items()))
        local_datetime = pd.Series(local_datetime_dict.values())
        try:
            date_local = local_datetime.dt.date
            time_local = local_datetime.dt.time
        except Exception as e:
            print(local_datetime)
        # conditions
        is_weekday = (local_datetime.dt.dayofweek <= 4)
        before_trading_hours = (time_local < mkt_open_time)
        within_trading_hours = (time_local >= mkt_open_time) & (time_local <= mkt_close_time)

        conditions = [is_weekday & before_trading_hours, is_weekday & within_trading_hours]
        choices = ['Before', 'Within']
        release_period = np.select(conditions, choices, default='After')

        self.df['date_local'] = date_local
        self.df['time_local'] = time_local
        self.df['release_period'] = release_period

    def _add_updated_ticker_col(self):
        """
        This function adds `ticker_updated` column to the df (based on `ticker` and `date_local`).
        
        ---------------------------------------------------------------------
        
        Example:
        
        - Previous ticker = BIOP US (Bioptix)
        
        - New ticker = RIOT US (Riot Blockchain)
        """
        ticker_change_df = pd.read_csv(f'{PLATFORM_PATH}/library/data/ticker_symbol_change.csv')
        ticker_change_df['effective_date'] = pd.to_datetime(ticker_change_df['effective_date']).dt.strftime('%Y-%m-%d')
        ticker_change_df.sort_values(by='effective_date')

        # columns to check and update, change 'GY' to 'GR' for searching ticker symbol change
        ticker = self.df['ticker'].str.replace(' GY', ' GR').copy()
        update_date = pd.to_datetime(self.df['date_local']).dt.strftime('%Y-%m-%d').copy()

        # check whether the ticker is in the 'old_ticker' column
        ticker_not_in_list = ~ticker.isin(ticker_change_df['old_ticker'])
        finished_update = ticker_not_in_list

        # loop if update not yet finished
        while (finished_update == False).sum() > 0:

            ticker_not_in_list = ~ticker.isin(ticker_change_df['old_ticker'])
            finished_update = ticker_not_in_list | finished_update
            index_to_check = finished_update[finished_update == False].index

            # only check index that hasn't finished update
            for each in index_to_check:

                each_ticker = ticker[each]
                each_update_date = update_date[each]

                try:
                    # update the ticker and update_date
                    index = np.where((ticker_change_df['old_ticker'] == each_ticker) & (
                                ticker_change_df['effective_date'] >= each_update_date))[0][0]
                    ticker[each] = ticker_change_df['new_ticker'].values[index]
                    update_date[each] = ticker_change_df['effective_date'].values[index]
                except:
                    # if couldn't get index, it means finished update
                    finished_update[each] = True

        # change country code back to 'GY'
        ticker_updated = ticker.str.replace(' GR', ' GY')

        self.df['ticker_updated'] = ticker_updated

        num_ticker_changed = (self.df['ticker'] != self.df['ticker_updated']).sum()
        if num_ticker_changed > 0:
            print(f'{num_ticker_changed} ticker(s) has been updated based on ticker_symbol_change.csv')

    def _add_entry_date_col(self):
        """
        This function adds `entry_date` column to the df (based on `ticker`, `date_local` and `release_period`).
        
        ---------------------------------------------------------------------
        
        - If the report is released before market opens, entry_date will be the same day.
        
        - If the report is released after market opens (or on weekend/holiday), entry_date will be the next trading day.
        """
        f = open(f'{PLATFORM_PATH}/library/data/ticker_to_trading_days.json')
        data = json.load(f)
        f.close()

        try:
            ticker = self.df['ticker_updated'] + ' Equity'
        except:
            ticker = self.df['ticker'] + ' Equity'

        date_local = self.df['date_local']
        release_period = self.df['release_period']

        entry_date = []
        ticker_not_in_lst = []
        entry_date_not_found = []

        for i in range(len(ticker)):
            try:
                date_lst = data[ticker[i]]
                publish_date = str(date_local[i])

                try:
                    # publish date is earlier than the trading days list (i.e. need to update json with more years)
                    if (pd.Series(date_lst)[0] > publish_date):
                        entry_date_not_found.append([ticker[i], str(date_local[i])])
                        entry_date.append(np.nan)
                    # publish before market hours
                    elif (release_period[i] == 'Before'):
                        index = np.where(pd.Series(date_lst) >= publish_date)[0][0]
                        entry_date.append(date_lst[index])
                    # publish within/after market hours
                    else:
                        index = np.where(pd.Series(date_lst) > publish_date)[0][0]
                        entry_date.append(date_lst[index])
                except:
                    entry_date_not_found.append([ticker[i], str(date_local[i])])
                    entry_date.append(np.nan)

            except:
                ticker_not_in_lst.append(ticker[i])
                entry_date.append(np.nan)

        ticker_not_in_lst = pd.Series(ticker_not_in_lst).unique()
        ticker_not_in_lst.sort()

        self.df['entry_date'] = pd.to_datetime(pd.Series(entry_date)).dt.date

        if len(ticker_not_in_lst) > 0:
            print(f'Tickers {ticker_not_in_lst} are not in the trading days json file.')
        if len(entry_date_not_found) > 0:
            print(f'Tickers {entry_date_not_found} are in the trading days json file, but cannot return an entry date.')

    def _mkt_data_json_to_df(self, path):
        """
        This function converts and return the json file (downloaded from BQuant) into a df.
        """
        f = open(path)
        data = json.load(f)
        f.close()

        # convert first layer data to df columns
        df = pd.DataFrame(data)

        # convert second layer data (i.e. dates and prices)
        price_column = df['Data']
        row_lst = []
        for each in price_column:
            row = {}
            try:
                for each_day in each:
                    prices = each[each_day]
                    for price in prices:
                        row[f'{each_day.lower()}_{price.lower()}'] = prices[price]
            except:
                pass
            row_lst.append(row)
        df_price = pd.DataFrame(row_lst)

        # concat first and second layer data
        df.drop(columns=['Data'], inplace=True)
        df = pd.concat([df, df_price], axis=1).reindex(df.index)

        # change column name for previous n-day data
        change_col_name = {}
        for column in df.columns:
            if 'd-' in column:
                temp = column[2:]
                change_col_name[column] = 'prev' + temp
        df.rename(columns=change_col_name, inplace=True)

        df['ticker'] = df['ticker'].str.replace(' Equity', '')
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.rename(columns={'ticker': 'ticker_updated', 'date': 'entry_date'}, inplace=True)

        return df

    def _drop_entries_without_necessary_fields(self):
        """
        This function drops the entries if:
        
        1. `atr` or `d0_open` is missing
        
        2. `round(atr, 10)` = 0    (may have floating point error for 0)
        """
        num_entries = len(self.df)

        self.df.dropna(subset=['atr', 'd0_open', 'volume_d_10_sma'], inplace=True)
        self.df = self.df[round(self.df['atr'], 10) != 0]
        self.df.reset_index(drop=True, inplace=True)

        if num_entries > len(self.df):
            print(f'Dropped {num_entries - len(self.df)} entries without necessary fields (ATR or D0 Data).')

    def _add_market_cap_grp_col(self):
        """
        This function adds `market_cap_grp` column to the df (based on `market_cap_usd`).
        
        ---------------------------------------------------------------------
        
        Example:
        
        - 2,326,907,290 -> 'Mid-Cap'
        """
        market_cap_usd = self.df['market_cap_usd']

        conditions = [
            (market_cap_usd >= 200000000000),  # Mega-Cap  200B
            (market_cap_usd >= 10000000000),  # Large-Cap 10B
            (market_cap_usd >= 2000000000),  # Mid-Cap   2B
            (market_cap_usd >= 300000000),  # Small-Cap 300M
            (market_cap_usd >= 50000000),  # Micro-Cap 50M
            (market_cap_usd < 50000000)  # Nano-Cap
        ]

        choices = ['Mega-Cap', 'Large-Cap', 'Mid-Cap', 'Small-Cap', 'Micro-Cap', 'Nano-Cap']
        self.df['market_cap_grp'] = np.select(conditions, choices, default=np.nan)

    def _add_fx_rate_col(self):
        """
        This function adds `fx_rate` column to the df (based on `currency`, `entry_date`).
        
        ---------------------------------------------------------------------
        
        Example:
        
        - ('EUR', '2021-01-01') -> 1.2216
        
        The fx_rate will be the 'PX_OPEN' for EUR/USD on 2021-01-01.
        """
        f = open(f'{PLATFORM_PATH}/library/data/fx_rates.json')
        data = json.load(f)
        f.close()

        flatten_fx_rate = flatten_json(data)
        currency_entry_date = self.df['currency'] + '_' + self.df['entry_date'].astype(str)

        fx_rate = currency_entry_date.map(flatten_fx_rate)
        fx_rate = np.where(self.df['currency'] == 'USD', 1, fx_rate)
        self.df['fx_rate'] = fx_rate

        num_null = self.df['fx_rate'].isnull().sum()
        if (num_null) > 0:
            print(f'Cannot get FX Rates for {num_null} entries.')

    # ============= to do =============     
    def _add_prev_and_next_earnings_date_col(self):
        return

    def _add_trade_direction_col(self, rule=0):
        """
        This function add `side` column to the df.
        
        ---------------------------------------------------------------------
        
        Rule 0: based on Sentiments
        
        ---------------------------------------------------------------------
        
        Rule 1: different report types will have different long/short conditions
        
        - Initiation, Reinstatement: based on rating_curr
        
        - Rating Change: based on Rating upgrade/downgrade
        
        - Target Price Change: based on TP increase/decrease
        
        - Estimate Change: based on Estimates increase/decrease
        
        - Other: based on Sentiments
        
        """
        if rule in (0, 'based_on_sentiments'):

            headline_senti = self.df['headline_senti']
            summary_senti = self.df['summary_senti']

            conditions = [
                (headline_senti == 'positive') & (summary_senti == 'positive'),
                (headline_senti == 'negative') & (summary_senti == 'negative'),
            ]

        elif rule in (1, 'based_on_report_types'):

            report_type = self.df['report_type']
            rating_curr = self.df['rating_curr'].str.lower()
            rating_chg = self.df['rating_chg']
            tp_chg_pct = self.df['tp_chg_pct']
            estimate_chg_0 = self.df['y0e_eps_chg']
            estimate_chg_1 = self.df['y1e_eps_chg']
            estimate_chg_2 = self.df['y2e_eps_chg']
            headline_senti = self.df['headline_senti']
            summary_senti = self.df['summary_senti']

            positive_rating = ['strong buy', 'buy', 'outperform', 'overweight', 'add']
            negative_rating = ['strong sell', 'sell', 'underperform', 'underweight', 'reduce']

            IO_RI_long = (report_type.isin(['Initiation', 'Reinstatement'])) & (rating_curr.isin(positive_rating))
            IO_RI_short = (report_type.isin(['Initiation', 'Reinstatement'])) & (rating_curr.isin(negative_rating))

            RC_long = (report_type == 'Rating Change') & (rating_chg > 0)
            RC_short = (report_type == 'Rating Change') & (rating_chg < 0)

            TP_long = (report_type == 'Target Price Change') & (tp_chg_pct > 0)
            TP_short = (report_type == 'Target Price Change') & (tp_chg_pct < 0)

            EC_long = (report_type == 'Estimate Change') & (estimate_chg_0 > 0) & (estimate_chg_1 > 0) & (
                        estimate_chg_2 > 0)
            EC_short = (report_type == 'Estimate Change') & (estimate_chg_0 < 0) & (estimate_chg_1 < 0) & (
                        estimate_chg_2 < 0)

            other_long = (report_type.isin(
                ['Earnings Preview', 'Earnings Review', 'Company Update', 'Government Policy'])) & (
                                     headline_senti == 'positive') & (summary_senti == 'positive')
            other_short = (report_type.isin(
                ['Earnings Preview', 'Earnings Review', 'Company Update', 'Government Policy'])) & (
                                      headline_senti == 'negative') & (summary_senti == 'negative')

            conditions = [
                IO_RI_long | RC_long | TP_long | EC_long | other_long,
                IO_RI_short | RC_short | TP_short | EC_short | other_short,
            ]

        choices = ['long', 'short']
        self.df['side'] = np.select(conditions, choices, default='no action')

    def _add_entry_price_col(self, order_type='market', limit=None):
        """
        This function adds `entry_price` column (based on `side`, `prev1_close` and `atr`).
        
        - type: 'market' or 'limit'
        
        - limit (for long): limit price set at prev1_close + limit * ATR
        
        - limit (for short):  limit price set at prev1_close - limit * ATR
        
        Note:
        
        1. limit can be a number (same limit for all trades) or a column (different limit for different trades)
        
        2. limit is based on `atr` (not `atr_used`)
        
        3. Assume the trade will only enter on day-0, cancel if couldn't enter on the first day.
        """
        if order_type == 'market':
            entry_price = self.df['d0_open']

        elif order_type == 'limit':
            side = self.df['side']
            prev1_close = self.df['prev1_close']
            atr = self.df['atr']

            # round to 8 decimal point to avoid floating point error
            limit_for_long = round(prev1_close + limit * atr, 8)
            limit_for_short = round(prev1_close - limit * atr, 8)

            conditions = [
                (((side == 'long') & (self.df['d0_open'] <= limit_for_long)) | (
                            (side == 'short') & (self.df['d0_open'] >= limit_for_short))),
                ((side == 'long') & (self.df['d0_low'] <= limit_for_long)),
                ((side == 'short') & (self.df['d0_high'] >= limit_for_short)),
            ]

            choices = [
                self.df['d0_open'],  # gap less than the limit
                limit_for_long,  # gap up more than the limit, but drop below the limit price within the first day
                limit_for_short,  # gap down more than the limit, but rise above the limit price within the first day
            ]

            self.df['entry_price'] = np.select(conditions, choices, default=np.nan)

        else:
            print(f'Invalid order type. Only market order and limit order are available.')

    def _add_r_columns_with_stop_loss(self, days=5, entry='d0_open', atrx=1):
        """
        This function calculates and adds `r` columns (with stop-loss).
        
        - days: number of Rs to calculate (days=3 means calculate R for day 1,2,3)
        
        - entry: column name for entry price ('d0_open' means trades are enter on day-0, 'entry_price' means the entry price is given)
        
        - atrx: ATR(x) for calculating Rs
        """
        df = self.df.copy()

        df = df[df['side'].isin(['long', 'short'])]
        df.reset_index(drop=True, inplace=True)

        no_action_count = len(self.df) - len(df)
        if no_action_count > 0:
            print(f'Dropped {no_action_count} entries with \'no trade direction\'.')

        df['entry_price'] = df[entry]
        df['atrx'] = atrx
        df['atr_used'] = df['atr'] * df['atrx']

        entry_price = df['entry_price']
        atr_used = df['atr_used']
        side = df['side']

        stopped_or_no_data = False
        exit_price_at_stop = None

        for n in range(days + 1):

            day_n_open = df[f'd{n}_open']
            day_n_high = df[f'd{n}_high']
            day_n_low = df[f'd{n}_low']
            day_n_close = df[f'd{n}_close']

            # if the security was delisted or expired, day-n data will be missing
            # assume close at the previous day, if this case happens
            try:
                day_n_minus_1_close = df[f'd{n - 1}_close']
            except:
                day_n_minus_1_close = np.nan

            # calculate r from day-n open to entry price
            # check whether the trade will stop at open (because of gap up/down)
            # round to 8 decimal point to avoid floating point error
            r_day_n_open_for_long = round((day_n_open - entry_price) / atr_used, 8)
            stopped_at_day_n_open = ((r_day_n_open_for_long <= -1) & (side == 'long')) | (
                        (-r_day_n_open_for_long <= -1) & (side == 'short'))

            # calculate r from day-n high/low to entry price
            # check whether the trade stopped in the day
            # round to 8 decimal point to avoid floating point error
            r_day_n_high_for_long = round((day_n_high - entry_price) / atr_used, 8)
            r_day_n_low_for_long = round((day_n_low - entry_price) / atr_used, 8)
            stopped_at_day_n_middle = ((r_day_n_low_for_long <= -1) & (side == 'long')) | (
                        (-r_day_n_high_for_long <= -1) & (side == 'short'))

            conditions = [
                stopped_or_no_data,  # stopped before day-n
                day_n_open.isnull(),  # no data for day-n (eg. de-list / contract expire)
                stopped_at_day_n_open,  # stopped at day-n Open
                stopped_at_day_n_middle,  # stopped at day-n Middle
            ]

            # exit price for the above conditions
            corrresponding_exit_price_day_n = [
                exit_price_at_stop,
                day_n_minus_1_close,
                day_n_open,
                np.where(side == 'long', entry_price - atr_used, entry_price + atr_used),
            ]

            # exit price for day-n
            exit_price_day_n = np.select(conditions, corrresponding_exit_price_day_n, default=day_n_close)

            # calculate day-n R
            r_exit_for_long = (exit_price_day_n - entry_price) / atr_used
            df[f'd{n}_r'] = np.where(side == 'long', r_exit_for_long, -r_exit_for_long)

            # update status(stopped or not) and exit price(if stopped)
            stopped_or_no_data = stopped_or_no_data | day_n_open.isnull() | stopped_at_day_n_open | stopped_at_day_n_middle
            exit_price_at_stop = np.where(stopped_or_no_data, exit_price_day_n, np.nan)

        self.df = df

    def _add_r_columns_without_stop_loss(self, days=5, entry='d0_open', atrx=1):
        """
        This function calculates and adds `r_ex_sl` columns (without stop-loss).
        
        - days: number of Rs to calculate (days=3 means calculate R for day 1,2,3)
        
        - entry: column name for entry price ('d0_open' means trades are enter on day-0, 'entry_price' means the entry price is given)
        
        - atrx: ATR(x) for calculating Rs
        """
        df = self.df.copy()

        df = df[df['side'].isin(['long', 'short'])]
        df.reset_index(drop=True, inplace=True)

        no_action_count = len(self.df) - len(df)
        if no_action_count > 0:
            print(f'Dropped {no_action_count} entries with \'no trade direction\'.')

        df['entry_price'] = df[entry]
        df['atrx'] = atrx
        df['atr_used'] = df['atr'] * df['atrx']

        entry_price = df['entry_price']
        atr_used = df['atr_used']
        side = df['side']

        closed = False
        exit_price = None

        for n in range(days + 1):

            day_n_open = df[f'd{n}_open']
            day_n_close = df[f'd{n}_close']

            # if the security was delisted or expired, day-n data will be missing
            # assume close at the previous day, if this case happens
            try:
                day_n_minus_1_close = df[f'd{n - 1}_close']
            except:
                day_n_minus_1_close = np.nan

            conditions = [
                closed,  # closed before day-n (becoz of no data)
                day_n_open.isnull(),  # no data for day-n (eg. de-list / contract expire)
            ]

            # exit price for the above conditions
            corrresponding_exit_price_day_n = [
                exit_price,
                day_n_minus_1_close,
            ]

            # exit price for day-n
            exit_price_day_n = np.select(conditions, corrresponding_exit_price_day_n, default=day_n_close)

            # calculate day-n R
            r_exit_for_long = (exit_price_day_n - entry_price) / atr_used
            df[f'd{n}_r_ex_sl'] = np.where(side == 'long', r_exit_for_long, -r_exit_for_long)

            # update status(closed or not) and exit price(if closed)
            closed = closed | day_n_open.isnull()
            exit_price = np.where(closed, exit_price_day_n, np.nan)

        self.df = df

    def _add_max_risk_col(self, atrx):
        """
        This function adds `max_risk` column to the df (based on `volume_d_10_sma`, `atr` and `fx_rate`).
        
        ---------------------------------------------------------------------
        
        For every trade, Position Size <= Volume(SMA,10) * 0.02 (while Position Size = Risk / ATR Used)
        
        Therefore,
        
        - Risk(in local currency) / ATR Used <= Volume(SMA,10) * 0.02
        
        - Risk(in USD) / FX Rate / ATR Used <= Volume(SMA,10) * 0.02
        
        - Risk(in USD) <= Volume(SMA,10) * 0.02 * FX Rate * ATR Used
        """
        self.df['max_risk'] = self.df['volume_d_10_sma'] * 0.02 * self.df['fx_rate'] * self.df['atr'] * atrx

    def _add_gap_in_atr_col(self):
        """
        This function adds `gap_in_atr` column to the df (based on `d0_open`, `prev1_close` and `atr`).
        
        ---------------------------------------------------------------------
        
        Example:
        
        - ('prev1_close' = 376, 'd0_open' = 383.5, 'atr' = 16.32) -> 'gap_in_atr' = 0.45955
        """
        self.df['gap_in_atr'] = (self.df['d0_open'] - self.df['prev1_close']) / self.df['atr']

    def _select_trades_with_portfolio_constraint(self, df, sort_by='d0_exp', max_trade=6, max_skew=2):
        """
        This function selects trades with portfolio constraints (based on a df with 'side' and score).
        
        - df: DataFrame with a list of possible trades
        
        - sort_by: 'score' column, used to rank the trades
        
        - max_trades: maximum trades from this DataFrame
        
        - max_skew: maximum skewness (i.e. net long/short position)
        """
        df = df[df['screening'] == ''].copy()

        exp_cols = [f'd{i}_exp' for i in range(11) if f'd{i}_exp' in df.columns]
        df['max_exp'] = df[exp_cols].max(axis=1)

        df.sort_values(by=sort_by, ascending=False, inplace=True)

        if len(df) == 0:
            return df

        long_queue = df[df['side'] == 'long']
        short_queue = df[df['side'] == 'short']

        pending_trades = pd.DataFrame()
        min_length = min(len(long_queue), len(short_queue))

        if max_skew == 0:
            min_length_for_balance = int(min(min_length, math.floor(max_trade / 2)))
            pending_trades = pd.concat(
                [long_queue.iloc[:min_length_for_balance], short_queue.iloc[:min_length_for_balance]])

        elif max_skew > 0:
            for i in range(min_length + max_skew):
                if i < len(long_queue): pending_trades = pd.concat([pending_trades, long_queue.iloc[[i]]])
                if i < len(short_queue): pending_trades = pd.concat([pending_trades, short_queue.iloc[[i]]])
                trades_skew = abs(np.where(pending_trades['side'] == 'long', 1, -1).sum())
                if trades_skew == max_skew:
                    break
                if len(pending_trades) >= max_trade:
                    break

        return pending_trades

    def _add_fee_in_r_col(self, estimate=True, atrx=1):
        """
        This function calculates and adds `fee_in_r` column (based on `exch_location`, `atr`, `entry_price`, `exit_price`).
        
        - estimate: True (use 'prev1_close' to estimate the fee) or False (use 'entry_price' and 'exit_price' to calculate fees)
        
        - atrx: ATR(x) for calculating fees
        
        ---------------------------------------------------------------------
        
        For Canada and US, fees are based on number of shares.
        
        - Fee(in local currency) = Rate * Position Size
        
        - Fee(in R) = Fee(in local currency) / Risk(in local currency)
        
        - Fee(in R) = Rate * (Risk(in local currency) / ATR Used) / Risk(in local currency)
                  
        - Fee(in R) = Rate / ATR Used)
                  
        ---------------------------------------------------------------------
        
        For other places, fees are based on the market value of the trade (with a different rate).
        
        - Fee(in local currency) = Rate * 0.0001 * Position Size * (Entry Price + Exit Price)
        
        - Fee(in R) = Fee(in local currency) / Risk(in local currency)
        
        - Fee(in R) = Rate * 0.0001 * (Risk(in local currency) / ATR Used) * (Entry Price + Exit Price) / Risk(in local currency)
                  
        - Fee(in R) = Rate * 0.0001 / ATR Used * (Entry Price + Exit Price)
        """
        exch_location = self.df['exch_location']
        atr = self.df['atr']

        if estimate == True:
            entry_price = self.df['prev1_close']
            exit_price = self.df['prev1_close']
        else:
            entry_price = self.df['entry_price']
            exit_price = self.df['exit_price']

        fee_rate = exch_location.map(location_to_fee_rate)
        fee_based_on_num_of_shares = fee_rate / atr / atrx
        fee_based_on_market_value = fee_rate * 0.0001 / atr / atrx * (entry_price + exit_price)
        fee_in_r = np.where(exch_location.isin(['Canada', 'United States']), fee_based_on_num_of_shares,
                            fee_based_on_market_value)

        self.df['fee_in_r'] = fee_in_r

    # ============= to do =============
    def _plot_heatmap_atrx_vs_holding_period(self, original=False):
        return

    # ============= to do =============
    def _plot_r_distribution(self, curve=False):
        return

    # ============= to do =============
    def _plot_heatmap_with_two_factors(self, param='Expectancy'):
        return

    def filter_data(self, df, filters):
        filtered_df = df.copy()

        for each in filters:
            column_name = each
            list = filters[each]
            if list != []:
                filtered_df = filtered_df[filtered_df[column_name].isin(list)]

        filtered_df.reset_index(drop=True, inplace=True)
        return filtered_df

    # ============= to do =============
    def bin_data_by_edge(self, data, binwidth, bins):
        """
        This function bins data (make the numbers to be discret) and return the 'interval' or 'bins name'.
        
        Example:
        
        data = []
        """
        # output='interval/class_mark(i.e.middle of interval'
        # return None if element is None

    def clean_bofa_raw(self):
        raw_df = self.df.copy()

        raw_df.rename(columns={'Product Id': 'report_id',
                               'Publish Date': 'publish_date_and_time',
                               'Special Symbol(Distr. Director)': 'ticker',
                               'Ticker': 'ticker_broker',
                               'Issuer': 'company_name',
                               'Display Industry': 'industry_broker',
                               'Primary Author': 'analyst_pri',
                               'Searchable Ref Analysts': 'analyst_all',
                               'Headline': 'headline',
                               'Key takeaways': 'summary',
                               'Rating': 'rating_curr',
                               'Subject': 'report_type_broker',
                               'Product Category': 'product_category',
                               'External Url': 'url'}, inplace=True)

        raw_df = raw_df[raw_df['ticker_broker'].notna()]
        raw_df['report_id'] = raw_df['report_id'].astype(int)

        filters = {
            'report_type_broker': ['Company Update', 'Coverage Resumed', 'Earnings Preview', 'Earnings Review',
                                   'Estimate Change', 'Government Regulations', 'Initial Opinion',
                                   'Price Objective Change', 'Rating Change', 'Reinstatement of Coverage'],
            'product_category': ['Breaking News', 'Report', 'Short Report'],
        }
        raw_df = self.filter_data(raw_df, filters)

        # add uid
        raw_df.insert(0, 'uid', 'bofa_' + raw_df['report_id'].astype(int).astype(str))

        # merge all analysts names
        analyst_searchable = np.where(raw_df['analyst_all'].isnull(), '', ';' + raw_df['analyst_all'])
        analyst_secondary = np.where(raw_df['Secondary Analyst'].isnull(), '', ';' + raw_df['Secondary Analyst'])
        raw_df['analyst_all'] = raw_df['analyst_pri'] + analyst_searchable + analyst_secondary
        raw_df.drop(columns=['Secondary Analyst'], inplace=True)

        # clean ticker column
        raw_df['ticker'] = (raw_df['ticker'].str.split(';', 1, expand=True))[0]  # get value before ';'

        # add security class
        raw_df['security_class'] = 'Equity'

        # split publish date into date(HK) and time (HKT)
        raw_df[['date_hkt', 'time_hkt']] = raw_df['publish_date_and_time'].str.split(' ', 1, expand=True)
        raw_df['date_hkt'] = pd.to_datetime(raw_df['date_hkt']).dt.date
        raw_df['time_hkt'] = pd.to_datetime(raw_df['time_hkt']).dt.time

        self.df = raw_df

    def clean(self, tz_name='Asia/Hong_Kong', mode='stable'):
        # self._drop_duplicate_entries()
        self._clean_ticker(mode=mode)  # BBG
        self._add_exch_location_col(mode=mode)
        self._add_exch_region_col()  # Continent
        self._drop_entries_with_non_tradable_location()  # Need input countries
        self._add_local_date_and_time_and_release_period_col(tz_name, mode=mode)
        self._add_updated_ticker_col()
        # self._add_entry_date_col()

    def add_market_data(self, path):
        data_mkt = self._mkt_data_json_to_df(path)
        self.df = pd.merge(self.df, data_mkt, on=['report_id', 'ticker_updated', 'entry_date'], how='left')
        self._drop_entries_without_necessary_fields()
        self._add_market_cap_grp_col()
        self._add_fx_rate_col()
        self._add_prev_and_next_earnings_date_col()

    def backtest(self):
        # self._add_max_risk_col(atrx=1)
        # self._add_gap_in_atr_col()
        # self._add_trade_direction_col(rule=0)
        # self._add_entry_price_col(order_type='limit', limit=1)
        # self._add_r_columns_with_stop_loss(days=5, entry='d0_open', atrx=1)
        # self._add_r_columns_without_stop_loss(days=5, entry='d0_open', atrx=1)
        # self._select_trades_with_portfolio_constraint(self.df, max_trade=6, max_skew=1)
        self._add_fee_in_r_col(estimate=True, atrx=1)

    def visualize(self):
        self._plot_heatmap_atrx_vs_holding_period(original=False)
        self._plot_r_distribution(curve=False)
        self._plot_heatmap_with_two_factors(param='Expectancy')


if __name__ == '__main__':
    try:

        # ===================================================================
        # =============================  clean  =============================
        # ===================================================================

        df = pd.read_csv(f'{PLATFORM_PATH}/library/data/bofa_test.csv')
        data = Dataset(df)

        # for cleaning broker raw file
        data.clean_bofa_raw()

        # for BofA testing
        # df = df[df['ticker_broker'].notna()]
        # df.reset_index(drop=True, inplace=True)

        # df['ticker'] = (df['ticker'].str.split(";", 1, expand=True))[0]
        # temp = df['ticker'].str.split(" ", 1, expand=True)[1]

        # for cleaning, input tz_name as the timezone of publish_data_and_time
        # list of tz_name: https://gist.github.com/pamelafox/986163

        data = Dataset(df)

        data.clean(tz_name='Asia/Hong_Kong')
        # data.add_market_data()

        data.clean(tz_name='Asia/Hong_Kong')
        data.add_market_data()

        # ===================================================================
        # ========================  get market data  ========================
        # ===================================================================

        data.add_market_data(path=os.path.join(BASE_PATH, 'Goldman/library/data/bofa_mkt_data.json'))

        # ===================================================================
        # ===========================  backtest  ============================
        # ===================================================================

        data.backtest()

        print(data.df)

    except:
        print(traceback.format_exc())
