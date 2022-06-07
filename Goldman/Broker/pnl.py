import numpy as np
from datetime import datetime
from uti import Logger
from Broker.settings import EXCH

logger = Logger()


def ricsregion(df):
    """
     This function is to transform eikon region tickers to Bloomberg region tickers

     df [DataFrame] - DataFrame that contains RICs
    """
    df[['Tick', 'Region']] = df['Ticker'].str.split(".", n=1, expand=True)
    df = df.replace({'Region': EXCH})
    df['Region'] = df['Region'].replace(np.nan, 'US')
    df = df.drop(columns='Tick')

    return df


def get_tr(data, rewrite=False):
    """
     This function is to calculate EMA 10-day ATR

     data [DataFrame] - DataFrame that contains price data for specific ticker for at least 10 days
    """
    if 'trmax' not in data.columns or rewrite:
        high = np.array(data.loc[:]['HIGH'])
        low = np.array(data.loc[:]['LOW'])
        close = np.array(data.loc[:]['CLOSE'].shift().fillna(method='bfill'))
        data['tr0'] = abs(high - low)
        data['tr1'] = abs(high - close)
        data['tr2'] = abs(low - close)
        data['trmax'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)

    return data


def get_atr(data):
    if 'trmax' not in data.columns:
        print('Calculating true range...')
        get_tr(data)
    assert 'trmax' in data.columns, 'True range data unavailable, please check.'
    data['ATR'] = data['trmax'].ewm(span=10, adjust=False).mean()

    if 'Gap' not in data.columns:
        data['Gap'] = data['OPEN'] - data['CLOSE'].shift(1).fillna(method='bfill')

    # if 'gap_atr' not in data.columns:
    data['gap_atr'] = data['Gap'] / data['ATR']
    data['d0_r'] = (data['CLOSE'] - data['OPEN']) / data['ATR']

    return data


# df = DL.loadDB('Backtest/results(Headline strategy).csv')
# row = df.iloc[311]
# i = 311
def get_pnl(df):
    """
     This function is to calculate Stop Loss and Rs for day1-3 of a given strategy,

     based on df['Side']. positive: long, negative: short, neutral: do not take risk.

     df [DataFrame] - DataFrame that contains price data for 3 days
    """
    logger.info('Calculating pnl...')
    df['stop_loss'] = 0.00
    df['d0_r'] = 0.00
    df['d1_r'] = 0.00
    df['d2_r'] = 0.00

    df_columns = list(df.columns)
    id_side = df_columns.index('side')
    id_o1 = df_columns.index('d0_open')
    id_o2 = df_columns.index('d1_open')
    id_o3 = df_columns.index('d2_open')
    id_c1 = df_columns.index('d0_close')
    id_c2 = df_columns.index('d1_close')
    id_c3 = df_columns.index('d2_close')
    id_h1 = df_columns.index('d0_high')
    id_h2 = df_columns.index('d1_high')
    id_h3 = df_columns.index('d2_high')
    id_l1 = df_columns.index('d0_low')
    id_l2 = df_columns.index('d1_low')
    id_l3 = df_columns.index('d2_low')
    id_atr = df_columns.index('atr_used')
    id_sl = df_columns.index('stop_loss')
    id_r1 = df_columns.index('d0_r')
    id_r2 = df_columns.index('d1_r')
    id_r3 = df_columns.index('d2_r')

    for i, row in df.iterrows():
        ticker = row['Ticker']
        # if ticker in ['600745.SS', 'FSR', '688083.SS', 'ING.AX', '600867.SS', '002511.SZ', '688008.SS', 'CGC.AX', '603160.SS']:
        #     print(ticker)
        atr = df.iat[i, id_atr]

        try:
            #         R for longs
            if df.iat[i, id_side] in ['positive', 'long']:
                #             calculate SL
                df.iat[i, id_sl] = df.iat[i, id_o1] - atr
                #             Day 1 R for not stopped out position
                if df.iat[i, id_l1] > df.iat[i, id_sl]:
                    df.iat[i, id_r1] = (df.iat[i, id_c1] - df.iat[i, id_o1]) / atr
                    #                 check if the day 2 open price is lower than SL
                    if df.iat[i, id_o2] == 0:
                        continue
                    elif df.iat[i, id_o2] < df.iat[i, id_sl]:
                        df.iat[i, id_r2] = (df.iat[i, id_o2] - df.iat[i, id_o1]) / atr
                        df.iat[i, id_r3] = df.iat[i, id_r2]
                    #                 Day 2 R for not stopped out position
                    elif df.iat[i, id_l2] > df.iat[i, id_sl]:
                        df.iat[i, id_r2] = (df.iat[i, id_c2] - df.iat[i, id_o1]) / atr
                        #                     check if the day 3 open price is lower than SL
                        if df.iat[i, id_o3] == 0:
                            continue
                        elif df.iat[i, id_o3] < df.iat[i, id_sl]:
                            df.iat[i, id_r3] = (df.iat[i, id_o3] - df.iat[i, id_o1]) / atr
                        #                     Day 3 R for not stopped out position
                        elif df.iat[i, id_l3] > df.iat[i, id_sl]:
                            df.iat[i, id_r3] = (df.iat[i, id_c3] - df.iat[i, id_o1]) / atr
                        else:
                            df.iat[i, id_r3] = -1.00
                    else:
                        df.iat[i, id_r2] = -1.00
                        df.iat[i, id_r3] = -1.00
                else:
                    df.iat[i, id_r1] = -1.00
                    df.iat[i, id_r2] = -1.00
                    df.iat[i, id_r3] = -1.00
            elif df.iat[i, id_side] in ['negative', 'short']:
                df.iat[i, id_sl] = df.iat[i, id_o1] + atr
                if df.iat[i, id_h1] < df.iat[i, id_sl]:
                    df.iat[i, id_r1] = (df.iat[i, id_o1] - df.iat[i, id_c1]) / atr
                    if df.iat[i, id_o2] == 0:
                        continue
                    elif df.iat[i, id_o2] > df.iat[i, id_sl]:
                        df.iat[i, id_r2] = (df.iat[i, id_o1] - df.iat[i, id_o2]) / atr
                        df.iat[i, id_r3] = df.iat[i, id_r2]
                    elif df.iat[i, id_h2] < df.iat[i, id_sl]:
                        df.iat[i, id_r2] = (df.iat[i, id_o1] - df.iat[i, id_c2]) / atr
                        if df.iat[i, id_o3] == 0:
                            continue
                        elif df.iat[i, id_o3] > df.iat[i, id_sl]:
                            df.iat[i, id_r3] = (df.iat[i, id_o1] - df.iat[i, id_o3]) / atr
                        elif df.iat[i, id_h3] < df.iat[i, id_sl]:
                            df.iat[i, id_r3] = (df.iat[i, id_o1] - df.iat[i, id_c3]) / atr
                        else:
                            df.iat[i, id_r3] = -1.00
                    else:
                        df.iat[i, id_r2] = -1.00
                        df.iat[i, id_r3] = -1.00
                else:
                    df.iat[i, id_r1] = -1.00
                    df.iat[i, id_r2] = -1.00
                    df.iat[i, id_r3] = -1.00
            elif df.iat[i, id_side] == 'neutral':
                df.iat[i, id_r1] = 0
                df.iat[i, id_r2] = 0
                df.iat[i, id_r3] = 0

            if i % 1000 == 0:
                logger.info(f'Row {i} {ticker} completed.')
        except Exception as e:
            logger.error(f'Row {i} {ticker} error')
            logger.error(row)

    return df

