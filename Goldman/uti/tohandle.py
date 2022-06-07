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
from Backtest import backtest_engine
from Backtest.simulation_functions import simulation_datecleaning, simulation_visualization

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path

new_df = np.array([])


def find_x(ini_str, sub_str, occurrence):
    val = -1
    for i in range(0, occurrence):
        val = ini_str.find(sub_str, val + 1)
    return val


with open(f'{DATABASE_PATH}/tohandle.txt', encoding='utf-8') as f:
    for line in f.readlines():
        print(line)
        line = line[line.find(' ')+1:]

        idx = line.find(' ')
        date = line[:idx]
        line = line[idx+1:]

        idx = find_x(line, "\"", 2)
        time = line[:idx].strip("\"")
        line = line[idx + 2:]

        idx = find_x(line, " ", 1)
        ticker = line[:idx]
        line = line[idx+1:]

        idx = find_x(line, " ", 1)
        source = line[:idx]
        line = line[idx+1:]

        idx = find_x(line, " ", 1)
        asset_class = line[:idx]
        line = line[idx+1:]

        idx = find_x(line, "\"", 2)
        headline = line[:idx].strip("\"")
        line = line[idx + 2:]

        idx = find_x(line, "\"", 2)
        summary = line[:idx].strip("\"")
        line = line[idx + 2:]

        idx = find_x(line, "\"", 2)
        analysts = line[:idx].strip("\"")
        line = line[idx + 2:]

        idx = find_x(line, "\"", 2)
        report_type = line[:idx].strip("\"")

        new_df = np.append(new_df, [date, time, ticker, source, asset_class, headline, summary, analysts, report_type])



new_df = new_df.reshape(-1, 9)
new_df = pd.DataFrame(new_df,
                      columns=['Date', 'Time', 'Ticker', 'Source', 'Asset Class', 'Headline',
                               'Summary', 'Analysts', 'Report Type'])
new_df = new_df.iloc[1:]

GS_raw = DL.loadDB('GS_raw.csv')

GS_raw = pd.concat([new_df, GS_raw], axis=0)

DL.toDB(GS_raw, 'GS_raw.csv')