import pandas as pd
import numpy as np
import math
from random import sample
import matplotlib.pyplot as plt
from uti import DataLoader, Logger

def select_trades_with_portfolio_constraint(df, sort_by="d0_exp", max_trade=6, max_skew=2
                                             ):
    """
        This function selects trades with portfolio constraints (based on a df with 'side' and score).
        
        - df: DataFrame with a list of possible trades
        
        - sort_by: 'score' column, used to rank the trades
        
        - max_trades: maximum trades from this DataFrame
        
        - max_skew: maximum skewness (i.e. net long/short position)
        """
    try:
        df = df[df["screening"] == ""].copy()
    except:
        pass

    df.sort_values(by=sort_by, ascending=False, inplace=True)

    if len(df) == 0:
        return df

    long_queue = df[df["Side"] == "long"]
    short_queue = df[df["Side"] == "short"]

    pending_trades = pd.DataFrame()
    min_length = min(len(long_queue), len(short_queue))

    if max_skew == 0:
        min_length_for_balance = int(min(min_length, math.floor(max_trade / 2)))
        pending_trades = pd.concat(
            [
                long_queue.iloc[:min_length_for_balance],
                short_queue.iloc[:min_length_for_balance],
            ]
        )

    elif max_skew > 0:
        for i in range(min_length + max_skew):
            if i < len(long_queue):
                pending_trades = pd.concat([pending_trades, long_queue.iloc[[i]]])
            if i < len(short_queue):
                pending_trades = pd.concat([pending_trades, short_queue.iloc[[i]]])
            trades_skew = abs(np.where(pending_trades["side"] == "long", 1, -1).sum())
            if trades_skew == max_skew:
                break
            if len(pending_trades) >= max_trade:
                break

    return pending_trades


def get_trade_record(data, sort_by='score', max_trade=6, max_skew=1):
    trade_record_uid = []
    for date, data_group in data.groupby('Date'):
        selected_trades = select_trades_with_portfolio_constraint(data_group, sort_by=sort_by, max_trade=max_trade,
                                                                   max_skew=max_skew)
        trade_record_uid = trade_record_uid + list(selected_trades['uid'])

    trade_record = data[data['uid'].isin(trade_record_uid)]
    return trade_record


def scoring_calculation(training_data, broker, ticker, date, scoring_type="Broker", mktcap=None, side=None,
                        strength=None, gics1=None, required_num_observation=5):
    """
    Based on your trading system
    """
    if scoring_type == "Broker":
        num_observation = len(training_data[training_data.broker == broker])
        if num_observation > required_num_observation:
            score = training_data[training_data.broker == broker]["d0_r"].mean()
        else:
            score = np.nan
        return num_observation, score


def simulation(scoring_type_sim, dfall_5DR, number_of_test=100, required_num_observation=5):
    different_test_R = {}
    print("Simulating " + scoring_type_sim + " System.")
    for k in range(1, number_of_test + 1):
        if k % 20 == 0:
            print(f"{k} trial simulation")
        R_dict = {}
        num_trade_dict = {}
        trading_R_details = []
        testing_date = sample(set(dfall_5DR['Date']), 252)

        training_data = dfall_5DR.loc[~dfall_5DR['Date'].isin(testing_date)]
        testing_data = dfall_5DR.loc[dfall_5DR['Date'].isin(testing_date)]

        for i in testing_date:
            date = i
            date_data = dfall_5DR.loc[dfall_5DR['Date'] == date]

            specific_broker_score = list(output['Expectancy'])

            date_data.insert(0, "Score", specific_broker_score)
            sort_date_data = date_data.sort_values(by=["Score"], ascending=False)

            # filter out the data with nan or negative score
            sort_date_data_clean = sort_date_data.loc[sort_date_data.Score > 0]

            sort_date_up = sort_date_data_clean.loc[sort_date_data_clean.rating_chg == 1]
            num_of_date_up = len(sort_date_up)
            sort_date_down = sort_date_data_clean.loc[sort_date_data_clean.rating_chg == -1]
            num_of_date_down = len(sort_date_down)

            ## today trades
            params_max_net_position = 2
            params_max_one_side_position = 3
            today_trades = get_trade_record(sort_date_data_clean, sort_by='Score',
                                            max_trade=params_max_one_side_position * 2,
                                            max_skew=params_max_net_position)

            today_num_trades = len(today_trades)
            today_R = today_trades["d0_r"]
            today_total_R = today_R.sum()
            R_dict[date] = today_total_R
            num_trade_dict[date] = today_num_trades
            trading_R_details = trading_R_details + today_R.values.toist()

        R_in_this_test = pd.Series(R_dict.values())
        num_trades_in_this_test = pd.Series(num_trade_dict.values())
        different_test_R[f"{k} trial test"] = [R_in_this_test, num_trades_in_this_test, trading_R_details]

    return different_test_R


def simulation_datecleaning(sim_result):
    equity_curve, num_trades, hit_ratio, expectancy = [], [], [], []
    for i in sim_result.keys():
        R_data_details = pd.Series(sim_result[i][2])

        equity_curve.append(pd.Series(sim_result[i][0]).cumsum())
        num_trades.append(len(R_data_details))
        hit_ratio.append(sum(R_data_details > 0) / len(R_data_details))
        expectancy.append(sum(R_data_details) / len(R_data_details))
    return equity_curve, num_trades, hit_ratio, expectancy


def simulation_visualization(equity_curve, num_trades, hit_ratio, expectancy, scoring_type=""):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
    plt.grid()
    fig.patch.set_facecolor('xkcd:gray')

    for equity in equity_curve:
        axes[0][0].plot(equity.index, equity, linewidth=1)
    # axes[0][0].set_xticks(equity.index, range(len(equity)))

    pd.Series(num_trades).plot(kind="hist", bins=40, ax=axes[0][1], cmap="coolwarm_r", density=True, alpha=0.6)
    pd.Series(num_trades).plot(kind="kde", ax=axes[0][1], cmap="coolwarm_r", linewidth=2.5)

    pd.Series(hit_ratio).plot(kind="hist", bins=40, ax=axes[1][0], cmap="pink", density=True, alpha=0.6)
    pd.Series(hit_ratio).plot(kind="kde", ax=axes[1][0], cmap="pink", linewidth=2.5)

    pd.Series(expectancy).plot(kind="hist", bins=40, ax=axes[1][1], cmap="Accent", density=True, alpha=0.6)
    pd.Series(expectancy).plot(kind="kde", ax=axes[1][1], cmap="Accent", linewidth=2.5)

    axes[0][0].set_title(scoring_type + "Scoring System R Curve Simulation", fontsize=25)
    axes[0][1].set_xlabel("Number of Trades", fontsize=20)
    axes[0][1].set_title("Distribution of Number of Trades", fontsize=25)
    axes[1][0].set_xlabel("Hit Ratio", fontsize=20)
    axes[1][0].set_title("Distribution of Hit Ratio", fontsize=25)
    axes[1][1].set_xlabel("Expectancy", fontsize=20)
    axes[1][1].set_title("Distribution of Expectancy", fontsize=25)
    plt.show()

if __name__ == '__main__':
    DL = DataLoader()
    logger = Logger()
    strategy = 'Benchmark strategy (region specific rules)'
    dfall_5DR = DL.loadBT(strategy)
    logger.info(dfall_5DR)
    simulation('Broker', dfall_5DR, number_of_test=100, required_num_observation=5)