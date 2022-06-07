import numpy as np
import pandas as pd
from Model.settings import DataCleaner
from Backtest import BacktestEngine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from uti import DataLoader, Logger

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path
DC = DataCleaner()
Engine = BacktestEngine()

class LR:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.mlr = LinearRegression()
        self.mlr2 = sm.OLS(self.y, sm.add_constant(self.X))

    def train(self):
        self.mlr.fit(self.X, self.y)
        print('Intercept:', self.mlr.intercept_)
        print('Coefficients:', self.mlr.coef_)

        self.mlr2 = self.mlr2.fit()


    def evaluate(self):
        prediction = self.mlr.predict(self.X)
        r2 = r2_score(self.y, prediction)
        mse = mean_squared_error(self.y, prediction)
        rmse = np.sqrt(mse)

        print('Score: ', self.mlr.score(self.X, self.y))
        print('R2 score: ', r2)
        print('MSE score: ', mse)
        print('RMSE score: ', rmse)

        print(self.mlr2.summary())

    def get_params(self):
        return self.mlr.intercept_, self.mlr.coef_


class RLR:
    def __init__(self, X, y, win):
        self.X = X
        self.y = y
        self.X = sm.add_constant(self.X)
        self.win = win
        self.model = RollingOLS(self.y, self.X, window=win)

    def train(self):
        self.rlr = self.model.fit()
        # print('Intercept:', self.rlr.intercept_)
        # print('Coefficients:', self.rlr.coef_)
        self.prediction = np.sum(self.X * self.rlr.params, axis=1)  # self.rlr.predict(self.X)

    def evaluate(self):
        mse = mean_squared_error(self.y, self.prediction)

        print('R2 score: ', r2_score(self.y, self.prediction))
        print('MSE score: ', mse)
        print('RMSE score: ', np.sqrt(mse))

    def get_param(self, rebalance_win=60):
        param = self.rlr.params
        param_to_keep_index = param.index[::rebalance_win]
        param[~param.index.isin(param_to_keep_index)] = np.nan
        param = param.fillna(method='ffill')
        return param



if __name__ == '__main__':
    strategy = f'Benchmark strategy (scoring system after fees)'
    data = DL.loadBT(strategy)

    model = LR(data[['Expectancy1', 'Expectancy2', 'Expectancy3']], data[['d0_r']])
    model.train()
    model.evaluate()
