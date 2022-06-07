from uti import DataLoader, Logger
from Model.settings import DataCleaner
from Backtest import backtest_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

logger = Logger()
DL = DataLoader()
DATABASE_PATH = DL.database_path
DC = DataCleaner()
Engine = backtest_engine()


import numpy as np
import pandas as pd

from uti import DataLoader, Logger

logger = Logger()
DL = DataLoader()

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

if __name__ == '__main__':
    strategy = f'Benchmark strategy (scoring system after fees)'
    data = DL.loadBT(strategy)

    model = LR(data[['Expectancy1', 'Expectancy2', 'Expectancy3']], data[['d0_r']])
    model.train()
    model.evaluate()
