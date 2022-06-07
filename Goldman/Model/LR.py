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
from Broker import get_pnl
from Model.ML import ML

logger = Logger()
DL = DataLoader()
ml = ML()

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



if __name__ == '__main__':
    strategy = f'Benchmark strategy (scoring system after fees)'
    train_data, test_data = DC.get_benchmark_test_data()
    X = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    train_data_index = train_data.index

    def prep_data(df):
        df['side'] = 'long'
        df['d0_r'] = (df['d0_close'] - df['d0_open']) / df['atr_used']
        df['d1_r'] = (df['d1_close'] - df['d0_open']) / df['atr_used']
        df['d2_r'] = (df['d2_close'] - df['d0_open']) / df['atr_used']
        # df['turnover'] = df['volume_d_10_sma'] / df['market_cap_usd']

    prep_data(X)
    prep_data(train_data)
    prep_data(test_data)

    x_columns = ['Headline sentiment', 'Summary sentiment', 'exch_location', 'exch_region', 'Report Type',
                 'market_cap_grp']
    X = pd.get_dummies(X[x_columns], drop_first=True)

    X_train = X.iloc[train_data_index]
    y_train = train_data[['d0_r']]
    X_test = X.iloc[train_data_index[-1]+1:]
    y_test = test_data[['d0_r']]

    model = LR(X_train, y_train)
    model.train()
    model.evaluate()

    ml.save_model(model.mlr, 'mlr')

    mlr_model = ml.load_model('mlr')
    test_data['Expectancy'] = mlr_model.predict(X_test)

    test_data_long_index = test_data[test_data['Expectancy'] >= 0].index
    test_data_short_index = test_data[test_data['Expectancy'] < 0].index

    test_data.loc[test_data_long_index, 'side'] = 'long'
    test_data.loc[test_data_short_index, 'side'] = 'short'
    test_data['Expectancy'] = test_data['Expectancy'].apply(lambda x: x if x >= 0 else -x)

    test_data = get_pnl(test_data)

    DL.toBT(test_data, 'mlr')
