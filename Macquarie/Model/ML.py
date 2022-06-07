from Model.settings import (X_train, X_test, y_train, y_test)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

import numpy as np

from uti import DataLoader, Logger

logger = Logger()
DL = DataLoader()

class ML(object):
    def __init__(self):
        super(ML, self).__init__()

    @classmethod
    def evaluate_train(cls, model):
        print('Training score:', model.score(X_train, y_train))
        print('Training ROC AUC:',
              roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovr'))  # , multi_class='ovr'

    @classmethod
    def evaluate_test(cls, model):
        print('Test score:', model.score(X_test, y_test))
        print('Test ROC AUC:', roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'))
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        return y_pred, y_pred_prob

    @classmethod
    def predict_test_data(cls, model, test_data):
        y_pred = model.predict(test_data)
        y_pred_prob = model.predict_proba(test_data)
        return y_pred, y_pred_prob

    @classmethod
    def plot_features(cls, model, top=25):
        FI = FeatureImportance(model)

        columns = X_train.columns.to_list()
        numerical_features_map = {}
        for title in ['Headline', 'Summary']:
            for senti in ['neutral', 'positive', 'negative']:
                numerical_features_map[columns.index(f'{title} {senti} score')] = f'{title} {senti} score'

        importances = FI.get_feature_importance().rename(index=numerical_features_map)
        sorted_indices = list(np.argsort(importances)[::-1])[:top]
        importances = importances.iloc[sorted_indices]

        fig, ax = plt.subplots(figsize=(8, 12))
        plt.title('Feature Importance')
        y_pos = np.arange(len(importances))

        ax.barh(y_pos, importances, align='center')
        plt.yticks(y_pos, labels=importances.index)
        ax.invert_yaxis()
        plt.show()

    # Parameters testing
    # n-estimators
    @classmethod
    def plot_convergence(cls, model, xlabel='param_clf__n_estimators'):
        plt.figure(figsize=(9, 6))
        xticks = list(model.cv_results_[xlabel])
        plt.plot(model.cv_results_['mean_test_score'])
        plt.xlabel(xlabel)
        plt.xticks(range(len(xticks)), xticks)
        plt.title('ROC AUC Score (5-fold CV)')
        plt.show()

    @classmethod
    def save_model(cls, model, name='RF'):
        with open(f'{DATABASE_PATH}/Backtest/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    @classmethod
    def load_model(cls, name='RF'):
        with open(f'{DATABASE_PATH}/Backtest/{name}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

