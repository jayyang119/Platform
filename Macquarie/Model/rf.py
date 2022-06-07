import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from uti import DataLoader, Logger
from Model.ML import ML
from Model.settings import (X_train, region_cat, sector_cat, senti_cat, analyst_cat, market_cat, mc_cat, report_type_cat)
from Backtest import backtest_engine

logger = Logger()
DL = DataLoader()



def random_forest():
    categorical_features = ['Head analyst',
                            'Market', 'Region', 'Sector', 'MarketCap',
                            'Headline sentiment',
                            'Summary sentiment',
                            'Earnings', 'Initiate', 'Rating', 'Estimates', 'Guidance', 'Review',
                            ]

    column_transformer = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore',
                       categories=[
                           analyst_cat,
                           market_cat, region_cat, sector_cat, mc_cat,
                           senti_cat,
                           senti_cat,
                           report_type_cat, report_type_cat, report_type_cat,
                           report_type_cat, report_type_cat, report_type_cat,
                       ]),
         categorical_features),
        remainder="passthrough")

    # Base model
    rf_model = make_pipeline(column_transformer,
                             RandomForestClassifier(oob_score=True, bootstrap=True,
                                                    random_state=9988))
    rf_model.fit(X_train, y_train)
    print(rf_model.steps[1][1].oob_score_)
    print(confusion_matrix(y_train, rf_model.predict(X_train)))

    # Grid search
    param_test1 = [{
        'clf__n_estimators': range(40, 161, 20),
        # 'clf__max_depth': range(3, 19, 3),
        # 'clf__min_samples_split': range(50, 201, 25),
        # 'clf__min_samples_leaf': range(10, 61, 10)
    }]
    pipeline = Pipeline([('preprocess', column_transformer),
                         ('clf', RandomForestClassifier(
                             # n_estimators=140,
                             # min_samples_split=150,
                             # min_samples_leaf=20,
                             # max_depth=14,
                             max_features='sqrt',
                             random_state=10,
                             oob_score=True,
                             bootstrap=True
                         ))])
    gsearch1 = GridSearchCV(estimator=pipeline, param_grid=param_test1,
                            scoring='roc_auc_ovr', refit=True)
    gsearch1.fit(X_train, y_train)
    # ML.plot_convergence(gsearch1)
    print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print('Oob score: ', gsearch1.best_estimator_.steps[1][1].oob_score_)
    print('Confusion matrix: \n', confusion_matrix(y_train, gsearch1.predict(X_train)))
    print('Classification report: \n', classification_report(y_train, gsearch1.predict(X_train)))
    ML.plot_convergence(gsearch1)
    ML.evaluate(gsearch1, test=True)  # Only print the score on training data.
    ML.plot_features(gsearch1.best_estimator_)
    ML.save_model(gsearch1, 'model2')
    # ML.plot_convergence(gsearch1, 'param_clf__max_depth')

    total_ticks = [40, 80, 120, 160, 200, 250, 300, 350, 400]
    total_score = [0.53133068, 0.53288507, 0.53338693, 0.53498576, 0.5357338846449456,
                   0.536149261639518, 0.5368253737188575, 0.5369179667329786, 0.5370889406823498]

    xlabel = 'param_clf__n_estimators'
    ticks1 = list(gsearch1.cv_results_[xlabel])
    score1 = gsearch1.cv_results_['mean_test_score']

    total_ticks.extend(ticks1)
    total_score.extend(score1)

    plt.figure(figsize=(9, 6))
    plt.plot(total_score)
    plt.xlabel(xlabel)
    plt.xticks(range(len(total_ticks)), total_ticks)
    plt.title('ROC AUC Score (5-fold CV)')
    plt.show()

    neutral_score = 0
    daily_position = 18
    region_position = 6
    min_region_position = 4

    data_package = DataCleaner()
    backtest_package = backtest_engine(daily_position=daily_position, region_position=region_position,
                                       skew=0, min_region_position=min_region_position)

    for neutral_score in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        # Benchmark
        # strats = ['Headline strategy', f'Benchmark (daily_position {daily_position} region_position {region_position})']
        # backtest(strats=strats, regenerate=True, daily_position=daily_position,
        #                    region_position=region_position)
        # visual(f'Benchmark (daily_position {daily_position} region_position {region_position})')

        # Model
        strategy = f'Daily {daily_position} Region {region_position} Region_min {min_region_position} Neutral score{neutral_score}'
        test_data = data_package.predict_model_test_data(strategy=strategy, model=gsearch1,
                                                         neutral_score=neutral_score)

        # backtest_package.set_parameters()
        results_df = backtest_package.backtest_job(test_data)

        results_df = backtest_package.backtest_job(test_data)
        DL.toDB(results_df, f'Backtest/{strategy}.csv')
        backtest_package.visual(strategy)

