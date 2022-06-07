def predict_model_test_data(self, strategy, model, neutral_score=0):
    class_expected_return = {0: -1, 1: neutral_score, 2: 1}
    # class_expected_return = {0: -1, 1: 1}

    test_data = self.get_model_test_data(strategy)
    y_pred, y_pred_prob = ML.predict_test_data(model, test_data[x_columns])

    prediction = pd.DataFrame(y_pred, columns=['Prediction'])
    prediction_proba = y_pred_prob.max(axis=1)
    prediction = np.multiply(np.array(prediction.replace(class_expected_return)['Prediction']), prediction_proba)

    test_data['Expectancy'] = prediction
    test_data['Side'] = test_data['Expectancy'].copy()
    test_data['Side'] = pd.cut(test_data['Expectancy'],
                               bins=[-np.inf, -0.000001, 0.000001, np.inf],
                               labels=['negative', 'neutral', 'positive'])
    test_data['Expectancy'] = abs(test_data['Expectancy'])
    test_data['Prediction'] = prediction

    test_data = test_data[~test_data['Region'].isin(['Africa', 'Oceania'])].reset_index(drop=True)

    return test_data