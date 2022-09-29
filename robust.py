"""Robust regressions"""

import pandas as pd
from sklearn.linear_model import(
    RANSACRegressor, HuberRegressor
)
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.head())

    cols = ['country', 'score']
    X = dataset.drop(cols, axis=1)
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=1.0),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35),
    }

    for name, model in estimators.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("=" * 32)
        print(f"{name: <10} : {mse}")

