"""Bagging"""
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    df = pd.read_csv('./data/heart.csv')
    print(df['target'].describe())

    X = df.drop(['target'], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    y_pred_boost = boost.predict(X_test)

    print(f"Accuracy Gradient Boosting: {accuracy_score(y_pred_boost, y_test)}")
    


