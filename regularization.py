"""Regularization"""
import pandas as pd
import sklearn
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    cols = ['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']
    X = dataset[cols]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)
    print("=" * 32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    linear = LinearRegression().fit(X_train, y_train)

    y_pred = linear.predict(X_test)

    lasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    
    ridge = Ridge(alpha=1).fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)


    linear_loss = mean_squared_error(y_test, y_pred)
    print(f"Linear Loss: {linear_loss}")


    lasso_loss = mean_squared_error(y_test, y_pred_lasso)
    print(f"Lasso Loss: {lasso_loss}")


    ridge_loss = mean_squared_error(y_test, y_pred_ridge)
    print(f"Ridge Loss: {ridge_loss}")

    print("="*32)

    print("LASSO COEF ")
    for i in range(len(cols)):
        print(f" {cols[i]} : {lasso.coef_[i]}")

    print()
    print("RIDGE COEF ")
    for i in range(len(cols)):
        print(f" {cols[i]} : {ridge.coef_[0][i]}") 




