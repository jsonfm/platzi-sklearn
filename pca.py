"""PCA"""

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    sns.set_theme()
    dt_heart = pd.read_csv('./data/heart.csv')
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']
    
    dt_features = StandardScaler().fit_transform(dt_features)
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    pca = PCA(n_components=3)
    pca.fit(X_train)
    pca_variance = pca.explained_variance_ 

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    ipca_variance = ipca.explained_variance_

    logistic = LogisticRegression(solver='lbfgs')
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)

    score_pca = logistic.score(dt_test, y_test)

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)

    score_ipca = logistic.score(dt_test, y_test)

    print(f"Score PCA: {score_pca}")
    print(f"Score Incremental PCA: {score_ipca}")

    plt.plot(range(len(pca_variance)), pca_variance)
    plt.plot(range(len(ipca_variance)), ipca_variance)
    plt.show()