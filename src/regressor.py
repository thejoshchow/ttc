import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from joblib import dump, load

def load_data():
    df = pd.read_excel('../data/sizing.xlsx', sheet_name=1)
    X1 = df[['ShoulderWidthBody', 'BustBody', 'WristBody']]
    X2 = df[['BehindClothLengthBody', 'FrontClothLengthBody', 'ShoulderWidthBody', 'BustBody', 'AbdomenBody', \
                            'LeftSleeveLengthBody', 'RightSleeveLengthBody', 'WristBody']]
    y = df[['BehindClothLengthSetNumber', 'FrontClothLengthSetNumber', 'ShoulderWidthSetNumber', 'BustSetNumber', 'AbdomenSetNumber', \
                        'HemSetNumber', 'LeftSleeveLengthSetNumber', 'RightSleeveLengthSetNumber', 'Wrist finish']]
    return X1, X2, y

def rand_grid_search(X, y):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 1000, num = 10)]
    # Number of features to consider at every split
    max_features = [i for i in range(1,len(X.columns))]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X, y)
    return rf_random.best_estimator_

def run_full():
    X1, X2, y = load_data()
    from_3 = rand_grid_search(X1, X2)
    from_3.fit(X1, X2)
    to_4 = rand_grid_search(X2, y)
    to_4.fit(X2, y)
    return from_3, to_4, list(y.columns)

def pickle():
    from_3, to_4, columns = run_full()
    dump(columns, 'column_names.pkl')
    dump(from_3, 'from_3.pkl')
    dump(to_4, 'to_4.pkl')

if __name__ == '__main__':
    pickle()