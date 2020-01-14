import pickle

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# determinism
np.random.seed(42)

pickle_file = './weather_outages_log.pkl'
with open(pickle_file, 'rb') as f:
    df = pickle.load(f)

df['outage_duration_lag_avg'] = df.sort_values(
    by='snapshot').groupby('regionName')['outage_duration_hrs'].transform(
        lambda x: x.rolling(30, 1).mean())

# outages with estimates
est_df = df.dropna(subset=['currentEtor'])

# len(est_df)

# Evaluate RMSE of PG&E predictions
# Existing baseline

rmse = est_df['est_repair_error_hrs'].apply(lambda x: x**2).mean()**.5
print('PGE RMSE baseline to beat: {}'.format(rmse))

# Feature engineering & ML Pipeline Definition
# - Vectorize (onehot): regionName, hazardFlag, (if time): day of week, time of 
# day started
# - Continous features: estCustAffected, outage_duration_lag_avg
# - Target: outage_duration_hrs

# create the preprocessing pipelines for both numeric and categorical data
numeric_features = ['estCustAffected', 'outage_duration_lag_avg']
numeric_transformer = Pipeline(
    steps=[('imputer',
            SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

categorical_features = ['regionName', 'hazardFlag']
categorical_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')
            ), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[(
    'num', numeric_transformer,
    numeric_features), ('cat', categorical_transformer, categorical_features)])

# full prediction pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor), ('svm', svm.SVR())])

# Define a pipeline to search for the best combination of one hot encoding and 
# classifier
X_train, X_test, y_train, y_test = train_test_split(
    est_df, est_df['outage_duration_hrs'], test_size=0.2)

est_df['month'] = est_df['when_start_time'].apply(lambda x: x.month)


def cat_train_test_split(df, split_month, y_col):
    X_train = df[df['month'] < split_month]
    X_test = df[df['month'] >= split_month]
    y_train = X_train[y_col]
    y_test = X_test[y_col]
    return X_train, X_test, y_train, y_test


# split out randomly - don't want to do that for this problem
# X_train, X_test, y_train, y_test = train_test_split(est_df, 
# est_df['outage_duration_hrs'], test_size=0.2)

# split out by time
X_train, X_test, y_train, y_test = cat_train_test_split(
    est_df, 11, 'outage_duration_hrs')

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': [1e-3, 1e-4],
    'svm__C': [1, 10, 100, 1000]
}

search = GridSearchCV(clf,
                      param_grid,
                      n_jobs=-1,
                      scoring='neg_root_mean_squared_error')
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

print('ML Model RMSE: {}'.format(-search.best_score_))