'''import the required packages and read the file.'''

import pandas as pd
import numpy as np


print('reading file')

data = pd.read_csv('../input_file_1.csv', sep = ',', index_col=0)

print('file shape', data.shape)

'''parse column to date format'''

print('date encoding')

data['issue_d'] = pd.to_datetime(data['issue_d'])

'''check for and remove datapoints with null values.'''

print(data['issue_d'].isnull().any(), data['purpose'].isnull().any())

print('remove null datapoints to see if it helps...')

data = data.loc[data['purpose'].isnull() == False]

'''eliminate purpose categories with low count.'''

print('eliminating small count categories')

threshold = 19000

counts = data['purpose'].value_counts()

keep_list = counts[counts > threshold].index

data = data[data['purpose'].isin(keep_list)]

'''replace the existing labels so that they can be called easily from pandas and TensorFlow'''

print('replacing labels')

to_replace = {
    'Debt consolidation': 'debt_consolidation',
    'Home improvement': 'home_improvement',
    'Credit card refinancing': 'credit_card',
    'Other': 'other',
    'Vacation': 'vacation',
    'Medical expenses': 'medical',
    'Car financing': 'car',
    'Major purchase': 'major_purchase',
    'Moving and relocation': 'moving',
    'Home buying': 'house'
}

data['purpose'] = data['purpose'].replace(to_replace)

print(data['purpose'].value_counts())

'''Create one-hot encoded dummy columns for categorical variables.'''

print('hot encoding')

data = pd.get_dummies(data, columns=['purpose'], drop_first=False)

print('data columns AFTER hot encoding      ', data.columns)

'''split training and test data by date quantile.'''

data_train = data.loc[data['issue_d'] < data['issue_d'].quantile(0.9)]
data_test = data.loc[data['issue_d'] >= data['issue_d'].quantile(0.9)]

print('Number of loans in the partition:   ', data_train.shape[0] + data_test.shape[0])
print('Number of loans in the full dataset:', data.shape[0])

'''Drop the date column as not needed for the model.'''

data_train.drop('issue_d', axis=1, inplace=True)
data_test.drop('issue_d', axis=1, inplace=True)

'''Split features and labels'''

y_train = data_train['rejected']
y_test = data_test['rejected']

X_train = data_train.drop('rejected', axis=1)
X_test = data_test.drop('rejected', axis=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier

'''Build a pipeline for preprocessing and training'''

pipeline_sgdlogreg = Pipeline([
    ('imputer', Imputer(copy=False)), # Mean imputation by default
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(
        class_weight='balanced',
        loss='log',
        max_iter=1000,
        tol = 1e-3,
        random_state=1,
        n_jobs=10,
        warm_start=True
        )
    )
])



param_grid_sgdlogreg = {
    'model__alpha': [10**-3, 10**-2, 10**1],
    'model__penalty': ['l1', 'l2']
}

'''Set up a grid search.'''

grid_sgdlogreg = GridSearchCV(
    estimator=pipeline_sgdlogreg,
    param_grid=param_grid_sgdlogreg,
    scoring='roc_auc',
    pre_dispatch=3,
    n_jobs=5,
    cv=5,
    verbose=5,
    return_train_score=False
)

'''Fit the model.'''

print('fitting')

grid_sgdlogreg.fit(X_train, y_train)

'''Print model parameters, best parameters and best score.'''

print('parameters       ', grid_sgdlogreg._get_param_iterator())

print(grid_sgdlogreg.best_params_, grid_sgdlogreg.best_score_)

from sklearn.metrics import roc_auc_score, recall_score

'''Make predictions on test dataset.'''

y_score = grid_sgdlogreg.predict_proba(X_test)[:,1]

y_score_flag = [int(round(i)) for i in y_score]

'''Two ways of evaluating results, check that they match.'''

print('LOOK FOR DISCREPANCIES HERE...')

print(roc_auc_score(y_test, y_score), recall_score(y_test, y_score_flag, pos_label=1), recall_score(y_test, y_score_flag, pos_label=0))

y_score_flag = grid_sgdlogreg.predict(X_test)

print(roc_auc_score(y_test, y_score), recall_score(y_test, y_score_flag, pos_label=1), recall_score(y_test, y_score_flag, pos_label=0))

