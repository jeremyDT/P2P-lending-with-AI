'''import the required packages and read the file.'''

import tensorflow as tf
import tensorflow.feature_column as fc
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.preprocessing import Imputer

from sklearn.metrics import recall_score

tf.enable_eager_execution()

import pandas as pd
import copy
import random

random.seed( 10 )

#######################################################

data = pd.read_csv('../input_file_2.csv', sep = ',', index_col=0)

'''parse column to date format'''

data['issue_d'] = pd.to_datetime(data['issue_d'])

print(data.shape)

'''Exclude the period from the beginning of 2016 onward as it is noticed in time-series plots that this period has
 a significant portion of the expected defaults not reported yet. This would hence be biased.'''

#data = data.loc[data['issue_d'] < '2016-1-1']

print(data.shape)

'''categorical columns to potentially drop if only numerical features are being considered and to exclude
 when scaling features to avoid errors.'''

to_drop_categorical = ['home_ownership', 'verification_status', 'purpose', 'application_type']

'''split training and test data by date quantile.'''

train_df = data.loc[data['issue_d'] < data['issue_d'].quantile(0.75)]
test_df = data.loc[data['issue_d'] >= data['issue_d'].quantile(0.75)]

print('train df ', train_df['issue_d'].describe())
print('test df ', test_df['issue_d'].describe())

'''Drop the date column as not needed for the model.'''

train_df.drop('issue_d', axis=1, inplace=True)
test_df.drop('issue_d', axis=1, inplace=True)

all_cols = list(train_df.columns)

print('to scale     ', all_cols)

print(len(all_cols))

all_cols.remove('charged_off')

'''filter for only numerical columns whicb will be then scaled.'''

for i in to_drop_categorical:

    all_cols.remove(i)

print('to scale     ', all_cols)

print(len(all_cols))

'''Fill null values by mean imputation.'''

train_df[all_cols] = train_df[all_cols].fillna(train_df[all_cols].mean())
test_df[all_cols] = test_df[all_cols].fillna(train_df[all_cols].mean())
print('null values      ', train_df.isnull().sum())
print(type(train_df))


from sklearn.preprocessing import StandardScaler

'''Scale values of numerical columns'''

scaler = StandardScaler(copy=False)

train_df[all_cols] = scaler.fit_transform(train_df[all_cols], train_df['charged_off'])#(train_df[all_cols])
test_df[all_cols] = scaler.transform(test_df[all_cols], test_df['charged_off'])#(test_df[all_cols])


#del X_train, y_train, X_test, y_test

################################################################################

'''Split by label categories to downsample and provide balanced classes for training.'''

train_dat_1s = train_df[train_df['charged_off'] == 1]

train_dat_0s = train_df[train_df['charged_off'] == 0]

keep_0s = train_dat_0s.sample(frac=train_dat_1s.shape[0]/train_dat_0s.shape[0])


train_df = pd.concat([keep_0s,train_dat_1s],axis=0)

################################################################################

'''Split by label categories to downsample and provide balanced classes for testing (to have a meaningful AUC score as well).'''


test_dat_1s = test_df[test_df['charged_off'] == 1]

test_dat_0s = test_df[test_df['charged_off'] == 0]

keep_0s = test_dat_0s.sample(frac=test_dat_1s.shape[0]/test_dat_0s.shape[0])


test_df = pd.concat([keep_0s,test_dat_1s],axis=0)

################################################################################

'''rep_1 =[train_dat_1s for x in range(train_dat_0s.shape[0]//train_dat_1s.shape[0] )]
keep_1s = pd.concat(rep_1, axis=0)

train_df = pd.concat([keep_1s,train_dat_0s],axis=0)'''

'''Check for test and training shapes and value counts'''

print(train_dat_1s.shape[0]/train_dat_0s.shape[0], train_dat_1s.shape[0], train_dat_0s.shape[0])

print('train and test shape     ', train_df.shape, test_df.shape)

print('value counts train     ', train_df['charged_off'].value_counts())

print('value counts test     ', test_df['charged_off'].value_counts())

del keep_0s
#del keep_1s
del train_dat_1s
del train_dat_0s
del data

print(train_df.head())

'''define a simple input function for the models with the intuitive input names below.'''

def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
  label = df[label_key]
  ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

  if shuffle:
    ds = ds.shuffle(10000)

  ds = ds.batch(batch_size).repeat(num_epochs)

  return ds

###################################################################
import functools

'''Define training and test input functions with their parameters.'''

train_inpf = functools.partial(easy_input_function, train_df, label_key='charged_off',  num_epochs=5, shuffle=True, batch_size=20000)#300000 #230934
test_inpf = functools.partial(easy_input_function, test_df, label_key='charged_off', num_epochs=1, shuffle=False, batch_size=200000) #200000
###################################################################
#print(fc.input_layer(feature_batch, [loan_amnt]).numpy())
##########################################################
'''classifier = tf.estimator.LinearClassifier(feature_columns=[loan_amnt])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

clear_output()  # used for display in notebook
print(result)'''

################################################
#DEFINE ALL NUMERIC COLUMNS

loan_amnt = fc.numeric_column('loan_amnt')
term = fc.numeric_column('term')
installment = fc.numeric_column('installment')
emp_length = fc.numeric_column('emp_length')
dti = fc.numeric_column('dti')
earliest_cr_line = fc.numeric_column('earliest_cr_line')
open_acc = fc.numeric_column('open_acc')
pub_rec = fc.numeric_column('pub_rec')
revol_util = fc.numeric_column('revol_util')
total_acc = fc.numeric_column('total_acc')
mort_acc = fc.numeric_column('mort_acc')
pub_rec_bankruptcies = fc.numeric_column('pub_rec_bankruptcies')
log_annual_inc = fc.numeric_column('log_annual_inc')
fico_score = fc.numeric_column('fico_score')
log_revol_bal = fc.numeric_column('log_revol_bal')

my_numeric_columns = [loan_amnt,
term,
installment,
emp_length,
dti,
earliest_cr_line,
open_acc,
pub_rec,
revol_util,
total_acc,
mort_acc,
pub_rec_bankruptcies,
log_annual_inc,
fico_score,
log_revol_bal]

##############################################

#RETRAIN MODEL ON ALL THESE CATEGORICAL COLUMNS AS WELL


def metric_auc(labels, predictions):
    return {
        'auc_precision_recall': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }

def metric_recall_0(labels, predictions):
    return {
        'recall_0': tf.metrics.recall(
            labels=labels, predictions=predictions['logistic'], name = '0')
    }

def metric_recall_1(labels, predictions):
    return {
        'recall_1': tf.metrics.recall(
            labels=labels, predictions=predictions['logistic'], name = '1')
    }

print('ALL NUMERIC COLUMNS LINEAR CLASSIFIER')

classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)

classifier = tf.contrib.estimator.add_metrics(classifier, metric_auc)
#classifier = tf.contrib.estimator.add_metrics(classifier, metric_recall_0)
#classifier = tf.contrib.estimator.add_metrics(classifier, metric_recall_1)


classifier.train(train_inpf)

result = classifier.evaluate(test_inpf)

predictions = classifier.predict(test_inpf)

#print('recall 0     ', recall_score(predictions, test_df['charged_off'], pos_label=0))
#print('recall 1     ', recall_score(predictions, test_df['charged_off'], pos_label=1))

#print(predictions.shape)

clear_output()

for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))