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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

random.seed( 10 )

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

#######################################################

data = pd.read_csv('../data/input_file_2.csv.zip', sep = ',', index_col=0, compression='zip')

'''parse column to date format'''

data['issue_d'] = pd.to_datetime(data['issue_d'])

print(data.shape)

'''Exclude the period from the beginning of 2016 onward as it is noticed in time-series plots that this period has
 a significant portion of the expected defaults not reported yet. This would hence be biased.'''

data = data.loc[data['issue_d'] < '2016-1-1']

print(data.shape)

'''categorical columns to potentially drop if only numerical features are being considered and to exclude
 when scaling features to avoid errors.'''

to_drop_categorical = ['home_ownership', 'verification_status', 'purpose', 'application_type']

'''split training and test data by date quantile.'''

train_df = data.loc[data['issue_d'] < data['issue_d'].quantile(0.90)]
test_df = data.loc[data['issue_d'] >= data['issue_d'].quantile(0.90)]

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

train_df[all_cols] = scaler.fit_transform(train_df[all_cols], train_df['charged_off'])
test_df[all_cols] = scaler.transform(test_df[all_cols], test_df['charged_off'])


################################################################################

'''Split by label categories to downsample and provide balanced classes for training.'''

train_dat_1s = train_df[train_df['charged_off'] == 1]

train_dat_0s = train_df[train_df['charged_off'] == 0]

keep_0s = train_dat_0s.sample(frac=train_dat_1s.shape[0]/train_dat_0s.shape[0])

train_df = pd.concat([keep_0s,train_dat_1s],axis=0)

#train_df, val_df = train_test_split(train_df, test_size=0.1)

################################################################################

'''Split by label categories to downsample and provide balanced classes for testing (to have a meaningful AUC score as well).'''


#test_dat_1s = test_df[test_df['charged_off'] == 1]

#test_dat_0s = test_df[test_df['charged_off'] == 0]

#keep_0s = test_dat_0s.sample(frac=test_dat_1s.shape[0]/test_dat_0s.shape[0])


#test_df = pd.concat([keep_0s,test_dat_1s],axis=0)

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
#val_inpf = functools.partial(easy_input_function, val_df, label_key='charged_off', num_epochs=1, shuffle=False, batch_size=val_df.shape[0]) #200000
test_inpf = functools.partial(easy_input_function, test_df, label_key='charged_off', num_epochs=1, shuffle=False, batch_size=test_df.shape[0]) #200000
###################################################################

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

########################################################

#NOW FOR CATEGORICAL COLUMNS...

print('Now encoding categorical columns')

'''relationship = fc.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

print(fc.input_layer(feature_batch, [age, fc.indicator_column(relationship)]))'''

home_ownership = tf.feature_column.categorical_column_with_hash_bucket(
    'home_ownership', hash_bucket_size=100000)

verification_status = tf.feature_column.categorical_column_with_hash_bucket(
    'verification_status', hash_bucket_size=100000)

purpose = tf.feature_column.categorical_column_with_hash_bucket(
    'purpose', hash_bucket_size=100000)

application_type = tf.feature_column.categorical_column_with_hash_bucket(
    'application_type', hash_bucket_size=100000)



print('actual DNN Grid Search with non-linearity')

def cross_validate(train_df, layer_1, layer_2, split_size=5):
    results = []
    kf = StratifiedKFold(n_splits=split_size, shuffle = True)
    kf.get_n_splits(train_df.drop('charged_off', axis = 1).values, train_df['charged_off'].values)
    for train_idx, val_idx in kf.split(train_df.drop('charged_off', axis = 1).values, train_df['charged_off'].values):
        train_x = train_df.iloc[train_idx]
        #train_y = train_df.iloc[train_idx]
        val_x = train_df.iloc[val_idx]
        print(train_x.shape,val_x.shape)
        print(train_idx, val_idx)
        #val_y = train_df.iloc[val_idx]
        train_inpf = functools.partial(easy_input_function, train_x, label_key='charged_off', num_epochs=5, shuffle=True,
                                 batch_size=20000)  # 300000 #230934
        val_inpf = functools.partial(easy_input_function, val_x, label_key='charged_off', num_epochs=1, shuffle=True,
                               batch_size=val_x.shape[0])  # 200000

        model_l2 = tf.estimator.DNNClassifier(
            hidden_units=[layer_1, layer_2],
            feature_columns=my_numeric_columns,
            activation_fn=tf.nn.tanh,
            dropout=0.2,
            optimizer="Adam")
            #batch_norm=False
            #callbacks=[es, mc, csv_logger])

        model_l2 = tf.contrib.estimator.add_metrics(model_l2, metric_auc)


        model_l2.train(train_inpf)

        #print('TEST RESULTS ', layer_1, layer_2)

        outputs = model_l2.evaluate(val_inpf)
        clear_output()
        results.append((outputs['recall'], outputs['auc']))
    auc = 0
    recall = 0
    for el in results:
        auc += el[1]
        recall += el[0]
    auc = auc/len(results)
    recall = recall / len(results)
    return auc, recall
recall_grid = np.zeros((4, 5))
auc_grid = np.zeros((4, 5))
nodes1 = [5, 10, 15, 20, 30]
nodes2 = [1, 3, 5, 10]

for index1, layer_1 in enumerate(nodes1):
    for index2, layer_2 in enumerate(nodes2):

        auc, recall = cross_validate(train_df, layer_1, layer_2, split_size=5)
        print('recall: {} auc: {}'.format(recall, auc))
        recall_grid[index2, index1] += recall
        auc_grid[index2, index1] += auc

"""for index1, layer_1 in enumerate(nodes1):
    for index2, layer_2 in enumerate(nodes2):

        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE, min_delta=MIN_DELTA)

        model_l2 = tf.estimator.DNNClassifier(
            hidden_units=[layer_1, layer_2],
            feature_columns=my_numeric_columns,
            activation_fn=tf.nn.tanh,
            dropout=0.2,
            optimizer="Adam")
            #callbacks=[es, mc, csv_logger])

        model_l2 = tf.contrib.estimator.add_metrics(model_l2, metric_auc)


        model_l2.train(train_inpf)

        #print('TEST RESULTS ', layer_1, layer_2)

        results = model_l2.evaluate(val_inpf)
        clear_output()
        for key in sorted(results):
            print('%s: %0.2f' % (key, results[key]))
            if key == 'recall':
                recall_grid[index2, index1] += results[key]
            elif key == 'auc':
                auc_grid[index2, index1] += results[key]"""


##########################################

'''Plot gridsearch results'''

fig, ax = plt.subplots()
im = ax.imshow(recall_grid, cmap='hot', vmin=0.65, vmax=0.69)#, cmap='hot', interpolation='nearest')

# We want to show all ticks...
ax.set_xticks(np.arange(len(nodes1)))
ax.set_yticks(np.arange(len(nodes2)))
# ... and label them with the respective list entries
ax.set_xticklabels( nodes1)
ax.set_yticklabels(nodes2)
ax.set_ylabel('layer 2')
ax.set_xlabel('layer 1')

# Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(nodes2)):
    for j in range(len(nodes1)):
        text = ax.text(j, i, '%0.1f' % (recall_grid[i, j]*100) + '%',
                       ha="center", va="center", color="black")

ax.set_title("Recall Values across network architectures")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(auc_grid, cmap='hot', vmin=0.65, vmax=0.71)

# We want to show all ticks...
ax.set_xticks(np.arange(len(nodes1)))
ax.set_yticks(np.arange(len(nodes2)))
# ... and label them with the respective list entries
ax.set_xticklabels(nodes1)
ax.set_yticklabels(nodes2)
ax.set_ylabel('layer 2')
ax.set_xlabel('layer 1')

# Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(nodes2)):
    for j in range(len(nodes1)):
        text = ax.text(j, i, '%0.1f' % (auc_grid[i, j]*100) + '%',
                       ha="center", va="center", color="black")

ax.set_title("AUC Values across network architectures")
fig.tight_layout()
plt.show()