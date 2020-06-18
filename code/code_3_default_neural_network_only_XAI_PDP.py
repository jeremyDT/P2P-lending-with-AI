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

'''Number of nodes per layer for architecture (can iterate over more than one option).'''

nodes1 = [20]#[5, 10, 15, 20, 30]
nodes2 = [5]#[1, 3, 5, 10]

"""for index1, layer_1 in enumerate(nodes1):
    for index2, layer_2 in enumerate(nodes2):

        auc, recall = cross_validate(train_df, layer_1, layer_2, split_size=5)
        print('recall: {} auc: {}'.format(recall, auc))
        recall_grid[index2, index1] += recall
        auc_grid[index2, index1] += auc"""

def feature_imp(model, test_df):

    dictionary = {'loss':{} , 'average_loss':{}}

    test_inpf = functools.partial(easy_input_function, test_df, label_key='charged_off', num_epochs=1, shuffle=False,
                                  batch_size=test_df.shape[0])  # 200000

    benchmark = model_l2.evaluate(test_inpf)
    for column in test_df.columns:
        temp = test_df.copy()
        temp[column] = np.random.permutation(temp[column].values)
        test_inpf = functools.partial(easy_input_function, temp, label_key='charged_off', num_epochs=1,
                                      shuffle=False, batch_size=temp.shape[0])  # 200000

        results = model_l2.evaluate(test_inpf)
        dictionary['loss'][column] = results['loss']/benchmark['loss']
        dictionary['average_loss'][column] = results['average_loss'] / benchmark['average_loss']

    return dictionary
def feature_auc(model, test_df):

    dictionary = {'auc':{} , 'recall':{}}

    test_inpf = functools.partial(easy_input_function, test_df, label_key='charged_off', num_epochs=1, shuffle=False,
                                  batch_size=test_df.shape[0])  # 200000

    benchmark = model_l2.evaluate(test_inpf)
    for column in test_df.columns:
        temp = test_df.copy()
        temp[column] = np.random.permutation(temp[column].values)
        test_inpf = functools.partial(easy_input_function, temp, label_key='charged_off', num_epochs=1,
                                      shuffle=False, batch_size=temp.shape[0])  # 200000

        results = model_l2.evaluate(test_inpf)
        dictionary['auc'][column] = benchmark['auc']/results['auc']
        dictionary['recall'][column] = benchmark['recall']/results['recall']

    return dictionary

print('getting Partial Dependence and average scatter plots for all input features')

for index1, layer_1 in enumerate(nodes1):
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

        matrices_dict = {}

        N_points = 100
        for col_num, column in enumerate(test_df.drop(
                ['charged_off', 'home_ownership', 'verification_status', 'purpose', 'application_type'],
                axis=1).columns):
            newdf = pd.DataFrame(np.repeat(test_df.values, N_points, axis=0))
            newdf.columns = test_df.columns
            newdf['charged_off'] = newdf.charged_off.astype('int64')
            for kol in test_df.drop(['charged_off', 'home_ownership', 'verification_status', 'purpose', 'application_type'],axis=1).columns:
                newdf[kol] = newdf[kol].astype('float64')
            min = test_df[column].min()
            max = test_df[column].max()
            vals = np.random.normal(0,1,N_points)
            #vals = np.arange(min, max, (max - min) / N_points)
            vals = vals.reshape((1, N_points))
            vals = np.repeat(vals, test_df.shape[0], axis=0).flatten()
            store_vals = np.zeros(newdf.shape[0])
            newdf[column] = vals
            """for k in range(0, newdf.shape[0], N_points):
                print(k)
                newdf.iloc[k:k+N_points] = vals"""

            pred_input_fn = tf.estimator.inputs.pandas_input_fn(
                x=newdf.drop(
                ['charged_off'],
                axis=1),
                num_epochs=1,
                shuffle=False
            )
            res = model_l2.predict(pred_input_fn)
            for index, r in enumerate(res):
                store_vals[index] += r['logistic'][0]
                #plt.scatter(vals[index],r['logistic'][0])

                print(index, r['logistic'][0])
            #plt.title(column)
            #plt.show()
            matrices_dict[column] = (vals, store_vals)
            plt.title(column, size = 15)
            h, XE,YE,img = plt.hist2d(vals, store_vals, bins=(15, 10), cmap='hot')
            figname = column + '_PDP.pdf'
            plt.xlabel('feature value', size = 15)
            plt.ylabel('default probability', size = 15)
            plt.savefig(figname, format='pdf', dpi=1000)
            plt.show()
            h_n = (h.T /np.sum(h, axis =1)).T
            h_w = np.repeat(np.array([(YE[i]+YE[i+1])/2 for i in range(YE.shape[0] -1)]).reshape(1,10),15,axis =0)
            plt.title(column, size = 15)
            plt.scatter(np.array([(XE[i]+XE[i+1])/2 for i in range(XE.shape[0] -1)]),np.sum(h_n*h_w, axis = 1))
            figname = column + '_average_PDP.pdf'
            plt.xlabel('feature value', size = 15)
            plt.ylabel('mean default probability', size = 15)
            plt.savefig(figname, format='pdf', dpi=1000)
            plt.show()

            #plt.scatter(vals.reshape((N_points,-1)), store_vals.reshape((N_points,-1)))
            #plt.title(column)
            #plt.show()

            #plt.errorbar(vals[:N_points].reshape((N_points, -1)), store_vals.reshape((N_points, -1)).median(axis=1),store_vals.reshape((N_points, -1)).std(axis=1))

            #plt.plot(vals[:20].reshape((N_points, -1)), store_vals.reshape((N_points, -1)).mean(axis=1))
            #plt.title(column)
            #plt.show()

        clear_output()