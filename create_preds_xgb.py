# Original idea from https://www.kaggle.com/opanichev/xgb-starter-lb-0-88232

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import pickle
import scipy
from sklearn.model_selection import KFold
import operator
import os.path
import sys
from PIL import Image

from sklearn.metrics import mean_squared_error

from hyperopt import STATUS_OK

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
train_path = '../input/train-jpg/'
test_path = '../input/test-jpg/'
train_labels = pd.read_csv('input/train.csv')
test_labels = pd.read_csv('input/sample_submission.csv')

#############################################################
#############################################################
def run_model(data=None, labels=None, train_index=None, val_index=None, num_label=0, params=None): # scale_pos_weight=1, lambda=1

    global best
    global model

    n_estimators = int(params[0])
    lr = params[1]
    min_child_weight = params[2]
    max_depth = int(params[3])
    params = {'n_estimators': n_estimators, 'lr':lr, 'min_child_weight':min_child_weight, 'max_depth':max_depth}

    # Get training and validation set
    train_data = data[train_index]
    train_labels = labels[train_index]
    val_data = data[val_index]
    val_labels = labels[val_index]

    # Parameters doc: http://xgboost.readthedocs.io/en/latest/parameter.html
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    xgb_model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=lr, n_estimators=n_estimators,
                              min_child_weight=min_child_weight,
                            scale_pos_weight=1, base_score=0.5, objective='binary:logistic')


    xgb_model.fit(train_data, train_labels)
    labels_pred = xgb_model.predict_proba(val_data)[: , 1]

    loss = mean_squared_error(val_labels, labels_pred) # Use weights to fight class imbalance?

    if loss < best:
        best = loss
        model = xgb_model
        pickle.dump(params, open('../models/xgb/xbg_best_{}.pkl'.format(num_label), 'wb'))
        print('New best params for label {}: {}'.format(num_label, params))

    return {'loss': loss, 'status': STATUS_OK}


def extract_features(df, data_path):
    im_features = df.copy()

    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    r_max = []
    g_max = []
    b_max = []

    r_min = []
    g_min = []
    b_min = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []

    r_skewness = []
    g_skewness = []
    b_skewness = []

    for image_name in tqdm(im_features.image_name.values, miniters=100):
        im = Image.open(data_path + image_name + '.jpg')
        im = np.array(im)[:, :, :3]

        r_mean.append(np.mean(im[:, :, 0].ravel()))
        g_mean.append(np.mean(im[:, :, 1].ravel()))
        b_mean.append(np.mean(im[:, :, 2].ravel()))

        r_std.append(np.std(im[:, :, 0].ravel()))
        g_std.append(np.std(im[:, :, 1].ravel()))
        b_std.append(np.std(im[:, :, 2].ravel()))

        r_max.append(np.max(im[:, :, 0].ravel()))
        g_max.append(np.max(im[:, :, 1].ravel()))
        b_max.append(np.max(im[:, :, 2].ravel()))

        r_min.append(np.min(im[:, :, 0].ravel()))
        g_min.append(np.min(im[:, :, 1].ravel()))
        b_min.append(np.min(im[:, :, 2].ravel()))

        r_kurtosis.append(scipy.stats.kurtosis(im[:, :, 0].ravel()))
        g_kurtosis.append(scipy.stats.kurtosis(im[:, :, 1].ravel()))
        b_kurtosis.append(scipy.stats.kurtosis(im[:, :, 2].ravel()))

        r_skewness.append(scipy.stats.skew(im[:, :, 0].ravel()))
        g_skewness.append(scipy.stats.skew(im[:, :, 1].ravel()))
        b_skewness.append(scipy.stats.skew(im[:, :, 2].ravel()))

    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min

    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis

    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness

    return im_features



#############################################################
#############################################################


# Extract features
#print('Extracting train features')
#train_features = extract_features(train_labels, train_path)
#print('Extracting test features')
#test_features = extract_features(test_labels, test_path)
# Save features
#with open('train_features.out', 'wb') as outfile:
#    pickle.dump(train_features, outfile)
#with open('test_features.out', 'wb') as outfile:
#    pickle.dump(test_features, outfile)
with open('scripts/train_features.out', 'rb') as data_file:
    train_features = pickle.load(data_file)
with open('scripts/test_features.out', 'rb') as data_file:
    test_features = pickle.load(data_file)


# Prepare data
# Delete tags from the dictionary
X = np.array(train_features.drop(['image_name', 'tags'], axis=1))
y_train = []

# Get labels
flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))

# Create dictionary with name-number and number-name
#label_map = {l: i for i, l in enumerate(labels)}
label_map = {'blow_down': 0,
          'bare_ground': 1,
          'conventional_mine': 2,
          'blooming': 3,
          'cultivation': 4,
          'artisinal_mine': 5,
          'haze': 6,
          'primary': 7,
          'slash_burn': 8,
          'habitation': 9,
          'clear': 10,
          'road': 11,
          'selective_logging': 12,
          'partly_cloudy': 13,
          'agriculture': 14,
          'water': 15,
          'cloudy': 16}

# Create one-hot encoding
for tags in tqdm(train_labels.tags.values, miniters=1000):
    targets = np.zeros(len(labels))
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)

y = np.array(y_train, np.uint8)
# Save encoding
with open('onehot_encoding.out', 'wb') as outfile:
    pickle.dump(y, outfile)


# Number of classes
n_classes = y.shape[1]


''' GET PREDICTIONS FOR THE TRAINING DATA'''

# # Reserve space for variables for the analysis
# pred_probs = np.zeros((X.shape[0], n_classes))
# pred_labels = np.zeros((X.shape[0], n_classes))
#
# if not os.path.isdir('models/xgb'):
#     print('Parameters for the models are not in models/xgb/')
#     sys.exit('Run the basic xgboost script to get the model parameters.')
#
# print('Training and making predictions')
#
# skf = KFold(n_splits=5)
#
# for train_index, val_index in skf.split(X):
#
#     for class_i in tqdm(range(n_classes), miniters=1):
#
#         file = open("models/xgb/xbg_best_{}.pkl".format(class_i), 'rb')
#         params_dict = pickle.load(file)
#         file.close()
#
#         n_estimators = params_dict['n_estimators']
#         lr = params_dict['lr']
#         min_child_weight = params_dict['min_child_weight']
#         max_depth = params_dict['max_depth']
#
#         model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=lr, n_estimators=n_estimators, \
#                                   min_child_weight=min_child_weight, \
#                                 scale_pos_weight=1, base_score=0.5, objective='binary:logistic')
#
#         model.fit(X[train_index], y[train_index, class_i])
#
#         val_pred = model.predict_proba(X[val_index])[:, 1]
#         pred_probs[val_index,class_i] = val_pred
#
# pred_labels = []
# for i, y_pred_row in enumerate(pred_probs):
#     pred_labels.append([1 if element > 0.2 else 0 for element in y_pred_row])
# pred_labels = np.asarray(pred_labels)
#
# # Write output arrays to .csv file, to be called after the main loop
# file =  "results/xgb/xgb"
#
# sorted_x = sorted(label_map.items(), key=operator.itemgetter(1))
# lbls = [i[0] for i in sorted_x]
#
# df_pred_probs = pd.DataFrame(pred_probs, columns=lbls)
# df_pred_labels = pd.DataFrame(pred_labels, columns=lbls)
#
# df_pred_probs.to_csv(file + '_pred_probs_train.csv')
# df_pred_labels.to_csv(file + '_pred_lbls_train.csv')

'''GET PREDICTIONS FOR THE TEST DATA'''
X_test = np.array(test_features.drop(['image_name', 'tags'], axis=1))
# Reserve memory space
y_pred = np.zeros((X_test.shape[0], n_classes))

for class_i in tqdm(range(n_classes), miniters=1):

    file = open("models/xgb/xbg_best_{}.pkl".format(class_i), 'rb')
    params_dict = pickle.load(file)
    file.close()

    n_estimators = params_dict['n_estimators']
    lr = params_dict['lr']
    min_child_weight = params_dict['min_child_weight']
    max_depth = params_dict['max_depth']

    model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=lr, n_estimators=n_estimators, \
                              min_child_weight=min_child_weight, \
                              scale_pos_weight=1, base_score=0.5, objective='binary:logistic')

    model.fit(X, y[:,class_i])

    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]

# Test data
X_test = np.array(test_features.drop(['image_name', 'tags'], axis=1))

sorted_x = sorted(label_map.items(), key=operator.itemgetter(1))
lbls = [i[0] for i in sorted_x]

df_y_pred = pd.DataFrame(y_pred, columns=lbls)

file =  "results/xgb/xgb"
df_y_pred.to_csv(file + '_y_probs_test.csv')

preds = [' '.join(labels[y_pred_row > 0.2]) for y_pred_row in y_pred]

subm = pd.DataFrame()
subm['image_name'] = test_features.image_name.values
subm['tags'] = preds
subm.to_csv(file + '_submission_xgb.csv', index=False)

