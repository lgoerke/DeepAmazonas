from Classifiers.classifier_base import Classifier_base

import numpy as np
from hyperopt import hp
import scipy
from tqdm import tqdm
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error

'''
Extreme Gradient Boosting classifier
'''
class Xgb(Classifier_base):

    '''
    Class variable containing parameters to optimize
    together with ranges.
    E.g. hp.uniform( 'dropout', 0, 0.6)
    '''
    space = [hp.choice('n_estimators', np.arange(50, 150+1, dtype=int)),
             hp.uniform('lr', 0.0001, 0.1),
             hp.uniform('min_child_weight', 0.8, 1.2),
             hp.choice('max_depth', np.arange(2, 9+1, dtype=int))
             ]


    def __init__(self, params_path="../models/xgb/xbg_best", n_classes = 17, train_images = 50000, val_images = 10000,
                 test_images = 10000, batch_size = 32):
        self.params_path = params_path
        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.models = []


    def fit(self, train_generator, validation_generator, steps):
        train_data, train_labels = self.generate_data(self.train_data, train_generator)
        for class_i in tqdm(range(self.n_classes), miniters=1):

            file = open(self.params_path + "_{}.pkl".format(class_i), 'rb')
            params_dict = pickle.load(file)
            file.close()

            n_estimators = params_dict['n_estimators']
            lr = params_dict['lr']
            min_child_weight = params_dict['min_child_weight']
            max_depth = params_dict['max_depth']

            model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=lr, n_estimators=n_estimators, \
                                      min_child_weight=min_child_weight, \
                                      scale_pos_weight=1, base_score=0.5, objective='binary:logistic')

            model.fit(train_data, train_labels[:,class_i])

            self.models.append(model)


    def predict(self, test_imgs):
        predictions = np.array(len(test_imgs), self.n_classes)
        for class_i in tqdm(range(self.n_classes), miniters=1):
            pred = self.models[class_i].predict_proba(test_imgs)[:, 1]
            predictions[:,class_i] = pred

        pred_labels = []
        for i, y_pred_row in enumerate(predictions):
            pred_labels.append([1 if element > 0.2 else 0 for element in y_pred_row])
        pred_labels = np.asarray(pred_labels)


        return pred_labels


    def evaluate(self, validation_generator, steps):
        val_data, val_labels = self.generate_data(self.val_images, validation_generator)
        predictions = np.array(self.val_images, self.n_classes)

        for class_i in tqdm(range(self.n_classes), miniters=1):
            val_pred = self.models[class_i].predict_proba(val_data)[:, 1]
            predictions[:,class_i] = val_pred

        pred_labels = []
        for i, y_pred_row in enumerate(predictions):
            pred_labels.append([1 if element > 0.2 else 0 for element in y_pred_row])
        pred_labels = np.asarray(pred_labels)

        return mean_squared_error(val_labels, pred_labels)


    def generate_data(self, num_images, generator):

        data = np.ndarray(shape=(num_images,18), dtype=float)
        labels = np.ndarray(shape=(num_images, self.num_classes), dtype=float)
        count = 0
        for imgs, lbls in generator.flow(batch_size=self.batch_size):
            features = self.extract_features(imgs)
            data[count : count + self.batch_size] = features
            labels[count : count + self.batch_size] = lbls
            count += self.batch_size
            if count >= self.num_images:
                break

        return data, labels

    def extract_features(df, data):

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

        for im in data:
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


        result = np.concatenate( (r_mean, g_mean, b_mean, r_std, g_std, b_std, r_max, g_max, b_max,
                                     r_min, g_min, b_min, r_kurtosis, g_kurtosis, b_kurtosis,
                                     r_skewness, g_skewness, b_skewness), axis=1)

        return result