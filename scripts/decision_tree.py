import numpy as np
import pandas as pd
from Classifiers.simple_net import SimpleNet
from skimage import io as skio
import keras
import cv2
import os
import pickle
from matplotlib import pyplot as plt

LABELS = {'blow_down': 0,
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


def test_Nodes():
    '''
     |       Little test tree... 
     |
    / \
    | |
    '''

    X, y = load_example()

    y = add_class('infrastructure', ['conventional_mine', 'artisinal_mine', 'slash_burn' ], y)
    y = add_class('forest_phenomena', ['blow_down', 'blooming', 'cultivation', 'slash_burn', 'selective_logging'], y)

    node_3_1_phenomena_investigator = Node('node_3_1_phenomena_investigator', [],
                                           ['blow_down', 'blooming', 'slash_burn', 'selective_logging'],
                                           ['haze', 'partly_cloudy', 'clear'])

    node_2_1_forestphenomena = Node('node_2_1_forestphenomena',
                                    ['node_3_1_phenomena_investigator'],
                                    ['forest_phenomena'],
                                    ['haze', 'partly_cloudy', 'clear'])

    node_3_0_infra_investigator = Node('node_3_0_infra_investigator', [],
                                       ['conventional_mine', 'artisinal_mine', 'road'],
                                       ['haze', 'partly_cloudy', 'clear'])

    node_2_0_infrastructure = Node('node_2_0_infrastructure',
                                   ['node_3_0_infra investigtor'],
                                   ['infrastructure'],
                                   ['haze', 'partly_cloudy', 'clear'])

    node_1_land = Node('node_1_land',
                       [node_2_0_infrastructure, node_2_1_forestphenomena],
                       ['agriculture', 'habitation', 'water', 'primary', 'cultivation', 'bare_ground'],
                       ['haze', 'partly_cloudy', 'clear'])

    node_0_weather = Node('node_0_weather', [node_1_land], ['cloudy', 'haze', 'partly_cloudy', 'clear'])

    return node_0_weather, [X, y]


def add_class(name, subclasses, y):
    indeces = []

    for c in subclasses:
        indeces.append(LABELS[c])

    LABELS[name] = len(LABELS)

    return np.append(y, np.expand_dims(np.any(y[:, indeces], axis=1), axis=1), axis=1)


# def load_example(n=10, path=''):
#     '''
#     load a few pictures/targets info a sample matrix.
#     '''
#
#     train_path = 'input/train-tif-v2/'
#     test_path = 'input/test-jpg/'
#     train_labels = pd.read_csv(path + 'input/train_v2.csv')
#     # test_labels = pd.read_csv('input/sample_submission.csv')
#
#     X = np.zeros((n, 224, 224, 4))
#     y = np.zeros((n, 17))
#
#     for i, imgstr in enumerate(train_labels.image_name.values):
#         p = train_path + imgstr + '.tif'
#
#         if i >= n: return X, y
#
#         img = cv2.resize(skio.imread(p), (224, 224))
#         X[i, :, :, :] = img  # cv2.resize(img,(255,255))
#
#         i = +i
#
#         for t in train_labels.tags.values[i].split(' '):
#             y[i, LABELS[t]] = 1


class Node():
    '''One node in a decision tree, which creates one clsfr instance and can create a dataset for it.'''

    def __init__(self, name, children, use_labels, include_classes=None, clsfr=None):
        '''
        @args:
        -name: Name for this node
        -children: array of Node()s that receive their input from this Node.
        -use_lables: Array of labels to include ['forest','clouds']
        -include_classes: Images to use when classified as 1 already, empty = all.
        
        '''
        self.name = name
        self.X_mask = None
        self.y_history = None
        self.use_lables = use_labels
        self.lbls = None
        self.children = children
        self.include_classes = include_classes
        self.name = name
        if clsfr is None:
            self.clsfr = self.__init_clsfr__()
        else:
            self.clsfr = clsfr

    def __init_clsfr__(self, size=None):
        '''Create a new classifier object bound to this node'''

        return SimpleNet(n_classes=len(self.use_lables), nb_epoch=1, batch_size=4)

    def create_training_set(self, X, t, y_history):
        '''create a training set tailored to this decision operation.
        
        @args:
        X: Pointer to your image set
        y: target values
        y_history: np array documenting the current classification situation: 

        '''

        t = np.copy(t)

        # Keep only certain defined objects that have been classified as sth before (e.g. all cloudy > .5)
        X_mask = np.ones((X.shape[0]))

        if self.include_classes is not None:
            for i, cat in enumerate(self.include_classes):
                X_mask = np.multiply(X_mask, (y_history[:, LABELS[cat]] > 0.5))

        X1 = np.copy(X[(X_mask.astype(np.bool))])

        # Cut away all uninportant class labels from y:
        lbls = np.zeros((len(LABELS.keys())))
        for lbl in (self.use_lables):
            lbls[LABELS[lbl]] = True  # TODO: make 1-dimensional

        t1 = t[:, lbls.astype(np.bool)]
        t1 = t1[(X_mask.astype(np.bool))]

        self.y_history = y_history
        self.X_mask = X_mask
        self.lbls = lbls

        return X1, t1

    def apply(self, X, y_history=None):
        '''
        Apply this node's classifier on X using the prior information y_history 
        from earlier classifiers.
        
        @args:
        -X classifier input
        -y_history: prior information from earlier clsfrs.
        
        @return:
        y_history to use as prior input for further classification
        
        '''

        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((X.shape[0], len(self.lbls)))

        X_mask = np.ones((X.shape[0]))
        if self.include_classes is not None:
            for i, cat in enumerate(self.include_classes):
                X_mask = np.multiply(X_mask, (y_history[:, LABELS[cat]] > 0.5))

        X1 = np.copy(X[X_mask.astype(np.bool)])

        print(X1.shape)

        dy = self.clsfr.predict(X1)

        for n, i in enumerate(np.where(X_mask)[0]):
            y_history[i, :][self.lbls.astype(np.bool)[:]] = y_history[i, :][self.lbls.astype(np.bool)[:]] + dy[n,
                                                                                                            :]  # TODO: fix update rule!

        return y_history

    def apply_rec(self, X, y_history=None):
        '''
        Classify images recursively in the Node() tree
        '''

        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((X.shape[0], self.lbls.shape[0]))

        y_history = self.apply(X, y_history)

        for child in self.children:
            y_history = child.apply_rec(X, y_history)

        return y_history

    def train(self, X, t, y_history=None):
        '''
        Train this node's classifier
        
        @args:
        X: X training data
        t: training targets
        y_history: prior training data
        
        '''

        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((t.shape))

        X1, y1 = self.create_training_set(X, t, y_history)

        self.clsfr.fit(X1, y1,[])

    def train_rec(self, X, t, y_history=None):
        '''
        Train this clsfr and its children recursively
        '''

        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((t.shape))

        self.train(X, t, y_history)

        y_history = self.apply(X, y_history)

        for child in self.children:
            child.train_rec(X, t, y_history)

    def save(self, path):
        '''
        Save the model to path/self.name.h5
        
        '''
        self.clsfr.model.save(path + self.name + ".h5")
        print('Saved model to ' + path + self.name
              + '.h5')

    def load(self, path):
        '''
        Load the model from path/self.name.h5
        '''

        self.clsfr.model = keras.models.load_model(path + self.name + ".h5")

        print("Loaded model from disk")
