import numpy as np
import pandas as pd
#from Classifiers.simple_net import SimpleNet
from skimage import io as skio
import keras
import cv2
#from matplotlib import pyplot as plt

import pickle
import sys
import os
from tqdm import tqdm
import utils
import data_hdf5_tree as data
from data_hdf5_tree import Validation_splitter
from data_hdf5_tree import HDF_line_reader
from sklearn.metrics import fbeta_score

from Classifiers.simpler_net import SimpleNet

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


def test_Tree():
    '''
     |       Little test tree... 
     |
    / \
    | |
    '''

    size = 64
    batch_size = 96
    nb_epoch = 1
    optimizer = 'adadelta'
    val_split = 0.2
    N_CLASSES = 17
    N_SAMPLES = 40479
    N_TEST = 61191
    test_batch_size = 9

    #labels = list(data.LABELS.keys())


    sp = Validation_splitter('/media/sebastian/7B4861FD6D0F6AA2/train.h5', val_split)
    train_reader = HDF_line_reader('/media/sebastian/7B4861FD6D0F6AA2/train.h5', load_rgb = False, img_size=size)
    test_reader = HDF_line_reader('/media/sebastian/7B4861FD6D0F6AA2/train.h5', load_rgb = False, img_size=size)
    
    new_cols = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation' ],
                        'forest_phenomena': ['blow_down', 'blooming', 'cultivation', 'slash_burn', 'selective_logging'] }
                        
    add_class(new_cols.keys())
    
    node_3_1_phenomena_investigator = Node('node_3_1_phenomena_investigator', [],
                    ['blow_down', 'blooming', 'slash_burn', 'selective_logging'],
                    ['forest_phenomena'],add_labels = {'forest_phenomena': ['blow_down', 'blooming', 'cultivation','slash_burn', 'selective_logging']},
                    clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_2_1_forestphenomena = Node('node_2_1_forestphenomena',
                    [node_3_1_phenomena_investigator],
                    ['forest_phenomena'],
                    ['primary'],
                    add_labels = {'forest_phenomena': ['blow_down', 'blooming', 'slash_burn', 'selective_logging']},clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_3_0_infra_investigator = Node('node_3_0_infra_investigator', [],
                    ['conventional_mine', 'artisinal_mine', 'road','slash_burn','bare_ground','habitation' ],['infrastructure'],
                    add_labels = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation' ]},clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_2_0_infrastructure = Node('node_2_0_infrastructure',
                    [node_3_0_infra_investigator],
                    ['infrastructure'],['haze', 'partly_cloudy', 'clear'],
                    add_labels = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation' ]},
                    clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_1_land = Node('node_1_land',
                    [node_2_0_infrastructure, node_2_1_forestphenomena],
                    ['agriculture', 'habitation', 'water', 'primary', 'cultivation', 'bare_ground'],
                    ['haze', 'partly_cloudy', 'clear'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_0_weather = Node('node_0_weather', [node_1_land], ['cloudy', 'haze', 'partly_cloudy', 'clear'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    return node_0_weather


def add_class(names ):
    for name in names:
        LABELS[name] = len(LABELS.keys())


class Node():
    '''One node in a decision tree, which creates one clsfr instance and can create a dataset for it.'''

    def __init__(self, name, children, use_labels, include_classes=None, clsfr=None,
                    validation_splitter = None, train_reader = None, test_reader = None, add_labels = [],
                    size = 64, batch_size = 96, nb_epoch = 1,optimizer = 'adadelta',val_split = 0.3, 
                     N_SAMPLES = 40479, N_TEST = 61191,test_batch_size = 9 ):
        '''
        @args:
        -name: Name for this node
        -children: array of Node()s that receive their input from this Node.
        -use_labels: Array of labels to include ['forest','clouds']
        -include_classes: Images to use when classified as 1 already, empty = all.
        
        '''
        self.name = name
        self.X_mask = None
        self.y_history = None
        self.use_labels = use_labels
        self.lbls = None
        self.children = children
        self.include_classes = include_classes
        self.add_labels = add_labels
        self.validation_splitter = validation_splitter
        self.train_reader = train_reader
        self.test_reader = test_reader
        self.size = size
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.val_split = val_split
        self.N_SAMPLES = N_SAMPLES
        self.N_TEST = N_TEST
        self.test_batch_size = test_batch_size
        
        if clsfr is None:
            self.clsfr = self.__init_clsfr__()
        else:
            self.clsfr = clsfr

    def __init_clsfr__(self, size=None):
        '''Create a new classifier object bound to this node'''

        return SimpleNet((self.size,self.size,4), n_classes=len(self.use_labels), nb_epoch = self.nb_epoch, batch_size=self.batch_size, optimizer=self.optimizer)

    def apply(self, y_history=None, test_batch_size = 20):
        '''
        Apply this node's classifier on X using the prior information y_history 
        from earlier classifiers.
        
        @args:
        -X classifier input
        -y_history: prior information from earlier clsfrs.
        
        @return:
        y_history to use as prior input for further classification
        
        '''
        
        if self.validation_splitter is not None:
            if y_history is None:
                print('Created new y-matrix')
                y_history = np.zeros((self.N_TEST,len(LABELS.keys())))
    
            X_mask = np.ones((self.N_TEST))
            if self.include_classes:
                X_mask[:] = 0
                for cls in (self.include_classes):
                    X_mask = np.logical_or(y_history[:,LABELS[cls]]>0.3,X_mask) 
    
    
            test_gen = data.test_generator(self.test_reader, test_batch_size)
            p_test = self.clsfr.predict(test_gen, self.N_TEST // test_batch_size)
    
            
            if self.use_labels:
                lbls = np.zeros((len(LABELS.keys())))    
                for lbl in (self.use_labels):
                    lbls[LABELS[lbl]] = True
            else:
                lbls = np.ones((len(LABELS.keys()))) 
        
            p_test = np.multiply(p_test,np.repeat(np.expand_dims(X_mask,axis = 1),len(self.use_labels),axis=1))
    
            for n, i in enumerate(np.where(X_mask)[0]):
                y_history[i, :][lbls.astype(np.bool)[:]] = y_history[i, :][lbls.astype(np.bool)[:]] + p_test[n,:] 
            


        return y_history

    def apply_rec(self, y_history=None):
        '''
        Classify images recursively in the Node() tree
        '''

        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((self.N_TEST,len(LABELS.keys())))

        y_history = self.apply(y_history)

        for child in self.children:
            y_history = child.apply_rec(y_history)

        return y_history

    def train(self):
        '''
        Train this node's classifier
        
        '''
        if self.validation_splitter is not None:
            
            tg = data.train_generator(self.train_reader, self.validation_splitter, self.batch_size,  
                                    use_labels =self.use_labels , new_columns= self.add_labels,included_columns = self.include_classes )
            vg = data.val_generator(self.train_reader, self.validation_splitter, self.batch_size,  
                                    use_labels=self.use_labels , new_columns= self.add_labels,included_columns = self.include_classes )
            
            print('training ',self.name, ' on ', self.use_labels)             
            self.clsfr.fit(tg, vg, ((1-self.val_split) * self.N_SAMPLES, self.val_split * self.N_SAMPLES))
        
        

    def train_rec(self, save_clf = False):
        '''
        Train this clsfr and its children recursively
        '''

        self.train()
        
        if save_clf:
            self.save('')
        
        for child in self.children:
            child.train_rec(save_clf)
            
        

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
