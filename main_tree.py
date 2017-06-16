from tree_test import  Node
from matplotlib import pyplot as plt
import data_hdf5_tree as data
from data_hdf5_tree import Validation_splitter
from data_hdf5_tree import HDF_line_reader
import numpy as np
import pickle
import sys
import os
import pandas as pd
from tqdm import tqdm
import pdb

import utils
import data_hdf5 as data
from data_hdf5 import Validation_splitter
from data_hdf5 import HDF_line_reader
from sklearn.metrics import fbeta_score

loc = '/media/sebastian/7B4861FD6D0F6AA2/input/'

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
          
size = 224          #image size
batch_size = 40     
nb_epoch = 5
optimizer = 'adam'
val_split = 0.2
N_CLASSES = 17
N_SAMPLES = 40479
N_TEST = 61191
test_batch_size = 20

def add_class(names ):
    for name in names:
        LABELS[name] = len(LABELS.keys())

def load_riccardo():
    weather = np.load(loc+'train_pred.npy')
    cls = ['clear','partly_cloudy','cloudy','haze']

    y = np.zeros((weather.shape[0],len(LABELS.keys())))
    for i, c in enumerate(cls):    
        y[:,LABELS[c]] = weather[:,i]
    return y

def invest_Tree():
    '''
     |       Little test tree... 
     |
    / \
    | |
    '''

    #labels = list(data.LABELS.keys())

    
    sp_3_1 = Validation_splitter((loc+'train.h5'), val_split)
    sp_3_0 = Validation_splitter((loc+'train.h5'), val_split)
    sp_2_1 = Validation_splitter((loc+'train.h5'), val_split)
    sp_2_0 = Validation_splitter((loc+'train.h5'), val_split)
    sp_1 = Validation_splitter((loc+'train.h5'), val_split)
    sp_0 = Validation_splitter((loc+'train.h5'), val_split)
    

    splitters = [sp_3_1,sp_3_0,sp_2_1,sp_2_0  ,sp_1 , sp_0]
    
    train_reader = HDF_line_reader((loc+'train.h5'), load_rgb = False, img_size=size)
    val_reader = HDF_line_reader((loc+'train.h5'), load_rgb = False, img_size=size)
    test_reader = HDF_line_reader((loc+'test.h5'), load_rgb = False, img_size=size)
    
    new_cols = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation','water','road' ],
                        'forest_phenomena': ['blow_down', 'blooming', 'cultivation', 'slash_burn', 'selective_logging'] }
                        
    add_class(new_cols.keys())
    
    node_3_1_phenomena_investigator = Node('node_3_1_phenomena_investigator', [],
                    ['blow_down', 'blooming', 'slash_burn', 'selective_logging'],
                    ['forest_phenomena'],add_labels = {'forest_phenomena': ['blow_down', 'blooming', 'cultivation','slash_burn', 'selective_logging']},
                    clsfr = None,  size = size, batch_size = batch_size,
                        validation_splitter = sp_3_1, train_reader = train_reader,val_reader = val_reader,  test_reader = test_reader, labels = LABELS)

    node_2_1_forestphenomena = Node('node_2_1_forestphenomena',
                    [node_3_1_phenomena_investigator],
                    ['forest_phenomena'],
                    ['primary'], size = size, batch_size = batch_size,
                    add_labels = {'forest_phenomena': ['blow_down', 'blooming', 'slash_burn', 'selective_logging']},clsfr = None,
                        validation_splitter = sp_2_1, train_reader = train_reader,val_reader = val_reader,  test_reader = test_reader, labels = LABELS)

    node_3_0_infra_investigator = Node('node_3_0_infra_investigator', [],
                    ['conventional_mine', 'artisinal_mine','slash_burn','bare_ground','habitation','water','road'  ],['infrastructure'],  size = size,                      
                    batch_size = batch_size,
                    add_labels = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation','water','road' ]},
                    clsfr = None,
                        validation_splitter = sp_3_0, train_reader = train_reader,val_reader = val_reader,  test_reader = test_reader, labels = LABELS)

    node_2_0_infrastructure = Node('node_2_0_infrastructure',
                    [node_3_0_infra_investigator],
                    ['infrastructure'],['haze', 'partly_cloudy', 'clear'],
                    add_labels = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation','water','road' ]},
                    clsfr = None,  size = size,  batch_size = batch_size,
                        validation_splitter = sp_2_0, train_reader = train_reader,val_reader = val_reader,  test_reader = test_reader, labels = LABELS)

    node_1_land = Node('node_1_land',
                    [node_2_0_infrastructure, node_2_1_forestphenomena],
                    ['agriculture', 'habitation',  'primary', 'cultivation', 'bare_ground'],
                    ['haze', 'partly_cloudy', 'clear'],clsfr = None,  size = size,  batch_size = batch_size,
                        validation_splitter =     sp_1, train_reader = train_reader,val_reader = val_reader,  test_reader = test_reader,  labels = LABELS)

    node_0_weather = Node('node_0_weather', [node_1_land], ['cloudy', 'haze', 'partly_cloudy', 'clear'],clsfr = None,  size = size,
                         batch_size = batch_size,
                        validation_splitter = sp_0, train_reader = train_reader, val_reader = val_reader, test_reader = test_reader,  labels = LABELS)

    return node_0_weather,splitters

def weather_Tree():
    '''
    Tree that trains on all weather predictions apart.
    '''
    

    #labels = list(data.LABELS.keys())


    sp_haze = Validation_splitter((loc+'train.h5'), 0.3)
    sp_clear = Validation_splitter((loc+'train.h5'), val_split)
    sp_partly_cloudy = Validation_splitter((loc+'train.h5'), val_split)

    splitters = [sp_clear,sp_haze,sp_partly_cloudy]

    train_reader = HDF_line_reader((loc+'train.h5'), load_rgb = False, img_size=size)
    val_reader = HDF_line_reader((loc+'train.h5'), load_rgb = False, img_size=size)
    test_reader = HDF_line_reader((loc+'test.h5'), load_rgb = False, img_size=size)
    
    node_1_haze = Node('node_1_haze',
                    [],
                    ['blow_down','bare_ground','conventional_mine','blooming','cultivation',
          'artisinal_mine','primary','slash_burn','habitation','road','selective_logging',
          'agriculture','water'],
                    ['haze'],clsfr = None,
                        validation_splitter = sp_haze, train_reader = train_reader, val_reader = val_reader, test_reader = test_reader)
    
    node_2_clear = Node('node_2_clear',
                    [],
                    ['blow_down','bare_ground','conventional_mine','blooming','cultivation',
          'artisinal_mine','primary','slash_burn','habitation','road','selective_logging',
          'agriculture','water'],
                    ['clear'],clsfr = None,
                        validation_splitter = sp_clear, train_reader = train_reader, val_reader = val_reader, test_reader = test_reader)
                        
    node_3_partly_cloudy = Node('node_3_partly_cloudy',
                    [],
                    ['blow_down','bare_ground','conventional_mine','blooming','cultivation',
          'artisinal_mine','primary','slash_burn','habitation','road','selective_logging',
          'agriculture','water'],
                    ['partly_cloudy'],clsfr = None,
                        validation_splitter = sp_partly_cloudy, train_reader = train_reader, val_reader = val_reader, test_reader = test_reader)
                                  
    
    node_0_weather = Node('node_0_weather', [node_2_clear, node_1_haze,node_3_partly_cloudy], ['cloudy', 'haze', 'partly_cloudy', 'clear'],clsfr = None,
                        validation_splitter = None, train_reader = train_reader, val_reader = val_reader, test_reader = test_reader)

    return node_0_weather,splitters


def main(args):


    img_rows, img_cols = size, size  # Resolution of inputs
    channel = 3

    labels = list(data.LABELS.keys())
    cross_val = True
    y = load_riccardo()

    n,splitters = invest_Tree()

    val_reader = HDF_line_reader((loc+'train.h5'), load_rgb=False, img_size=size)
    test_reader = HDF_line_reader((loc+'test.h5'), load_rgb=False, img_size=size)

    result = np.zeros((N_TEST, N_CLASSES))

    while (splitters[0].next_fold() and cross_val):

        for i in range(len(splitters)-1):
            splitters[i+1].next_fold()

        print('start training: ')

        n.train_rec(True)
        print('validating')
        p_valid = n.apply_rec(validation = True)[:,0:16]#.predict(vg, np.ceil(len(splitter.val_idx) / batch_size))
        p_valid = p_valid[:len(splitters[0].val_idx)]
        idx = list(splitters[0].val_idx)
        idx.sort()
        val_labels = val_reader.labels[idx]

        loss = fbeta_score(val_labels, np.array(p_valid) > 0.2, beta=2, average='samples')
        print('validation loss: {}'.format(loss))

        print('save model:')

        thres_opt = utils.optimise_f2_thresholds(val_labels, p_valid)

        p_test = n.apply_rec()[:,0:16]
        result += p_test[:N_TEST]

        cross_val = False

    result /= splitters[0].num_folds
    result = pd.DataFrame(result, columns=labels)

    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    df = pd.DataFrame(np.zeros((N_TEST, 2)), columns=['image_name', 'tags'])
    df['image_name'] = test_reader.filenames
    df['tags'] = preds

    id = 0
    df.to_csv('submission{}.csv'.format(id), index=False)



main([])
