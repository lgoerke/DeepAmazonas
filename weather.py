import numpy as np
import pickle
import sys
import os
import pandas as pd
from tqdm import tqdm

import utils
import data_hdf5 as data
from data_hdf5 import Validation_splitter
from data_hdf5 import HDF_line_reader
from sklearn.metrics import fbeta_score

from keras.optimizers import SGD
from keras.layers import Input, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

from Classifiers.simple_net import SimpleNet
from Classifiers.densenet import DenseNet
from pathlib import Path
import h5py

from keras.callbacks import EarlyStopping
import pdb

def main(args):
    #size = 96
    size = 224
    #batch_size = 128
    batch_size = 3
    nb_epoch = 5
    optimizer = 'adam'
    val_split = 0.2
    N_CLASSES = 13
    N_SAMPLES = 40479
    N_TEST = 61191
    test_batch_size = 39

    exclude = ['cloudy', 'partly_cloudy', 'clear', 'haze']
    CONDITION = 0

    img_rows, img_cols = size, size # Resolution of inputs
    channel = 3


    labels = data.labels
    cross_val = True

    splitter = Validation_splitter('input/train.h5', val_split)
    reader = HDF_line_reader('input/train.h5', load_rgb = False, img_size=size)
    test_reader = HDF_line_reader('input/test.h5', load_rgb = False, img_size=size)
    
    classifier = DenseNet(img_rows, img_cols, batch_size=batch_size, nb_epoch=nb_epoch, color_type=channel, num_classes=N_CLASSES)
    model = Model(inputs=[classifier.model.input], outputs=[classifier.model.get_layer(name='relu5_blk').output])
    model.compile(optimizer='adam', loss='binary_crossentropy') 
    
    reader.test =True
    pre_gen_train = data.test_generator(reader,3)
    pre_gen_test = data.test_generator(test_reader,test_batch_size)
    
    remove = [data.LABELS[e] for e in exclude]
    rlabels = np.delete(reader.labels[:], remove, axis=1)

    hfile = Path('input/dens_predictions.h5')
    f = h5py.File(str(hfile), 'r')
    train_data = f['train_data']
    test_data = f['test_data'] 
    
    img_input = Input(( 7, 7, 2208))
    
    x_newfc = GlobalAveragePooling2D()(img_input)
    x = Dense(512,kernel_initializer="he_normal")(x_newfc)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x_newfc = Dropout(0.2)(x)
    x_newfc = Dense(N_CLASSES)(x_newfc)
    x_newfc = Activation('sigmoid')(x_newfc)

    model = Model(img_input, x_newfc)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
   
    train_data = train_data[:]
    result = np.zeros((N_TEST,N_CLASSES))
    while(splitter.next_fold() and cross_val):

        print('start training: ')
        val_idx = list(splitter.val_idx)
        val_idx = list(np.delete(val_idx, np.where(reader.labels[val_idx][:,remove[CONDITION]] == 1)[0], axis=0))
        val_idx.sort()


        t_idx = list(splitter.train_idx)
        t_idx = list(np.delete(t_idx, np.where(reader.labels[t_idx][:,remove[CONDITION]] == 1)[0], axis=0))
        t_idx.sort()
        
        
        val_data = train_data[val_idx]
        
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        
        model.fit(train_data[t_idx], rlabels[t_idx], validation_data=(val_data, rlabels[val_idx]),batch_size=128,epochs=10, callbacks=[early], verbose=1)
        
        print('validating')
        p_valid = model.predict(val_data, batch_size=1000, verbose=1)
        val_labels = rlabels[val_idx]
        loss = fbeta_score(val_labels, np.array(p_valid) > 0.2, beta=2, average='samples')
        print('validation loss: {}'.format(loss))

        print('save model:')
        model.save(os.path.join('models', 'dense_combined{:2.2f}'.format(3)))
        
        thres_opt = utils.optimise_f2_thresholds(val_labels, p_valid, n_classes=N_CLASSES) 
        print(thres_opt)
        
        p_test = model.predict(test_data,batch_size=1000, verbose=1)
        result += p_test
        
        #cross_val = False 
    
    
    result /= splitter.num_folds
    
    columns = [l for l in labels if l not in exclude]  
    
    train_preds = model.predict(train_data, batch_size=1000, verbose=1)
    train_results = pd.DataFrame(train_preds, columns = columns) 
    test_results = pd.DataFrame(result, columns = columns)
    
    train_results = pd.concat([pd.DataFrame({'image_names': reader.filenames}), train_results], axis=1)
    test_results = pd.concat([pd.DataFrame({'image_names': test_reader.filenames}), test_results], axis=1)
    
    train_results.to_csv('train_preds_dense_combined.csv')
    test_results.to_csv('test_preds_dense_combined.csv')

    result = pd.DataFrame(result, columns = columns)    
    
    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    df = pd.DataFrame(np.zeros((N_TEST,2)), columns=['image_name','tags'])
    df['image_name'] = test_reader.filenames
    df['tags'] = preds

    id = 2
    df.to_csv('submission_combined{}.csv'.format(id), index=False)
    f.close()
    
if __name__ == '__main__':
    main([])