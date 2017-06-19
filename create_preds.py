import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
import data as data
import data_hdf5 as d
from data_hdf5 import HDF_line_reader
from data_hdf5 import Validation_splitter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

predictions_train = np.zeros((40479, 17))
predictions_test = np.zeros((61191, 17))
one_third = 13493
batch_size = 103

splitter = Validation_splitter('input/train.h5', 1.0 / 3.0)
print(splitter.num_folds)
print(splitter.fold_size)



print('Loading model ', 'my model')
# Give you clsfr a name
name = 'ricci'
# TODO however you access you clsfr



model = Sequential()
model.add(BatchNormalization(input_shape=(64, 64,3)))
model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

# so if you trained on something else it does not make sense to use it)
filepath = 'ensemble/weights_kfold_5_riccardo.h5'
model.load_weights(filepath)

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
img_size = 64




print('Create predictions train')
for j in tqdm(range(3)):
    splitter.next_fold()

    reader_train = HDF_line_reader('input/train.h5', load_rgb=False, img_size=img_size)
    tg = d.train_generator(reader=reader_train, splitter=splitter, batch_size=batch_size)

    # print(classifier.model.getShape())
    predictions_train[one_third * j:one_third * j + one_third, :] = model.predict_generator(tg,
                                                                                                    one_third // batch_size)

pd.DataFrame(predictions_train,columns=data.labels).to_csv('ensemble/train_predictions_{}.csv'.format(name), index=False)

one_third = 20397
batch_size = 523

splitter = Validation_splitter('input/test.h5', 1.0 / 3.0)
print(splitter.num_folds)
print(splitter.fold_size)

print('Create predictions test')
for j in tqdm(range(3)):
    splitter.next_fold()

    reader_test = HDF_line_reader('input/test.h5', load_rgb=False, img_size=img_size)
    test_g = d.test_generator(reader=reader_test, batch_size=batch_size)

    predictions_test[one_third * j:one_third * j + one_third, :] = model.predict_generator(test_g,
                                                                                                       one_third // batch_size)


pd.DataFrame(predictions_test,columns=data.labels).to_csv('ensemble/test_predictions_{}.csv'.format(name), index=False)
