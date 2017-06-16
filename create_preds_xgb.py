import numpy as np
import pandas as pd
from tqdm import tqdm

import data as data
import data_hdf5 as d
from data_hdf5 import HDF_line_reader
from data_hdf5 import Validation_splitter

predictions_train = np.zeros((40479, 17))
predictions_test = np.zeros((61191, 17))
one_third = 13493
batch_size = 103

splitter = Validation_splitter('input/train.h5', 1.0 / 3.0)
print(splitter.num_folds)
print(splitter.fold_size)



print('Loading model ', 'my model')
# Give you clsfr a name
name = 'xxx'
# TODO however you access you clsfr
classifier = 'xxx'
# Image size you clsfr was trained on (beware, this script will make predictions on the jpg images,
# so if you trained on something else it does not make sense to use it)
img_size = 'xxx'




print('Create predictions train')
for j in tqdm(range(3)):
    splitter.next_fold()

    reader_train = HDF_line_reader('input/train.h5', load_rgb=False, img_size=img_size)
    tg = d.train_generator(reader=reader_train, splitter=splitter, batch_size=batch_size)

    # print(classifier.model.getShape())
    predictions_train[one_third * j:one_third * j + one_third, :] = classifier.predict_generator(tg,
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

    predictions_test[one_third * j:one_third * j + one_third, :] = classifier.predict_generator(test_g,
                                                                                                       one_third // batch_size)


pd.DataFrame(predictions_test,columns=data.labels).to_csv('ensemble/test_predictions_{}.csv'.format(name), index=False)