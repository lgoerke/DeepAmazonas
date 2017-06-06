import data as d
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm


reader = d.CSV_line_reader('input/train_v2.csv')
splitter = d.Validation_splitter('input/train_v2.csv', 0.2)
splitter.next_fold()

#test_tiff, test_filenames = d.get_all_test('input/test-tif-v2')
gen = d.get_train_generator('input/train-tif-v2', reader, splitter, chunk_size=5000)
hdf = h5py.File('input/train.h5', 'w')

dt = h5py.special_dtype(vlen=bytes)

img, label, name = next(gen)
np.random.shuffle(img)
np.random.shuffle(label)
np.random.shuffle(name)

maxshape = (None,) + img.shape[1:]
images = hdf.create_dataset('imgs', shape=img.shape, maxshape=maxshape,
        chunks=(1000,*img.shape[1:]), dtype=img.dtype)

maxshape = (None,) + label.shape[1:]
labels = hdf.create_dataset("labels", shape=label.shape, maxshape=maxshape,
        chunks=(1000,*label.shape[1:]), dtype=label.dtype)

maxshape = (None,) + name.shape[1:]
ids = hdf.create_dataset("filenames", shape=name.shape, maxshape=maxshape,
        chunks=(1000,*name.shape[1:]), dtype=dt)

# Write the first chunk of rows
images[:] = img
labels[:] = label
ids[:] = name
row_count = len(img)
for img, label, name in tqdm(gen, desc='Create hdf5'):
    # Resize the dataset to accommodate the next chunk of rows
    np.random.shuffle(img)
    np.random.shuffle(label)
    np.random.shuffle(name)
    
    images.resize(row_count + img.shape[0], axis=0)
    labels.resize(row_count + label.shape[0], axis=0)
    ids.resize(row_count + name.shape[0], axis=0)

    # Write the next chunk
    images[row_count:] = img
    labels[row_count:] = label
    ids[row_count:] = name

    # Increment the row count
    row_count += img.shape[0]

hdf.close()
