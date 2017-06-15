import data as d
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

#test_tiff, test_filenames = d.get_all_test('input/test-tif-v2')
gen = d.get_test_generator_jpg('input/test-jpg', chunk_size=5000)
hdf = h5py.File('input/test.h5', 'w')

dt = h5py.special_dtype(vlen=bytes)

img, name = next(gen)

maxshape = (None,) + img.shape[1:]
images = hdf.create_dataset('imgs', shape=img.shape, maxshape=maxshape,
        chunks=(1000,*img.shape[1:]), dtype=img.dtype)

maxshape = (None,) + name.shape[1:]
ids = hdf.create_dataset("filenames", shape=name.shape, maxshape=maxshape,
        chunks=(1000,*name.shape[1:]), dtype=dt)

# Write the first chunk of rows
images[:] = img
ids[:] = name

row_count = len(img)
for img, name in tqdm(gen, desc='Create hdf5', total=61191):
    # Resize the dataset to accommodate the next chunk of rows
    images.resize(row_count + img.shape[0], axis=0)
    ids.resize(row_count + name.shape[0], axis=0)

    # Write the next chunk
    images[row_count:] = img
    ids[row_count:] = name

    # Increment the row count
    row_count += img.shape[0]

hdf.close()
