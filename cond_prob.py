import numpy as np
import pickle
import sys
import os
import pandas as pd
from tqdm import tqdm
import pdb
import data_hdf5 as d


def cond_prob(args):
    labels = ['blow_down',
              'bare_ground',
              'conventional_mine',
              'blooming',
              'cultivation',
              'artisinal_mine',
              'haze',
              'primary',
              'slash_burn',
              'habitation',
              'clear',
              'road',
              'selective_logging',
              'partly_cloudy',
              'agriculture',
              'water',
              'cloudy']

    # Get targets from training data hdf5 file
    reader = d.HDF_line_reader('input/train.h5', load_rgb=False)
    _, targets, file_ids = d.get_all_train(reader=reader)
    LABELS = d.LABELS
    REV_LABELS = d.REV_LABELS

    df = pd.DataFrame(np.array(targets), columns=labels)

    # Create 2dim co occurence matrix by matrix multiplication
    df_asint = df.astype(int)
    coocc2 = df_asint.T.dot(df_asint)

    # Create 3dim co occurence matrix by counting when sums of cols is 3
    coocc3 = np.zeros((len(LABELS), len(LABELS), len(LABELS)))
    for i in tqdm(range(len(LABELS))):
        for j in range(len(LABELS)):
            for k in range(len(LABELS)):
                if not (i == j or j == k or k == i):
                    cols = df[[REV_LABELS[i], REV_LABELS[j], REV_LABELS[k]]]
                    sums = np.sum(cols.as_matrix(), axis=1)
                    coocc3[i, j, k] = np.sum(sums == 3)

    # Make probability from counts
    # P(A and B)
    c2 = coocc2.as_matrix()
    np.fill_diagonal(c2, 0)
    s2 = np.sum(c2)
    cooccp2 = coocc2.as_matrix() / s2

    # Make probability from counts
    # P(A and B and C)
    s3 = np.sum(coocc3)
    cooccp3 = coocc3 / s3

    # Count individual probabilities
    # P(A)
    summat = df_asint.values.sum(axis=0)
    summatp = summat / np.sum(summat)

    # Init conditional probability arrays
    condp2 = np.zeros((len(LABELS), len(LABELS)))
    condp3 = np.zeros((len(LABELS), len(LABELS), len(LABELS)))

    # For all A
    for i in tqdm(range(len(LABELS))):
        # For all B
        for j in range(len(LABELS)):
            # P(A|B) = P(A and B) / P(B)
            condp2[i, j] = cooccp2[i, j] / summatp[j]
            # For all C
            for k in range(len(LABELS)):
                # P(A|B,C) = P(A and B and C) / P(B and C)
                if cooccp2[j, k] == 0:
                    condp3[i, j, k] = 0
                else:
                    condp3[i, j, k] = cooccp3[i, j, k] / cooccp2[j, k]


    print('haldlaslf')
    # Put in pickle
    #pickle.dump(condp2, open('input/condp2.pkl', 'wb'))
    #pickle.dump(condp3, open('input/condp3.pkl', 'wb'))


if __name__ == '__main__':
    cond_prob([])
