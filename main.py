import numpy as np
import pickle
import sys
import os
import pandas as pd
from tqdm import tqdm
import pdb

import utils
import data
from data import Validation_splitter
from data import CSV_line_reader

from Classifiers.simple_net import SimpleNet


def main(args):
    size = 64
    batch_size = 96
    nb_epoch = 2
    optimizer = 'adadelta'
    val_split = 0.2
    N_CLASSES = 17
    N_SAMPLES = 40478

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

    classifier = SimpleNet((size, size, 3), n_classes=N_CLASSES, nb_epoch=nb_epoch, batch_size=batch_size,
                           optimizer=optimizer)

    splitter = Validation_splitter('input/train_v2.csv', val_split)
    splitter.next_fold()

    reader = CSV_line_reader('input/train_v2.csv')
    tg = data.train_generator('input/train-tif-v2', reader, splitter, batch_size, img_size=size, load_rgb=True)
    vg = data.val_generator('input/train-tif-v2', reader, splitter, batch_size, img_size=size, load_rgb=True)

    print('start training: ')
    classifier.fit(tg, vg, ((1 - val_split) * N_SAMPLES, val_split * N_SAMPLES))

    print('validating')
    val_data, val_labels = data.get_all_val('input/train-tif-v2', reader, splitter, img_size=size, load_rgb=True)
    pdb.set_trace()
    p_valid = classifier.predict(val_data)

    print('validation loss: {}'.format(fbeta_score(val_labels, np.array(p_valid) > 0.2, beta=2, average='samples')))

    print('save model:')
    classifier.model.save(os.path.join('models', 'simple_net_{}'.format(loss)))

    thres_opt = utils.optimise_f2_thresholds(val_labels, p_valid)

    test_data = data.get_all_test('input/test-tif-v2', img_size=size, load_rgb=True)

    p_test = classifier.predict(test_data)

    result = pd.DataFrame(p_test, columns=labels)

    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    result['tags'] = preds

    result.to_csv('submission_keras.csv', index=False)


if __name__ == '__main__':
    main([])
