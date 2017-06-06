
import numpy as np
import pandas as pd

'''
Example code after main training loop:
csvWriter = CSVWriter()
for train_index, valid_index in kf:
    X_valid = x_train[valid_index]
    Y_valid = y_train[valid_index]
    pred_prob = model.predict(X_valid, batch_size=128, verbose=2)
    pred_label = np.array(1 * (pred_prob > thres_opt))
    csvWriter.collect(valid_index, Y_valid, pred_prob, pred_label)
csvWriter.to_csv(file="SimpleCNNtiff")
'''

'''
Class to export validation predictions to .csv file for result analysis notebook Analysis.ipynb.
'''
class CSVWriter:
    def __init__(self):
    # Define set of labels and initialize output arrays, to be called before main loop
        self.labels = ['blow_down',
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
        self.imgs = []
        self.ground_truths = []
        self.pred_probs = []
        self.pred_labels = []

    def collect(self, img, ground_truth, pred_prob, pred_label):
        # Collect predictions and thresholds in lists of arrays
        self.imgs.append(img)
        self.ground_truths.append(ground_truth)
        self.pred_probs.append(pred_prob)
        self.pred_labels.append(pred_label)

    def to_csv(self, file="runname"):
        # Write output arrays to .csv file, to be called after the main loop
        array_imgs = np.concatenate(self.imgs, axis=0)
        array_ground_truths = np.concatenate(self.ground_truths, axis=0)
        array_pred_probs = np.concatenate(self.pred_probs, axis=0)
        array_pred_labels = np.concatenate(self.pred_labels, axis=0)

        df_ground_truths = pd.DataFrame(array_ground_truths, index=array_imgs, columns=self.labels)
        df_pred_probs = pd.DataFrame(array_pred_probs, index=array_imgs, columns=self.labels)
        df_pred_labels = pd.DataFrame(array_pred_labels, index=array_imgs, columns=self.labels)

        df_ground_truths.to_csv(file + '_ground_truths.csv')
        df_pred_probs.to_csv(file+'_pred_probs.csv')
        df_pred_labels.to_csv(file+'_pred_lbls.csv')
