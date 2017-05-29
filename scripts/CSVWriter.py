
import numpy as np
import pandas as pd

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
        self.yfull_valid = []
        self.yfull_labels = []
        self.yfull_thresholds = []

    def collect(self, valid_index, Y_valid, p_valid, thresholds):
        # Collect predictions and thresholds in arrays, to be called in main loop
        self.yfull_valid.append(p_valid)
        self.yfull_labels.append(Y_valid)
        self.yfull_thresholds.append(thresholds)
        self.valid_index = valid_index

    def to_csv(self, valid_index, file="runname"):
        # Write output arrays to .csv file, to be called after the main loop
        result_valid = np.array(self.yfull_valid[2])
        result_thresholds = np.array(self.yfull_thresholds[2])
        result_valid_labels = np.array(1*(result_valid>result_thresholds))
        ground_truth_valid = np.array(self.yfull_labels[2])

        df_pred_prob = pd.DataFrame(result_valid, index=valid_index, columns=self.labels)
        df_pred_lbl = pd.DataFrame(result_valid_labels, index=valid_index, columns=self.labels)
        df_ground_truth = pd.DataFrame(ground_truth_valid, index=valid_index, columns=self.labels)

        df_pred_prob.to_csv(file+'_pred_prob.csv')
        df_pred_lbl.to_csv(file+'_pred_lbl.csv')
        df_ground_truth.to_csv(file+'_ground_truth.csv')