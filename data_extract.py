import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import wfdb
import numpy as np
from scipy.signal import find_peaks

def extract_features(record_path, annotation_path):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(annotation_path, 'atr')
    signals = record.p_signal

    #gets r peaks of the signal
    #works for both channels at once (data is not yet separated)
    r_peaks = annotation.sample     


    # Like the Xing paper, remove the first 10 and last 5 heartbeats to prevent 
    #effects of setting up/taking off the ECG leads
    r_peaks = r_peaks[10:-5]


    segments_channel1 = []
    segments_channel2 = []
    labels = []
    for r_peak in r_peaks:
        #segment separate heartbeats from channel 1
        start_idx_channel1 = r_peak - 100  #start 100 miliseconds before the r peak
        end_idx_channel1 = r_peak + 200    #end 200 miliseconds after the r peak
        segment_channel1 = signals[start_idx_channel1:end_idx_channel1, 0]

        #segment separate heartbeats from channel 2
        start_idx_channel2 = r_peak - 100  #same as channel 1
        end_idx_channel2 = r_peak + 200   
        segment_channel2 = signals[start_idx_channel2:end_idx_channel2, 1]

        #Translate MIT-BIH heartbeat annotation to AAMI heartbeat classification
        label_idx = np.where(annotation.sample == r_peak)[0]
        if label_idx.size > 0:
            label = annotation.symbol[label_idx[0]]
            if label in ['N', 'L', 'R', 'e', 'j']:
                labels.append('N')
            elif label in ['A', 'a', 'J', 'S']:
                labels.append('S')
            elif label in ['V', 'E']:
                labels.append('V')
            elif label in ['F', '/']:
                labels.append('F')
            else:
                labels.append('Q')  #unclassifiable
            segments_channel1.append(segment_channel1)
            segments_channel2.append(segment_channel2)

    return segments_channel1, segments_channel2, labels