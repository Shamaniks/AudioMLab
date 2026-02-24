import os
import librosa
import numpy as np

def prepare_data(data_dir, n_mfcc=13, max_len=32):
    X, y = [], []
    label_map = {'yes': 0, 'no': 1, 'unknown': 2}

    for label_name, label_idx in label_map.items():
        folder_path = os.path.join(data_dir, label_name)
        
        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.wav'): continue
            
            file_path = os.path.join(folder_path, file_name)
            audio, sr = librosa.load(file_path, sr=16000)
            
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]
            
            X.append(mfcc)
            y.append(label_idx)
            
    return np.array(X), np.array(y)
