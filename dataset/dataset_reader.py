import numpy as np
import os


def read_curbstone_dataset():
    path = r'E:\Storage\7 Master Thesis\dataset\curbstone'
    X = np.zeros(514)
    y = np.zeros(2)
    for i in range(20, 60):
        pano_path = os.path.join(path, "pano" + str(i) + ".npy")
        label_path = os.path.join(path, "curb" + str(i) + ".npy")
        if os.path.exists(pano_path) and os.path.exists(label_path):
            pano = np.load(pano_path)
            label = np.load(label_path)
            for channel in range(64):
                channel_pano = pano[channel]
                channel_label = label[channel]
                invalid_count = len(np.where(channel_pano == -1)[0])
                breakpoint_list = np.where(channel_label == 1)[0]
                breakpoints_count = len(breakpoint_list)
                if (invalid_count < 50) and (breakpoints_count == 2):
                    X = np.vstack((X, channel_pano))
                    y = np.vstack((y, breakpoint_list))
    X = X[1:]
    y = y[1:]
    return X.astype(np.float32), y.astype(np.float32)
