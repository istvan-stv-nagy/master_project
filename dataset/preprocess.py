import numpy as np
import os
import matplotlib.pyplot as plt

from labeling.labeling_params import LabelingParams


def process_example(pano_path):
    if not os.path.exists(pano_path):
        return None

    pano = np.load(pano_path)
    return pano

def plot_channel(x):
    plt.figure()
    plt.plot(x)
    plt.xlim((0, 514))
    plt.show()


class InputLabeling:
    def __init__(self):
        self.coords = []
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.button == 1:
            self.coords += [event.xdata]
        elif event.button == 2:
            print("exit")
            plt.gcf().canvas.mpl_disconnect(self.cid)

    def run(self, channel):
        self.ax.plot(channel)
        plt.show()
        return self.coords

if __name__ == '__main__':
    for f in range(35, 36):
        path = r'E:\Storage\7 Master Thesis\dataset\curbstone_with_nan\pano' + str(f) + ".npy"
        pano = process_example(path)
        label = np.zeros((64, 514))
        for i in range(64):
            labeling = InputLabeling()
            coords = labeling.run(pano[i])
            for c in coords:
                label[i, int(c)] = 1
        np.save(LabelingParams.OUTPUT_ROOT + "\\pano_gt" + str(f) + ".npy", label)