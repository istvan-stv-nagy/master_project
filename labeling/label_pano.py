from datastructures.pano_image import PanoImage
import matplotlib.pyplot as plt
import numpy as np



class LabelPanoTool:
    def __init__(self):
        self.min_pano_height = -0.2
        self.max_pano_height = 0.75
        self.pts = []
        self.lines = []
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.button == 1:
            self.pts += [(event.xdata, event.ydata)]
        elif event.button == 3:
            self.lines += [self.pts]
            self.pts = []
        elif event.button == 2:
            print("exit")
            plt.gcf().canvas.mpl_disconnect(self.cid)
        for pts in self.lines:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.gca().plot(xs, ys, 'r-')
            plt.draw()
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        plt.gca().plot(xs, ys, 'k-')
        plt.draw()

    def label(self, pano_image: PanoImage):
        pano = np.clip((pano_image.img - self.min_pano_height) / (self.max_pano_height - self.min_pano_height), 0.0, 1.0) * 255
        self.ax.imshow(pano, cmap='jet')
        plt.show()
        return self.lines
