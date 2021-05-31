from implementation.datastructures.pano_image import PanoImage
import matplotlib.pyplot as plt
import numpy as np
from labeling.label_output import LabelOutput
from labeling.labeling_params import LabelingParams
from utils.line import Line


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

    def label(self, frame_count, pano_image: PanoImage):
        pano = np.clip((pano_image.img - self.min_pano_height) / (self.max_pano_height - self.min_pano_height), 0.0, 1.0) * 255
        self.ax.imshow(pano, cmap='jet')
        delimitations = self.postprocess_lines()
        label_image = self.generate_label(delimitations)
        output = LabelOutput(pano_image=pano_image, label_image=label_image)
        self.save(frame_count, output)
        return output

    def postprocess_lines(self):
        lines = []
        for points in self.lines:
            for i in range(len(points) - 1):
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = points[i + 1][0]
                y2 = points[i + 1][1]
                lines += [Line(x1, y1, x2, y2)]

        results = []
        for y in range(64):
            coords = []
            for line in lines:
                x = line.get_x(y)
                if line.contains(x, y):
                    coords += [x]
            results += [coords]
        return results

    def generate_label(self, delimitations):
        label = np.zeros((64, 514))
        print(len(delimitations))
        for i in range(len(delimitations)):
            for j in range(len(delimitations[i])):
                label[i, delimitations[i][j]] = 1
        return label

    def save(self, frame_count, label_output: LabelOutput):
        np.save(LabelingParams.OUTPUT_ROOT + "\\pano" + str(frame_count) + ".npy", label_output.data().img)
        np.save(LabelingParams.OUTPUT_ROOT + "\\curb" + str(frame_count) + ".npy", label_output.label())
