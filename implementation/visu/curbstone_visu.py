from typing import List

from implementation.curbstone.edge_height_profile import EdgeHeightProfile
from implementation.curbstone.edge_line import EdgeLine
import matplotlib.pyplot as plt


class CurbstoneVisu:
    def __init__(self):
        pass

    def show(self, image, prediction, edges: List[EdgeLine], edge_profiles: List[EdgeHeightProfile]):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(image)
        axs[0, 1].imshow(prediction)
        for edge in edges:
            for point in edge.points:
                axs[0, 1].plot(point[1], point[0], 'w.')

        for i, profile in enumerate(edge_profiles):
            axs[1, i].plot(profile.elevations)
            axs[1, i].set_ylim(-0.5, 0.5)
