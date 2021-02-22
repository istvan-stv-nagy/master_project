import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np


def plot_images(images, cmap='jet'):
    fig, axs = plt.subplots(len(images))
    for i, image in enumerate(images):
        axs[i].imshow(image, cmap=cmap)


def plot_point_cloud(point_cloud):
    x = point_cloud[0, :]
    y = point_cloud[1, :]
    z = point_cloud[2, :]
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mlab.points3d(x, y, z, -y, mode="point", colormap="spectral", figure=fig)


def plot_signals(signals):
    fig, axs = plt.subplots(len(signals))
    for i, signal in enumerate(signals):
        axs[i].plot(np.arange(len(signal)), signal)
