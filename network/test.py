import torch

from network.fcnn import FullyConnectedNeuralNetwork
from run.runnable import Runnable
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    model = FullyConnectedNeuralNetwork()
    model.load_state_dict(torch.load(r'E:\Storage\7 Master Thesis\results\network\model'))
    model.eval()
    runnable = Runnable()
    pano = runnable.run(2)
    res = model(torch.from_numpy(pano.img.astype(np.float32)))
    plt.figure()
    img = np.clip((pano.img + 0.2) / (0.75 + 0.2), 0.0, 1.0) * 255
    plt.imshow(img, cmap='jet')
    pts = res.detach().numpy()
    for i, p in enumerate(pts):
        if 0 <= p[0] <= 514 and 0 <= p[1] <= 514:
            plt.plot(p[0], i, 'w*')
            plt.plot(p[1], i, 'w*')
    plt.show()