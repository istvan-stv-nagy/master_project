import matplotlib.pyplot as plt


class EvaluationVisu:
    def __init__(self):
        pass

    def show(self, gt_image, prediction_image):
        fig, axs = plt.subplots(2)
        axs[0].imshow(gt_image)
        axs[1].imshow(prediction_image)
