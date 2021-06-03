import numpy as np
import matplotlib.pyplot as plt


class PixelMetrics:
    def __init__(self, precision, recall, f_measures, accuracy, fpr, fnr):
        self.precision = precision
        self.recall = recall
        self.f_measures = f_measures
        self.accuracy = accuracy
        self.fpr = fpr
        self.fnr = fnr


class PixelEvaluation:
    def __init__(self):
        pass

    def run(self, prediction, ground_truth):
        tp = np.sum(prediction * ground_truth)
        fp = np.sum((prediction != True) & (ground_truth == False))
        fn = np.sum((prediction == False) & (ground_truth == True))
        tn = np.sum((prediction == False) & (ground_truth == False))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_measures = [(1 + beta*beta) * (precision * recall) / (beta*beta*precision + recall) for beta in range(1, 10)]
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        print(tp, fp, precision, recall, f_measures[0], accuracy, fpr, fnr)
        return PixelMetrics(
            precision=precision,
            recall=recall,
            f_measures=f_measures,
            accuracy=accuracy,
            fpr=fpr,
            fnr=fnr
        )

if __name__ == '__main__':
    pred = np.random.randn(100, 100) > 0.5
    gt = np.random.randn(100, 100) > 0.5
    eval = PixelEvaluation()
    eval.run(pred, gt)
    fig, axs = plt.subplots(2)
    axs[0].imshow(gt)
    axs[1].imshow(pred)
    plt.show()
    plt.close()
