import numpy as np

from implementation.curbstone.edge_line import EdgeLine


def extract_road_edges(prediction: np.ndarray):
    #append zeros to prediction left and right margins
    prediction_padded = np.c_[np.zeros(len(prediction)), np.c_[prediction, np.zeros(len(prediction))]]

    # find left margins
    left_points = []
    right_points = []

    diffs = np.diff(prediction_padded)
    for i in range(len(diffs)):
        row = diffs[i]
        find_left = np.where(row == 1)[0]
        find_right = np.where(row == -1)[0]
        if len(find_left) >= 1 and len(find_right) >= 1:
            left_points.append((i, find_left[0]))
            right_points.append((i, find_right[-1] - 1))
        elif len(find_left) == 1:
            col = find_left[0] - 1
            if col < len(prediction.size[1]):
                left_points.append((i, col))
            else:
                right_points.append((i, col))
    left_edge_line = EdgeLine(points=left_points)
    right_edge_line = EdgeLine(points=right_points)
    return [left_edge_line, right_edge_line]
