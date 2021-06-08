from typing import List

import numpy as np

from implementation.curbstone.edge_extraction import extract_road_edges
from implementation.curbstone.edge_height_profile import compute_height_profiles, EdgeHeightProfile
from implementation.curbstone.edge_line import EdgeLine
from implementation.datastructures.pano_image import PanoImage


class RoadEdgeClassifier:
    def __init__(self):
        pass

    def run(self, prediction: np.ndarray, pano_image: PanoImage):
        edges: List[EdgeLine] = extract_road_edges(prediction)
        profiles: List[EdgeHeightProfile] = compute_height_profiles(edges, pano_image)
        return edges, profiles

