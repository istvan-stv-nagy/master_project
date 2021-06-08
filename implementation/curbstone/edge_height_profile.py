from typing import List
import numpy as np
from implementation.curbstone.edge_line import EdgeLine
from implementation.datastructures.pano_image import PanoImage


class EdgeHeightProfile:
    def __init__(self, height_values, median_height, elevations):
        self.height_values = height_values
        self.median_height = median_height
        self.elevations = elevations


def compute_height_profiles(edges: List[EdgeLine], pano_image: PanoImage):
    profiles: List[EdgeHeightProfile] = []
    for edge in edges:
        edge_profile: EdgeHeightProfile = compute_height_profile(edge, pano_image)
        profiles.append(edge_profile)
    return profiles


def compute_height_profile(edge: EdgeLine, pano_image: PanoImage):
    h = pano_image.y_road_img
    kernel = np.array([-2, -1, 0, 1, 2])
    height, width = pano_image.y_road_img.shape
    height_values = []
    elevations = []
    for point in edge.points:
        r = point[0]
        c = point[1]
        x = [0, 0, h[r][c], 0, 0]
        if 0 < c < width - 1:
            x = [0, h[r][c - 1], h[r][c], h[r][c+1], 0]
        if 1 < c + 1 < width - 2:
            x = [h[r][c - 2], h[r][c-1], h[r][c], h[r][c+1], h[r][c+2]]
        height = h[r][c]
        height_values.append(height)
        elevation = np.sum(np.array(x) * kernel)
        elevations.append(elevation)
    height_values = np.array(height_values)
    median_height = np.median(height_values)
    return EdgeHeightProfile(height_values=height_values, median_height=median_height, elevations=elevations)
