from enum import Enum


class RoadType(Enum):
    UNMARKED = 1


IMAGE_PREFIX = {RoadType.UNMARKED: 'um_'}

GT_PREFIX = {RoadType.UNMARKED: 'um_lane_'}
