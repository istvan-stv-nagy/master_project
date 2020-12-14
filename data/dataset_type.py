from enum import Enum


class DatasetType(Enum):
    UNMARKED = 1


IMAGE_PREFIX = {DatasetType.UNMARKED: 'um_'}

GT_PREFIX = {DatasetType.UNMARKED: 'um_lane_'}
