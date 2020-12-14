from data.data_reader import DataReader
from visu.frame_visu import FrameVisu

if __name__ == '__main__':
    reader = DataReader()
    frame_data = reader.read_frame(60)
    visu = FrameVisu(frame_data)
    visu.show()
