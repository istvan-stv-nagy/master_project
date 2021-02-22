class PanoImage:
    def __init__(self, img):
        self.img = img

    def get_channel(self, channel_id):
        return self.img[channel_id]