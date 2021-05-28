class PanoImage:
    def __init__(self, y_img, x_img, z_img):
        self.y_img = y_img
        self.x_img = x_img
        self.z_img = z_img

    def get_channel(self, channel_id):
        return self.y_img[channel_id]