class PanoImage:
    def __init__(self, y_img, x_velo_img, y_velo_img, z_velo_img):
        self.y_img = y_img
        self.x_velo_img = x_velo_img
        self.y_velo_img = y_velo_img
        self.z_velo_img = z_velo_img

    def get_channel(self, channel_id):
        return self.y_img[channel_id]