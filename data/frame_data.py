class FrameData:
    def __init__(self, image_color, gt_image, point_cloud, calib):
        self.image_color = image_color
        self.gt_image = gt_image
        self.point_cloud = point_cloud
        self.calib = calib
