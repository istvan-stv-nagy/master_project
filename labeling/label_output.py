class LabelOutput:
    def __init__(self, pano_image, label_image):
        self.pano_image = pano_image
        self.label_image = label_image

    def data(self):
        return self.pano_image

    def label(self):
        return self.label_image

