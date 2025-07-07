from skimage import feature


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def get_blp_image_v1(self, image, eps=1e-7, method="default"):

        lbp_image = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method=method)

        return lbp_image

