import alog
import numpy as np
from skimage.transform import resize

ASCII_CHARS = ['@', ' ', ',', ';', '#', '!', '0', '*', '1', '2', '3']


class AsciiImage(object):
    def __init__(self, img, new_width=28):
        if img.shape[0] != new_width:
            img = resize(img, (new_width, new_width * 4),
                         anti_aliasing=False, preserve_range=True)
        self.result = self.convert_image_to_ascii(img)

    def __repr__(self):
        return self.result

    def convert_image_to_ascii(self, image, range_width=1):
        image_combined_channels = [
           [np.sum(pixel_value)/3 for pixel_value in row]
            for row in image.tolist()
        ]

        pixels_in_image = np.rint(image_combined_channels).astype(int)

        unique_pixels = np.unique(pixels_in_image)

        pixels_to_chars = [
           ''.join([ASCII_CHARS[np.where(unique_pixels==pixel_value)[0][0]] for
                    pixel_value in row])
            for row in pixels_in_image.tolist()
        ]

        return '\n'.join(pixels_to_chars)
