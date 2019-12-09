import alog
import numpy as np
from skimage.transform import resize

ASCII_CHARS = ['@', ' ']


class AsciiImage(object):
    def __init__(self, img, new_width=28):
        if img.shape[0] != new_width:
            img = resize(img, (new_width, new_width * 4), anti_aliasing=False)

        self.result = self.convert_image_to_ascii(img)

    def __repr__(self):
        return self.result

    def convert_image_to_ascii(self, image, range_width=1):
        image_combined_channels = [
           [sum(pixel_value[:3])/3 for pixel_value in row]
            for row in image.tolist()
        ]

        product_ascii_chars = image_combined_channels * (len(ASCII_CHARS) - 1)

        pixels_in_image = np.floor(product_ascii_chars).astype(int)

        pixels_to_chars = [
           ''.join([ASCII_CHARS[pixel_value] for pixel_value in row])
            for row in pixels_in_image.tolist()
        ]

        return '\n'.join(pixels_to_chars)
