import numpy
import pywt

from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from scipy.ndimage import convolve

import cv2
import random

class RGBRotation:
    def __call__(self, image):
        image_array = numpy.array(image)
        r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

        # rotate the RGB channels around each pixel in a clockwise direction
        rotated_r = numpy.roll(r, shift=1, axis=0) # rotate the red channel downwards by one pixel
        rotated_g = numpy.roll(g, shift=1, axis=1) # rotate the green channel downwards by one pixel
        rotated_b = numpy.roll(b, shift=-1, axis=0) # rotate the blue channel upwards by one pixel

        # merge the rotated RGB channels back into an image
        transformed_image = numpy.dstack((rotated_r, rotated_g, rotated_b))

        return Image.fromarray(transformed_image)
    
class HSVRotation:
    def __call__(self, image):

        image = image.convert('HSV')
        image_array = numpy.array(image)

        h, s, v = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

        rotated_h = numpy.roll(h, shift=1, axis=0) 
        rotated_s = numpy.roll(s, shift=1, axis=1) 
        rotated_v = numpy.roll(v, shift=-1, axis=0)

        transformed_image = numpy.dstack((rotated_h, rotated_s, rotated_v))

        return Image.fromarray(transformed_image)

class HSVSwap:
    def __call__(self, image):

        image = image.convert('HSV')
        image_array = numpy.array(image)
        
        h, s, v = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]


        transformed_image = numpy.dstack((v, s, h))

        return Image.fromarray(transformed_image)
    
class Posterization:
    def __init__(self, levels):
        self.levels = levels

    def __call__(self, image):
        # Convert image to 'RGB' mode
        image = image.convert('RGB')
        # Apply posterization to each color channel separately
        posterized_channels = [self.posterize_channel(channel) for channel in image.split()]
        # Merge the posterized color channels back into an RGB image
        return Image.merge('RGB', posterized_channels)

    def posterize_channel(self, channel):
        # Apply posterization by quantizing each pixel
        levels = min(self.levels, 256)
        div = 256 // levels
        posterized_channel = numpy.array(channel) // div * div
        # Convert posterized channel array back to PIL image
        return Image.fromarray(posterized_channel)

class ColorWarping:
    def __init__(self, frequency, amplitude):
        self.frequency = frequency
        self.amplitude = amplitude

    def __call__(self, image):
        # Convert image to 'RGB' mode
        image = image.convert('RGB')
        # Apply color warping to each color channel separately
        warped_channels = [self.warp_channel(channel) for channel in image.split()]
        # Merge the warped color channels back into an RGB image
        return Image.merge('RGB', warped_channels)

    def warp_channel(self, channel):
        # Get the size of the image
        width, height = channel.size
        # Generate a grid of x and y coordinates
        x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))
        # Compute the amount of warping based on the x and y coordinates
        warp_x = self.amplitude * numpy.sin(2 * numpy.pi * self.frequency * y / height)
        # Apply the warp to the x coordinates
        warped_x = numpy.clip(x + warp_x, 0, width - 1).astype(int)
        # Map the original channel values to the warped x coordinates
        warped_channel = numpy.array(channel)
        warped_channel = warped_channel[y, warped_x]
        # Convert warped channel array back to PIL image
        return Image.fromarray(warped_channel)
    
class ChromaticAberration:
    def __init__(self, shift_amount):
        self.shift_amount = shift_amount

    def __call__(self, image):
        # Convert image to 'RGB' mode
        image = image.convert('RGB')
        # Split the image into color channels
        red_channel, green_channel, blue_channel = image.split()
        # Apply chromatic aberration to each color channel separately
        red_shifted = self.shift_channel(red_channel, 0, 0)
        green_shifted = self.shift_channel(green_channel, self.shift_amount, self.shift_amount)
        blue_shifted = self.shift_channel(blue_channel, -self.shift_amount, -self.shift_amount)
        # Merge the shifted color channels back into an RGB image
        return Image.merge('RGB', (red_shifted, green_shifted, blue_shifted))

    def shift_channel(self, channel, dx, dy):
        # Shift the channel by dx pixels horizontally and dy pixels vertically
        width, height = channel.size
        shifted_channel = Image.new('L', (width, height))
        shifted_channel.paste(channel, (int(dx), int(dy)))
        return shifted_channel

class ColorQuantization:
    def __init__(self, num_colors):
        self.num_colors = num_colors

    def __call__(self, image):
        # Convert image to 'RGB' mode
        image = image.convert('RGB')
        # Convert image to numpy array
        image_array = numpy.array(image)
        # Flatten the image array to make it compatible with KMeans
        flattened_image = image_array.reshape(-1, 3)
        # Fit KMeans clustering algorithm to the image data
        kmeans = KMeans(n_clusters=self.num_colors, random_state=0).fit(flattened_image)
        # Get the centroids of the clusters
        centroids = kmeans.cluster_centers_.astype(numpy.uint8)
        # Replace each pixel with the nearest centroid
        quantized_image_array = centroids[kmeans.labels_]
        # Reshape the quantized image array to its original shape
        quantized_image_array = quantized_image_array.reshape(image_array.shape)
        # Convert quantized image array back to PIL image
        quantized_image = Image.fromarray(quantized_image_array)
        return quantized_image
    
class WaveletTransform:
    def __init__(self, wavelet='haar', level=1):
        self.wavelet = wavelet
        self.level = level

    def __call__(self, image):
        # Convert image to numpy array
        image_array = numpy.array(image)

        # Apply 2D wavelet transform
        coeffs = pywt.dwt2(image_array, self.wavelet)

        # Modify coefficients (e.g., shift, scale, thresholding) to augment data
        # For simplicity, let's just shift the approximation coefficients (LL subband) by one pixel
        coeffs_LL = numpy.roll(coeffs[0], shift=10, axis=0)

        # Reconstruct the augmented image from modified coefficients
        augmented_image_array = pywt.idwt2((coeffs_LL, coeffs[1]), self.wavelet)

        # Clip values to ensure they are within the valid range for image pixels
        augmented_image_array = numpy.clip(augmented_image_array, 0, 255)

        # Convert back to PIL Image
        augmented_image = Image.fromarray(augmented_image_array.astype(numpy.uint8))

        return augmented_image
