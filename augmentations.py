import numpy
import random
import pywt
import os
from PIL import Image

class NoneAug:
    def __init__(self):
        self.args_images = 1
    
    def __call__(self, imgs):
        return imgs[0]

class DWTAverageFusion:
    def __init__(self):
        self.args_images = 2
        self.wavelet = 'haar'

    def __call__(self, imgs):
        
        img1 = imgs[0]
        img2 = imgs[1]

        result = self.fuse_images(img1, img2)

        return result
    
    def fuse_images(self, img1, img2):

        images = [img1, img2]

        # Convert the images to numpy arrays
        images_array = [numpy.array(img) for img in images]

        # Separate the RGB channels of the images
        # [[R,G,B], [R,G,B], [R,G,B], [R,G,B]]
        channels = [[img_arr[:,:,i] for i in range(3)] for img_arr in images_array]

        # Apply the wavelet transform to each color channel
        fused_channels = []
        for i in range(3):
            # [RC1, RC2, RC3, RC4]
            coeffs = [pywt.dwt2(c[i], self.wavelet) for c in channels]
            
            cA = cV = cH = cD = 0
            for coeff in coeffs:
                cA += coeff[0] 
                cH += coeff[1][0]
                cV += coeff[1][1] 
                cD += coeff[1][2]

            cA = cA / self.args_images
            cH = cH / self.args_images
            cV = cV / self.args_images
            cD = cD / self.args_images
            

            fused_coeffs = (cA, (cH, cV, cD))
                
            # Reconstruct the fused channel
            fused_channel_array = pywt.idwt2(fused_coeffs, self.wavelet)
            
            # Clip values to ensure they are within 0-255 range
            fused_channel_array = (fused_channel_array * 255 / numpy.max(fused_channel_array)).astype('uint8')
            
            # Append the fused channel
            fused_channels.append(fused_channel_array)

        # Combine the fused color channels into an RGB image
        fused_image_array = numpy.stack(fused_channels, axis=-1)

        # Convert array back to uint8 and create PIL image
        return Image.fromarray(numpy.uint8(fused_image_array))
    
class DWTRandomFusion:
    def __init__(self):
        self.args_images = 2
        self.wavelet = 'db1'

    def __call__(self, imgs):
        
        img1 = imgs[0]
        img2 = imgs[1]

        return self.fuse_images(img1, img2)
    
    def fuse_images(self, img1, img2):

        images = [img1, img2]

        # Convert the images to numpy arrays
        images_array = [numpy.array(img1) for img in images]

        # Separate the RGB channels of the images
        # [[R,G,B], [R,G,B]]
        channels = [[img_arr[:,:,i] for i in range(3)] for img_arr in images_array]

        # Apply the wavelet transform to each color channel
        fused_channels = []
        for i in range(3):
            # [RC1, RC2]
            coeffs = [pywt.dwt2(c[i], self.wavelet) for c in channels]

            cA = coeffs[0][0] if random.random() <= 0.5 else coeffs[1][0]
            cH = coeffs[0][1][0] if random.random() <= 0.5 else coeffs[1][1][0]
            cV = coeffs[0][1][1] if random.random() <= 0.5 else coeffs[1][1][1]
            cD = coeffs[0][1][2] if random.random() <= 0.5 else coeffs[1][1][2]

            fused_coeffs = (cA, (cH, cV, cD))
                
            # Reconstruct the fused channel
            fused_channel_array = pywt.idwt2(fused_coeffs, self.wavelet)
            
            # Clip values to ensure they are within 0-255 range
            fused_channel_array = (fused_channel_array * 255 / numpy.max(fused_channel_array)).astype('uint8')
            
            # Append the fused channel
            fused_channels.append(fused_channel_array)

        # Combine the fused color channels into an RGB image
        fused_image_array = numpy.stack(fused_channels, axis=-1)

        # Convert array back to uint8 and create PIL image
        return Image.fromarray(numpy.uint8(fused_image_array))
    
class DWTMaxFusion:
    def __init__(self):
        self.args_images = 2
        self.wavelet = 'db1'

    def __call__(self, imgs):
        
        img1 = imgs[0]
        img2 = imgs[1]

        return self.fuse_images(img1, img2)
    
    def fuse_images(self, img1, img2):

        images = [img1, img2]

        # Convert the images to numpy arrays
        images_array = [numpy.array(img1) for img in images]

        # Separate the RGB channels of the images
        # [[R,G,B], [R,G,B]]
        channels = [[img_arr[:,:,i] for i in range(3)] for img_arr in images_array]

        # Apply the wavelet transform to each color channel
        fused_channels = []
        for i in range(3):
            # [RC1, RC2]
            coeffs = [pywt.dwt2(c[i], self.wavelet) for c in channels]

            cA = numpy.maximum(coeffs[0][0], coeffs[1][0])
            cH = numpy.maximum(coeffs[0][1][0], coeffs[1][1][0])
            cV = numpy.maximum(coeffs[0][1][1], coeffs[1][1][1])
            cD = numpy.maximum(coeffs[0][1][2], coeffs[1][1][2])

            fused_coeffs = (cA, (cH, cV, cD))
                
            # Reconstruct the fused channel
            fused_channel_array = pywt.idwt2(fused_coeffs, self.wavelet)
            
            # Clip values to ensure they are within 0-255 range
            fused_channel_array = (fused_channel_array * 255 / numpy.max(fused_channel_array)).astype('uint8')
            
            # Append the fused channel
            fused_channels.append(fused_channel_array)

        # Combine the fused color channels into an RGB image
        fused_image_array = numpy.stack(fused_channels, axis=-1)

        # Convert array back to uint8 and create PIL image
        return Image.fromarray(numpy.uint8(fused_image_array))
    
class DWTMinFusion:
    def __init__(self):
        self.args_images = 2
        self.wavelet = 'db1'

    def __call__(self, imgs):
        
        img1 = imgs[0]
        img2 = imgs[1]

        return self.fuse_images(img1, img2)
    
    def fuse_images(self, img1, img2):

        images = [img1, img2]

        # Convert the images to numpy arrays
        images_array = [numpy.array(img1) for img in images]

        # Separate the RGB channels of the images
        # [[R,G,B], [R,G,B]]
        channels = [[img_arr[:,:,i] for i in range(3)] for img_arr in images_array]

        # Apply the wavelet transform to each color channel
        fused_channels = []
        for i in range(3):
            # [RC1, RC2]
            coeffs = [pywt.dwt2(c[i], self.wavelet) for c in channels]

            cA = numpy.minimum(coeffs[0][0], coeffs[1][0])
            cH = numpy.minimum(coeffs[0][1][0], coeffs[1][1][0])
            cV = numpy.minimum(coeffs[0][1][1], coeffs[1][1][1])
            cD = numpy.minimum(coeffs[0][1][2], coeffs[1][1][2])

            fused_coeffs = (cA, (cH, cV, cD))
                
            # Reconstruct the fused channel
            fused_channel_array = pywt.idwt2(fused_coeffs, self.wavelet)
            
            # Clip values to ensure they are within 0-255 range
            fused_channel_array = (fused_channel_array * 255 / numpy.max(fused_channel_array)).astype('uint8')
            
            # Append the fused channel
            fused_channels.append(fused_channel_array)

        # Combine the fused color channels into an RGB image
        fused_image_array = numpy.stack(fused_channels, axis=-1)

        # Convert array back to uint8 and create PIL image
        return Image.fromarray(numpy.uint8(fused_image_array))
    
class RGBRotation:

    def __init__(self):
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]

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

    def __init__(self):
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]

        image = image.convert('HSV')
        image_array = numpy.array(image)

        h, s, v = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

        rotated_h = numpy.roll(h, shift=1, axis=0) 
        rotated_s = numpy.roll(s, shift=1, axis=1) 
        rotated_v = numpy.roll(v, shift=-1, axis=0)

        transformed_image = numpy.dstack((rotated_h, rotated_s, rotated_v))

        return Image.fromarray(transformed_image)