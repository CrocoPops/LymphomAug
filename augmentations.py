import numpy
import random
import pywt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
        rotated_g = numpy.roll(g, shift=1, axis=1) # rotate the green channel right by one pixel
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


class HSVSwap:

    def __init__(self):
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]

        image = image.convert('HSV')
        image_array = numpy.array(image)

        h, s, v = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

        # Swap the hue and saturation channels randomly
        if numpy.random.rand() < 0.20:
            h, s, v = v, h, s
        elif numpy.random.rand() < 0.20:
            h, s, v = s, v, h
        elif numpy.random.rand() < 0.20:
            h, s, v = s, h, v
        elif numpy.random.rand() < 0.20:
            h, s, v = h, v, s
        else:
            h, s, v = v, s, h

        transformed_image = numpy.dstack((h, s, v))

        return Image.fromarray(transformed_image)

class SaltAndPepper:

    def __init__(self, prob=0.05):
        self.salt_prob = prob
        self.pepper_prob = prob
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]

        image_array = numpy.array(image)
        noisy_image = self.add_salt_and_pepper_noise(image_array, self.salt_prob, self.pepper_prob)

        return Image.fromarray(noisy_image)

    def add_salt_and_pepper_noise(self, image, salt_prob, pepper_prob):
        noisy = image.copy()
        num_salt = numpy.ceil(salt_prob * image.size)
        num_pepper = numpy.ceil(pepper_prob * image.size)

        # Add salt noise
        coords = [numpy.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255

        # Add pepper noise
        coords = [numpy.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0

        return noisy
    
class ShuffleSquares:

    def __init__(self, square_size=10):
        self.square_size = square_size
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]
        image_array = numpy.array(image)
        shuffled_image = self.shuffle_squares(image_array, self.square_size)

        return Image.fromarray(shuffled_image)

    def shuffle_squares(self, image, square_size):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the number of fully covered squares along each dimension
        num_squares_y = height // square_size
        num_squares_x = width // square_size

        # Create a list of coordinates for the fully covered squares
        square_coords = [(i * square_size, j * square_size) for i in range(num_squares_y) for j in range(num_squares_x)]

        # Create an array to store the shuffled image
        shuffled_image = numpy.zeros_like(image)

        # Create a copy of the coordinates and shuffle them
        shuffled_coords = square_coords.copy()
        numpy.random.shuffle(shuffled_coords)

        # Place the shuffled squares into the new image
        for orig_coord, shuf_coord in zip(square_coords, shuffled_coords):
            orig_y, orig_x = orig_coord
            shuf_y, shuf_x = shuf_coord
            shuffled_image[shuf_y:shuf_y+square_size, shuf_x:shuf_x+square_size] = image[orig_y:orig_y+square_size, orig_x:orig_x+square_size]

        # Copy the remaining parts (edges) that were not covered by full squares
        if height % square_size != 0:
            shuffled_image[num_squares_y * square_size:, :] = image[num_squares_y * square_size:, :]
        if width % square_size != 0:
            shuffled_image[:, num_squares_x * square_size:] = image[:, num_squares_x * square_size:]
        if height % square_size != 0 and width % square_size != 0:
            shuffled_image[num_squares_y * square_size:, num_squares_x * square_size:] = image[num_squares_y * square_size:, num_squares_x * square_size:]

        return shuffled_image
    
class RandomGeometricTransform:

    def __init__(self):
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]
        image_array = numpy.array(image)
        transformed_image = self.apply_random_transformations(image_array)

        return Image.fromarray(transformed_image)

    def apply_random_transformations(self, image):
        # Apply random rotation
        num_rotation = [0, 1, 2, 3] 
        rotation_angle = numpy.random.choice(num_rotation)
        image = numpy.rot90(image, k=rotation_angle)

        # Apply random flip
        flip_type = numpy.random.choice(['none', 'horizontal', 'vertical', 'both'])
        if flip_type == 'horizontal':
            image = numpy.fliplr(image)
        elif flip_type == 'vertical':
            image = numpy.flipud(image)
        elif flip_type == 'both':
            image = numpy.fliplr(numpy.flipud(image))

        return image
    
class Rotation:

    def __init__(self):
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]
        image_array = numpy.array(image)
        transformed_image = self.apply_random_transformations(image_array)

        return Image.fromarray(transformed_image)

    def apply_random_transformations(self, image):
         # Apply random rotation
        num_rotation = [0, 1, 2, 3] 
        rotation_angle = numpy.random.choice(num_rotation)
        image = numpy.rot90(image, k=rotation_angle)

        return image
    

class Flip:
    
    def __init__(self):
        self.args_images = 1
    
    def __call__(self, imgs):
        image = imgs[0]
        image_array = numpy.array(image)
        transformed_image = self.apply_random_transformations(image_array)

        return Image.fromarray(transformed_image)
    
    def apply_random_transformations(self, image):

        # Apply random flip
        flip_type = numpy.random.choice(['horizontal', 'vertical', 'both'])
        if flip_type == 'horizontal':
            image = numpy.fliplr(image)
        elif flip_type == 'vertical':
            image = numpy.flipud(image)
        elif flip_type == 'both':
            image = numpy.fliplr(numpy.flipud(image))
        
        return image
    
class GridColored:

    def __init__(self):
        self.num_grid_lines = 10
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]
        image_array = numpy.array(image)
        transformed_image = self.apply_grid_coloring(image_array)

        return Image.fromarray(transformed_image)

    def apply_grid_coloring(self, image):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Apply the augmentation
        for i in range(0, self.num_grid_lines):
            # Randomize the column to draw the line
            column = numpy.random.randint(0, width)
            for j in range(0, height):
                image[j, column] = self.get_random_color()
                
            # Randomize the row to draw the line 
            row = numpy.random.randint(0, height)
            for j in range(0, width):
                image[row, j] = self.get_random_color()
            
        return image

    def get_random_color(self):
        return numpy.random.randint(0, 256, 3, dtype=numpy.uint8)
    
class RandomBrightness:
    def __init__(self):
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_brightness(image)

        return Image.fromarray(transformed_image)
    
    def apply_brightness(self, image):
        datagen = ImageDataGenerator(brightness_range=[0.2, 1.9], fill_mode='nearest') # range of brightness values
        image_array = numpy.array(image)
        image_array = image_array.reshape((1,) + image_array.shape) # reshape of the image to have batch size of 1
        for batch in datagen.flow(image_array, batch_size=1):
            transformed_image = batch[0].astype('uint8')  # convert the pixel values back to integers
            break  # we only need one image
        return transformed_image
    
class RandomShifts:
    def __init__(self):
        self.args_images = 1

    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_shifts(image)

        return Image.fromarray(transformed_image)
    
    def apply_shifts(self, image):
        datagen = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3) # shift content by a maximum of 45% of the image size
        image_array = numpy.array(image)
        image_array = image_array.reshape((1,) + image_array.shape) # reshape of the image to have batch size of 1
        for batch in datagen.flow(image_array, batch_size=1):
            transformed_image = batch[0].astype('uint8')
            break
        return transformed_image
    
# class ComboGeometricBrightness:
#     def __init__(self):
#         self.args_images = 1

#     def __call__(self, imgs):
#         image = imgs[0]
#         transformed_image = self.apply_combo(image)
#         return Image.fromarray(transformed_image)
    
#     def apply_combo(self, image):
#         # Apply RandomRotation and RandomBrightness
#         image_array = numpy.array(image)
#         transformed_image = Image.fromarray(RandomGeometricTransform().apply_random_transformations(image_array))
#         transformed_image = RandomBrightness().apply_brightness(transformed_image)
#         return transformed_image

# class ComboBrightnessRandomShifts:
#     def __init__(self):
#         self.args_images = 1

#     def __call__(self, imgs):
#         image = imgs[0]
#         transformed_image = self.apply_combo(image)
#         return Image.fromarray(transformed_image)
    
#     def apply_combo(self, image):
#         # Apply RandomBrightness and RandomShifts
#         image_array = numpy.array(image)
#         transformed_image = Image.fromarray(RandomBrightness().apply_brightness(image_array))
#         transformed_image = RandomShifts().apply_shifts(transformed_image)
#         return transformed_image
    
# class ComboGeometricRGBRotation:
#     def __init__(self):
#         self.args_images = 1
#     def __call__(self, imgs):
#         image = imgs[0]
#         transformed_image = self.apply_combo(image)
#         return transformed_image
#     def apply_combo(self, image):
#         # Apply RandomRotation and RGBRotation
#         image_array = numpy.array(image)
#         transformed_image = Image.fromarray(RandomGeometricTransform().apply_random_transformations(image_array))
#         transformed_image = RGBRotation().__call__([transformed_image])
#         return transformed_image

# class ComboGeometricHSVRotation:
#     def __init__(self):
#         self.args_images = 1
#     def __call__(self, imgs):
#         image = imgs[0]
#         transformed_image = self.apply_combo(image)
#         return transformed_image
#     def apply_combo(self, image):
#         # Apply RandomRotation and HSVRotation
#         image_array = numpy.array(image)
#         transformed_image = Image.fromarray(RandomGeometricTransform().apply_random_transformations(image_array))
#         transformed_image = HSVRotation().__call__([transformed_image])
#         return transformed_image
    
# class ComboGeometricShift:
#     def __init__(self):
#         self.args_images = 1
#     def __call__(self, imgs):
#         image = imgs[0]
#         transformed_image = self.apply_combo(image)
#         return Image.fromarray(transformed_image)
#     def apply_combo(self, image):
#         # Apply RandomShifts and RandomGeometricTransform
#         image_array = numpy.array(image)
#         transformed_image = Image.fromarray(RandomShifts().apply_shifts(image_array))
#         transformed_image = RandomGeometricTransform().apply_random_transformations(transformed_image)
#         return transformed_image

# class ComboHSVShift:
#     def __init__(self):
#         self.args_images = 1
        
#     def __call__(self, imgs):
#         image = imgs[0]
#         transformed_image = self.apply_combo(image)
#         return transformed_image
    
#     def apply_combo(self, image):
#         # Apply RandomShifts and HSVRotation
#         # image_array = numpy.array(image)
#         transformed_image = Image.fromarray(RandomShifts().apply_shifts(image))
#         transformed_image = HSVRotation().__call__([transformed_image])
#         return transformed_image

class ComboGeometricBrightness:
    def __init__(self, probGeo = 0.5, probBright = 0.5):
        self.args_images = 1
        self.probGeo = probGeo
        self.probBright = probBright

    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_combo(image)
        return transformed_image
    
    def apply_combo(self, image):
        # Apply RandomRotation and RandomBrightness
        tranformed = False
        transformed_image = image
        while not tranformed:
            if numpy.random.rand() < self.probGeo:
                tranformed = True
                image_array = numpy.array(image)
                transformed_image = Image.fromarray(RandomGeometricTransform().apply_random_transformations(image_array))
            if numpy.random.rand() < self.probBright:
                tranformed = True
                transformed_image = Image.fromarray(RandomBrightness().apply_brightness(transformed_image))
        
        return transformed_image
    
class ComboHSVShift:
    def __init__(self, probShift = 0.5, probHSV = 0.5):
        self.args_images = 1
        self.probShift = probShift
        self.probHSV = probHSV
    
    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_combo(image)
        return transformed_image
    
    def apply_combo(self, image):
        tranformed = False
        transformed_image = image
        while not tranformed:
            if numpy.random.rand() < self.probShift:
                tranformed = True
                image_array = numpy.array(image)
                transformed_image = Image.fromarray(RandomShifts().apply_shifts(image_array))
            if numpy.random.rand() < self.probHSV:
                tranformed = True
                transformed_image = HSVRotation().__call__([transformed_image])
        return transformed_image

class ComboGeometricHSVRotation:
    def __init__(self, probRot = 0.33, probFlip = 0.33, probHSV = 0.33):
        self.args_images = 1
        self.probRot = probRot
        self.probFlip = probFlip
        self.probHSV = probHSV
        
    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_combo(image)

        return transformed_image

    def apply_combo(self, image):
        transformed = False
        transformed_image = image
        while not transformed:
            if numpy.random.rand() < self.probRot:
                transformed = True
                image_array = numpy.array(image)
                transformed_image = Image.fromarray(Rotation().apply_random_transformations(image_array))

            if numpy.random.rand() < self.probFlip:
                transformed = True
                image_array = numpy.array(transformed_image)
                transformed_image = Image.fromarray(Flip().apply_random_transformations(image_array))

            if numpy.random.rand() < self.probHSV:
                transformed = True
                transformed_image = HSVRotation().__call__([transformed_image])

        return transformed_image

class ComboGeometricShift:
    def __init__(self, probShift = 0.3, probRot = 0.33, probFlip = 0.33):
        self.args_images = 1
        self.probShift = probShift
        self.probRot = probRot
        self.probFlip = probFlip
        
    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_combo(image)

        return transformed_image

    def apply_combo(self, image):
        transformed_image = numpy.array(image)
        trasformed = False
        while not trasformed:
            if numpy.random.rand() < self.probShift:
                trasformed = True
                transformed_image = Image.fromarray(RandomShifts().apply_shifts(transformed_image))

            if numpy.random.rand() < self.probRot:
                trasformed = True
                transformed_image = Image.fromarray(Rotation().apply_random_transformations(transformed_image))

            if numpy.random.rand() < self.probFlip:
                trasformed = True
                transformed_image = Image.fromarray(Flip().apply_random_transformations(transformed_image))

        return transformed_image

class ComboBrightnessRandomShifts:
    def __init__(self, probBright = 0.5, probShift = 0.5):
        self.args_images = 1
        self.probBright = probBright
        self.probShift = probShift
        
    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_combo(image)

        return transformed_image

    def apply_combo(self, image):
        transformed_image = numpy.array(image)
        trasformed = False
        while not trasformed:
            if numpy.random.rand() < self.probBright:
                trasformed = True
                transformed_image = Image.fromarray(RandomBrightness().apply_brightness(transformed_image))

            if numpy.random.rand() < self.probShift:
                trasformed = True
                transformed_image = Image.fromarray(RandomShifts().apply_shifts(transformed_image))

        return transformed_image

class ComboGeometricRGBRotation:
    def __init__(self, probRot = 0.5, probGeo = 0.5):
        self.args_images = 1
        self.probRot = probRot
        self.probGeo = probGeo
        
    def __call__(self, imgs):
        image = imgs[0]
        transformed_image = self.apply_combo(image)

        return transformed_image

    def apply_combo(self, image):
        transformed = False
        transformed_image = image
        while not transformed:
            if numpy.random.rand() < self.probRot:
                transformed = True
                transformed_image = RGBRotation().__call__([transformed_image])

            if numpy.random.rand() < self.probGeo:
                transformed = True
                image_array = numpy.array(transformed_image)
                transformed_image = Image.fromarray(RandomGeometricTransform().apply_random_transformations(image_array))

        return transformed_image


# class ComboGeometricHSVRotation:
#     def __init__(self, probRot = 0.33, probFlip = 0.33, probHSV = 0.33):
#         self.args_images = 1
#         self.probRot = probRot
#         self.probFlip = probFlip
#         self.probHSV = probHSV
        
#     def __call__(self, imgs):
#         image = imgs[0]
#         image_array = numpy.array(image)
#         transformed_image = self.apply_combo(image_array)

#         return transformed_image

#     def apply_combo(self, image):
#         # Apply random rotation
#         if numpy.random.rand() < self.probRot:
#             num_rotation = [0, 1, 2, 3]  
#             rotation_angle = numpy.random.choice(num_rotation)
#             image = numpy.rot90(image, k=rotation_angle)

#         if numpy.random.rand() < self.probFlip:
#             flip_type = numpy.random.choice(['horizontal', 'vertical', 'both'])
#             if flip_type == 'horizontal':
#                 image = numpy.fliplr(image)
#             elif flip_type == 'vertical':
#                 image = numpy.flipud(image)
#             elif flip_type == 'both':
#                 image = numpy.fliplr(numpy.flipud(image))

#         if numpy.random.rand() < self.probHSV:
#             image = Image.fromarray(image).convert('HSV')
#             image_array = numpy.array(image)

#             h, s, v = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

#             rotated_h = numpy.roll(h, shift=1, axis=0) 
#             rotated_s = numpy.roll(s, shift=1, axis=1) 
#             rotated_v = numpy.roll(v, shift=-1, axis=0)

#             image = numpy.dstack((rotated_h, rotated_s, rotated_v))

#         return Image.fromarray(image)


