"""
This steganography method is named: 'HideJPEGinJPEGLSB'

* Created by: Ruinan Ma
* Created time: 2022/09/30

Description:
    Hide one JPEG image into another JPEG image using 4-bits LSB method.
"""
from PIL import Image


class HideJPEGinJPEGLSB():
    def __init__(self):
        self.BLACK_PIXEL = (0, 0, 0)
        
    def _int_to_bin(self, rgb):
        """Convert an integer tuple to a binary (string) tuple.
        :param rgb: An integer tuple like (220, 110, 96)
        :return: A string tuple like ("00101010", "11101011", "00010110")
        """ 
        r, g, b = rgb
        return f'{r:08b}', f'{g:08b}', f'{b:08b}'
    
    def _bin_to_int(self, rgb):
        """Convert a binary (string) tuple to an integer tuple.
        :param rgb: A string tuple like ("00101010", "11101011", "00010110")
        :return: Return an int tuple like (220, 110, 96)
        """
        r, g, b = rgb
        return int(r, 2), int(g, 2), int(b, 2)
            
        
    def _merge_rgb(self, rgb1, rgb2):
        '''Merge two RGB tuples.
        :param rgb1: An integer tuple like (220, 110, 96)
        :param rgb2: An integer tuple like (240, 95, 105)
        :return: An integer tuple with the two RGB values merged.
        '''
        r1, g1, b1 = self._int_to_bin(rgb=rgb1)
        r2, g2, b2 = self._int_to_bin(rgb=rgb2)
        rgb = r1[:4] + r2[:4], g1[:4] + g2[:4], b1[:4] + b2[:4]
        return self._bin_to_int(rgb)
    
    def _unmerge_rgb(self, rgb):
        """Unmerge RGB.
        :param rgb: An integer tuple like (220, 110, 96)
        :return: An integer tuple with the two RGB values merged.
        """
        r, g, b = self._int_to_bin(rgb)
        # Extract the last 4 bits (corresponding to the hidden image)
        # Concatenate 4 zero bits because we are working with 8 bit
        new_rgb = r[4:] + '0000', g[4:] + '0000', b[4:] + '0000'
        return self._bin_to_int(new_rgb)
        
        
    def encode(self, host_img, secret_img):
        """Merge secret_img into host_img.
        :param host_img: host_img image
        :param secret_img: secret_img image
        :return: Container image.
        """
        # Check the images dimensions. host_img.size() has to be '>=' secret_img.size()
        if secret_img.size[0] > host_img.size[0] or secret_img.size[1] > host_img.size[1]:
            raise ValueError('Sorry, secret image should be smaller than host image.')
        
        # Get the pixel map of the two images
        map1 = host_img.load()
        map2 = secret_img.load()

        # Create a new image that will be outputted    
        new_image = Image.new(host_img.mode, host_img.size)
        new_map = new_image.load()

        for i in range(host_img.size[0]):
            for j in range(host_img.size[1]):
                is_valid = lambda: i < secret_img.size[0] and j < secret_img.size[1]
                rgb1 = map1[i, j]
                rgb2 = map2[i, j] if is_valid() else self.BLACK_PIXEL
                new_map[i, j] = self._merge_rgb(rgb1, rgb2)
        
        return new_image

        
    def decode(self, container):
        """Unmerge an image.
        :param container: The input container.
        :return: The unmerged/extracted secret image.
        """
        pixel_map = container.load()
        
        # Create the new image and load the pixel map
        new_image = Image.new(container.mode, container.size)
        new_map = new_image.load()

        for i in range(container.size[0]):
            for j in range(container.size[1]):
                new_map[i, j] = self._unmerge_rgb(pixel_map[i, j])
        
        return new_image


# For test:
# host = Image.open("test-pic/2.JPEG")
# secret = Image.open("test-pic/1.JPEG")
# stego = HideJPEGinJPEGLSB()
# container =  stego.encode(host_img=host, secret_img=secret)
# container.save("./container.png")
# container = Image.open("./container.png")
# revealer = stego.decode(container)
# revealer.save("./revealer.png")