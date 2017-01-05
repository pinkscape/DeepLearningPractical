import tensorflow as ts

# Let's start from some basics
# In this section, we will read in an image, and detect its edge map using the built-in sobel edge detection from skimage library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.filter as filters
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np

# Read and display an Image
img = mpimg.imread('data/test.jpg')

# Let's detect edge with skimage's sobel filter
img_gray = rgb2gray(img)
edges = filters.sobel(img_gray)

# Now let's display the color image and the edge map
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(img)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

ax2.axis('off')
ax2.imshow(edges, cmap="gray")
ax2.set_title('Sobel edge')
ax1.set_adjustable('box-forced')

plt.show()
"""

################################################################################
# TODO: Implement your own sobel edge detector                                 #
################################################################################
def my_sobel(img):
    # Input:
    # img: an grayscale image
    # Output:
    # edge: a map for the magnitude of gradient
    # Hint:
    # 1) The formula for sobel filter can be found here:https://en.wikipedia.org/wiki/Sobel_operator
    # 2) You can use scipy.ndimage.convolve for 2D convolution

    # Step 1:
    # Design two filters sobel_kernel_h and sobel_kernel_v.
    # They will be used to computer image gradient in horizontal and vertical directions
    sobel_kernel_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_kernel_v = np.transpose(sobel_kernel_h)

    # Step 2:
    # convolve img with these two filters
    conv_x = convolve(img,sobel_kernel_h)
    conv_y = convolve(img,sobel_kernel_v)

    # Step 3:
    # Compute edge: the map for the magnitude of gradient
    edge = np.sqrt(conv_x**2 + conv_y**2)

    return edge


################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

edge = my_sobel(img_gray)

# display
plt.imshow(edge, cmap="gray");
plt.show()