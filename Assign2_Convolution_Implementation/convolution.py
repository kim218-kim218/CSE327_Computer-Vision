import cv2
import math
import numpy as np
from scipy.signal import convolve2d
# from google.colab.patches import cv2_imshow

# Load a grayscale image
org_image = cv2.imread('image.jpg')
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# print(image)

# Ensure the image is loaded correctly
if image is None:
    raise FileNotFoundError("Error: Unable to load image. Check the file path.")

# Apply Gaussian Blur to smooth the image (kernel size = 5x5, sigmaX = 1)
# ------------- For test --------------------
# gaussian_blurred_CV = cv2.GaussianBlur(image, (5, 5), 1)
# print(gaussian_blurred_CV)
# -------------------------------------------

def GaussianBlur_NahyunKim(image, kernel_size, sigma):
    # 1. create gaussian kernel
    k = kernel_size[0] // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    gaussian_kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= np.sum(gaussian_kernel)

    # 2. image pedding with 'reflect' (in cv, they use 'reflect' for pedding img)
    padded_image = np.pad(image, ((k, k), (k, k)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)

    # 3. matrix calculation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size[0], j:j+kernel_size[1]]
            output[i, j] = int(np.sum(region * gaussian_kernel))

    return cv2.convertScaleAbs(output)

gaussian_blurred = GaussianBlur_NahyunKim(image, (5, 5), 1)

# Define a kernel (e.g., edge detection filter)
kernel = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

# Apply cross-correlation (OpenCV's filter2D)
# ------------- For test --------------------
# cross_correlation_result_CV = cv2.filter2D(image, -1, kernel)
# cc_filtered_image_uint8 = cv2.convertScaleAbs(cross_correlation_result_CV)
# print(cc_filtered_image_uint8)
# -------------------------------------------

def filter2D_NahyunKim(image, kernel):
    kernel_size = kernel.shape[0]
    k = kernel_size // 2

    padded_image = np.pad(image, ((k, k), (k, k)), mode='reflect')
    # print(padded_image)

    output = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = int(np.sum(region * kernel))
    
    return cv2.convertScaleAbs(output)

cross_correlation_result = filter2D_NahyunKim(image, kernel)

# Apply convolution (SciPy's convolve2d)
# ------------- For test --------------------
# convolution_result_CV = convolve2d(image, np.flip(kernel), mode='same')
# conv_filtered_image_uint8 = cv2.convertScaleAbs(convolution_result_CV)
# print(conv_filtered_image_uint8)
# -------------------------------------------

def convolve2d_NahyunKim(image, kernel,mode='same'):
  return filter2D_NahyunKim(image, kernel)

convolution_result = convolve2d_NahyunKim(image, np.flip(kernel),mode='same')


# Display results using cv2_imshow
cv2.imshow('Original Image', org_image)
cv2.imshow('Grayscale Image', image)
cv2.imshow('Gaussian Blurred Image', gaussian_blurred)
cv2.imshow('Cross-Correlation', cross_correlation_result)
cv2.imshow('Convolution', convolution_result)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
