import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Bilinear Interpolation (Using OpenCV)
def bilinear_interpolation(image, new_width, new_height):
    height, width, channels = image.shape
    # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    x_ratio = width / new_width
    y_ratio = height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = x_ratio * j
            y = y_ratio * i

            x1 = int(np.floor(x))
            y1 = int(np.floor(y))

            # width - 1 -> to prevent going outside of the boundaries
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)

            dx = x - x1
            dy = y - y1

            for c in range(channels):
                P = dx * image[y1, x2, c]+ (1 - dx) * image[y1, x1, c]
                Q = dx * image[y2, x2, c]+(1 - dx) * image[y2, x1, c]
                resized_image[i, j, c] = np.clip(dy * Q + (1 - dy) * P, 0, 255)

    return resized_image

def cubic_a(x, a):
    abs_t = abs(x)
    if abs_t < 1:
        return 1-(a+3)*(abs_t**2) + (a+2)*(abs_t**3)
    elif 1 <= abs_t < 2:
        return a*(abs_t-1) * (abs_t-2)**2
    else:
        return 0

def cubic_interp(p, x):
    # p: list  [P0, P1, P2, P3]
    # x: float, interpolation position
    a0 = p[1]
    a1 = 0.5 * p[2] - 0.5 * p[0]
    a2 = p[0] - 2.5 * p[1] + 2.0 * p[2] - 0.5 * p[3]
    a3 = -0.5 * p[0] + 1.5 * p[1] - 1.5 * p[2] + 0.5 * p[3]
    return a0 + a1 * x + a2 * x**2 + a3 * x**3


# Bicubic Interpolation (Using OpenCV)
def bicubic_interpolation(image, new_width, new_height):
    # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    height, width, channels = image.shape
    resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    x_ratio = width / new_width
    y_ratio = height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = x_ratio * j
            y = y_ratio * i

            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            dx = x - x1
            dy = y - y1

            a = -0.5  # Catmull-Rom -> -0.5

            # More complex -> calculate one by one
            # for c in range(channels):
            #     col = []
            #     for m in range(-1, 3):
            #         row = []
            #         for n in range(-1, 3):
            #             px = np.clip(x1 + n, 0, width - 1)
            #             py = np.clip(y1 + m, 0, height - 1)
            #             row.append(image[py, px, c])
            #         col.append(cubic_interp(row, dx)) 
            #     pixel = cubic_interp(col, dy)
            #     resized_image[i, j, c] = np.clip(pixel, 0, 255)

            for c in range(channels):
                pixel_value = 0.0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        P_ix = min( x1 + n, width - 1)
                        P_jy = min( y1 + m, height - 1)
                        aij = cubic_a(m-dy, a) * cubic_a(n-dx, a)
                        # NumPy -> [행(row), 열(column)] -> x축 y축 바꿔줘야함
                        pixel_value += image[P_jy,P_ix,c] * aij
                resized_image[i, j, c] = np.clip(pixel_value, 0, 255)
    return resized_image

# Load the input image
image_path = 'image.jpg'  # Provide your image path here
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Error: Unable to load image. Check the file path.")

# Convert the image from BGR (OpenCV) to RGB (matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the original dimensions of the image
original_height, original_width, _ = image.shape

# Define new dimensions (you can adjust these based on your needs)
new_width = original_width * 2
new_height = original_height * 2

# Apply Bilinear interpolation (using OpenCV)
bilinear_img = bilinear_interpolation(image_rgb, new_width, new_height)

# Apply Bicubic interpolation (using OpenCV)
bicubic_img = bicubic_interpolation(image_rgb, new_width, new_height)

# Plot the original and resized images in a single row
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Bilinear interpolation (OpenCV)
axes[1].imshow(bilinear_img)
axes[1].set_title('Bilinear Interpolation ')
axes[1].axis('off')

# Bicubic interpolation (OpenCV)
axes[2].imshow(bicubic_img)
axes[2].set_title('Bicubic Interpolation ')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Save the results
cv2.imwrite('bilinear_interpolation.jpg', cv2.cvtColor(bilinear_img, cv2.COLOR_RGB2BGR))
cv2.imwrite('bicubic_interpolation.jpg', cv2.cvtColor(bicubic_img, cv2.COLOR_RGB2BGR))
