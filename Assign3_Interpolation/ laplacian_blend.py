import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to create Gaussian pyramid
def gaussian_pyramid(image, levels):
    g_pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)  # Downsample the image using Gaussian blur
        g_pyramid.append(image)
    return g_pyramid

# Function to create Laplacian pyramid
def laplacian_pyramid(g_pyramid):
    l_pyramid = []
    for i in range(len(g_pyramid) - 1):
        # Upsample the next Gaussian level to the current level size
        upsampled = cv2.pyrUp(g_pyramid[i + 1])
        # !!!!!!!!!! Resize to match current level !!!!!!!!!!!!
        upsampled = cv2.resize(upsampled, (g_pyramid[i].shape[1], g_pyramid[i].shape[0]))
        # Subtract the upsampled image from the current Gaussian level to get the Laplacian 
        laplacian = cv2.subtract(g_pyramid[i], upsampled)
        l_pyramid.append(laplacian)

    # Add the last level of the Gaussian pyramid as the last Laplacian level (no next level to subtract)
    l_pyramid.append(g_pyramid[-1])
    return l_pyramid

# Function to reconstruct image from Laplacian pyramid
def reconstruct_laplacian_pyramid(l_pyramid):
    image = l_pyramid[-1]
    for i in range(len(l_pyramid) - 2, -1, -1):
        image = cv2.pyrUp(image)  # Upsample the image
        # !!!!!!! Resize !!!!!!!!!!
        image = cv2.resize(image, (l_pyramid[i].shape[1], l_pyramid[i].shape[0]))
        image = cv2.add(image, l_pyramid[i])  # Add the Laplacian to reconstruct
    return image

# Function to blend two images using Laplacian pyramids
def laplacian_blending(image1, image2, mask, levels=6):
    # Generate Gaussian pyramids
    g_pyramid1 = gaussian_pyramid(image1, levels)
    g_pyramid2 = gaussian_pyramid(image2, levels)
    g_pyramid_mask = gaussian_pyramid(mask, levels)
    
    # Generate Laplacian pyramids
    l_pyramid1 = laplacian_pyramid(g_pyramid1)
    l_pyramid2 = laplacian_pyramid(g_pyramid2)
    
    # Blend the Laplacian pyramids
    l_pyramid_blend = []
    for i in range(levels):
        h, w = l_pyramid1[i].shape[:2]
        m = cv2.resize(g_pyramid_mask[i], (w, h))
        l1 = cv2.resize(l_pyramid1[i], (w, h))
        l2 = cv2.resize(l_pyramid2[i], (w, h))
        blended = cv2.multiply(l1, m) + cv2.multiply(l2, 1.0 - m)
        l_pyramid_blend.append(blended)

    # Reconstruct the blended image from the Laplacian pyramid
    blended_image = reconstruct_laplacian_pyramid(l_pyramid_blend)
    return np.clip(blended_image, 0, 1)

# Load the input images (ensure they are the same size)
image1 = cv2.imread('greenblue.jpg')  # Replace with your first image
image2 = cv2.imread('image.jpg')  # Replace with your second image
mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your mask image (binary mask)

if image1 is None or image2 is None or mask is None:
    raise FileNotFoundError("Error: Unable to load image. Check the file path.")

# Convert images to RGB (matplotlib uses RGB, OpenCV uses BGR)
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

# Resize mask to match image size
height, width = image1.shape[:2]
mask_rgb = cv2.resize(mask_rgb, (width, height))

# Perform Laplacian pyramid blending
blended_image = laplacian_blending(image1_rgb, image2_rgb, mask_rgb)

# Display the result (show all images created and reconstructed)

fig, axes = plt.subplots(1, 4, figsize=(18, 6))

axes[0].imshow(image1_rgb)
axes[0].set_title('Image 1')
axes[0].axis('off')

axes[1].imshow(image2_rgb)
axes[1].set_title('Image 2')
axes[1].axis('off')

axes[2].imshow(mask_rgb)
axes[2].set_title('Mask Image')
axes[2].axis('off')

axes[3].imshow(blended_image)
axes[3].set_title('Blend Image')
axes[3].axis('off')

g_pyr = gaussian_pyramid(image1_rgb, levels=6)

fig2, axes2 = plt.subplots(1, 6, figsize=(20, 4))
for i, g in enumerate(g_pyr):
    axes2[i].imshow(np.clip(g, 0, 1))
    axes2[i].set_title(f'Gaussian L{i}')
    axes2[i].axis('off')

plt.suptitle("Gaussian Pyramid of Image 1", fontsize=16)

plt.tight_layout()
plt.show()
