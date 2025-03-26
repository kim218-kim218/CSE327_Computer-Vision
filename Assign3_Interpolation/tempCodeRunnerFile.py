g_pyr = gaussian_pyramid(image1_rgb, levels=6)

# fig2, axes2 = plt.subplots(1, 6, figsize=(20, 4))
# for i, g in enumerate(g_pyr):
#     axes2[i].imshow(np.clip(g, 0, 1))
#     axes2[i].set_title(f'Gaussian L{i}')
#     axes2[i].axis('off')

# plt.suptitle("Gaussian Pyramid of Image 1", fontsize=16)
