import os
'''This code was originally done on a simple platform called Google Colab, to make it more accessible, I transfered the following code here.'''
# List of image paths
image_paths = ['/content/UID_1_2_1_all.bmp', '/content/UID_1_10_4_all.bmp', '/content/UID_H6_3_1_hem.bmp','/content/UID_H6_10_1_hem.bmp']

# 1. Setup the subplot grid (1 row, N columns)
num_images = len(image_paths)
fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

# Handle single image case so axes is always indexable
if num_images == 1:
    axes = [axes]

for i, img_path in enumerate(image_paths):
    try:
        # Load and preprocess each image
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Generate the heatmap using the fixed deep-drill function
        heatmap = get_gradcam_heatmap(img_batch, model)

        # Visualization Overlay
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(np.uint8(255 * img_array), 0.6, heatmap_colored, 0.4, 0)

        # Display in the correct subplot
        axes[i].imshow(overlay)
        axes[i].set_title(f"Image {i+1}: {os.path.basename(img_path)}")
        axes[i].axis('off')

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

plt.tight_layout()
plt.show()
