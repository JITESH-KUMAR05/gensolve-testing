import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the uploaded image
image_path = "circle.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))  # Increase resolution
img = img.astype(np.float32) / 255.0

# Shape detection function
def detect_shape(image):
    # Convert to binary image
    _, thresh = cv2.threshold(image, 0.5, 1.0, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Assume the largest contour is the shape
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "rectangle"  # Could be square or rectangle
    elif len(approx) == 5:
        return "pentagon"
    elif len(approx) > 5:
        return "circle"
    else:
        return "unknown"

# Generate reference shape
def generate_shape_image(shape, img_size=(128, 128)):
    img = np.zeros(img_size, dtype=np.uint8)
    
    if shape == "circle":
        cv2.circle(img, (64, 64), 40, 255, -1)
    elif shape == "square":
        cv2.rectangle(img, (32, 32), (96, 96), 255, -1)
    elif shape == "rectangle":
        cv2.rectangle(img, (20, 40), (108, 88), 255, -1)
    
    img = img.astype(np.float32) / 255.0
    return img

# Detect the shape in the uploaded image
detected_shape = detect_shape(img)

# Expected shape
expected_shape = "circle"  # Change this to the shape you expect

# If the detected shape is not as expected, generate a corrected shape
if detected_shape != expected_shape:
    corrected_img = generate_shape_image(expected_shape)
    print(f"Detected shape: {detected_shape}. Correcting to {expected_shape}.")

    # Save the corrected image
    corrected_img_path = "corrected_shape.jpg"
    cv2.imwrite(corrected_img_path, (corrected_img * 255).astype(np.uint8))
    print(f"Corrected image saved at: {corrected_img_path}")
else:
    print(f"Shape detected correctly as {expected_shape}.")
    corrected_img = img

# Denoising, Edge Detection, and Sharpening

# Load the trained denoising model
model = load_model('denoising_cnn_model.h5')

# Resize the image back to (64, 64) for the model
input_img = cv2.resize(corrected_img, (64, 64))

# Expand dimensions to match model input
noisy_img_input = np.expand_dims(input_img, axis=0)
noisy_img_input = np.expand_dims(noisy_img_input, axis=-1)

# Denoise the image using the trained model
denoised_img = model.predict(noisy_img_input)

# Squeeze dimensions for visualization
denoised_img = denoised_img.squeeze()

# Step 1: Apply Gaussian Blur to smooth the image
denoised_img = cv2.GaussianBlur(denoised_img, (5, 5), 0)

# Step 2: Apply Histogram Equalization to enhance contrast
denoised_img = cv2.equalizeHist((denoised_img * 255).astype(np.uint8))
denoised_img = denoised_img.astype(np.float32) / 255.0

# Step 3: Apply Edge Detection using Canny Edge Detector with lower thresholds
edges = cv2.Canny((denoised_img * 255).astype(np.uint8), threshold1=50, threshold2=150)

# Step 4: Apply Morphological Operations to enhance edges with a smaller kernel
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=1)

# Step 5: Apply Unsharp Masking (Advanced Sharpening)
gaussian = cv2.GaussianBlur(denoised_img, (9, 9), 2.0)
unsharp_img = cv2.addWeighted(denoised_img, 2.0, gaussian, -1.0, 0)

# Step 6: Combine edges with the unsharp image to enhance edges
final_img = cv2.addWeighted(unsharp_img, 0.8, edges.astype(np.float32)/255.0, 0.5, 0)

# Visualize the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title("Uploaded Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 4, 2)
plt.title("Denoised Image")
plt.imshow(denoised_img, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Edges Detected")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Final Sharp Image")
plt.imshow(final_img, cmap='gray')

plt.show()

# Save the final image
output_image_path = "sharpened_image.png"
cv2.imwrite(output_image_path, (final_img * 255).astype(np.uint8))
print(f"Sharpened image saved at: {output_image_path}")