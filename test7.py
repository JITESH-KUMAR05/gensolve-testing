import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
image_path = "circle.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(image_path)  # Load the color image for overlay
img = cv2.resize(img, (128, 128))  # Increase resolution
img_color = cv2.resize(img_color, (128, 128))

# Preprocess the image
def preprocess_image(image):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to remove noise and enhance edges
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

# Shape detection function
def detect_shape(image):
    edges = preprocess_image(image)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None
    
    # Assume the largest contour is the shape
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 3:
        return "triangle", contour
    elif len(approx) == 4:
        return "rectangle", contour  # Could be square or rectangle
    elif len(approx) == 5:
        return "pentagon", contour
    elif len(approx) > 5:
        return "circle", contour
    else:
        return "unknown", contour

# Generate reference shape
def generate_shape_image(shape, img_size=(128, 128)):
    img = np.zeros(img_size, dtype=np.uint8)
    
    if shape == "circle":
        cv2.circle(img, (img_size[0]//2, img_size[1]//2), min(img_size)//3, 255, -1)
    elif shape == "square":
        side = int(min(img_size) // 1.5)
        start = int((img_size[0] - side) // 2)
        end = int(start + side)
        cv2.rectangle(img, (start, start), (end, end), 255, -1)
    elif shape == "rectangle":
        width = int(img_size[0] // 1.5)
        height = int(img_size[1] // 2)
        start_x = int((img_size[0] - width) // 2)
        start_y = int((img_size[1] - height) // 2)
        cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), 255, -1)
    
    img = img.astype(np.float32) / 255.0
    return img

# Detect the shape in the uploaded image
detected_shape, contour = detect_shape(img)

# If a shape is detected, generate a corrected shape
if detected_shape:
    corrected_img = generate_shape_image(detected_shape)
    print(f"Detected shape: {detected_shape}. Correcting to {detected_shape}.")

    # Overlay the detected shape outline on the original image
    overlay_img = img_color.copy()
    cv2.drawContours(overlay_img, [contour], -1, (0, 255, 0), 2)  # Green outline

    # Save the corrected image
    corrected_img_path = "corrected_shape.jpg"
    cv2.imwrite(corrected_img_path, overlay_img)
    print(f"Corrected image saved at: {corrected_img_path}")
else:
    print("No shape detected.")
    overlay_img = img_color

# Visualize the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Uploaded Image")
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Corrected Shape Image")
plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))

plt.show()

# Save the final image
output_image_path = "corrected_shape.png"
cv2.imwrite(output_image_path, overlay_img)
print(f"Corrected shape image saved at: {output_image_path}")