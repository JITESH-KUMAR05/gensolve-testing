import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded img
img_path = "circle.jpg"  # Change this to the path of your input img
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the img was loaded successfully
if img is None:
    print(f"Error: Unable to open img file '{img_path}'. Please check the file path and try again.")
    exit()

img = cv2.resize(img, (128, 128))  # Increase resolution

# Preprocess the img
def preprocess_img(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Shape detection function
def detect_shape(img):
    edges = preprocess_img(img)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return "unknown"
    
    # Assume the largest contour is the shape
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        # Check if the shape is a square or rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "square"
        else:
            return "rectangle"
    elif num_vertices == 5:
        return "pentagon"
    elif num_vertices == 6:
        return "hexagon"
    elif num_vertices > 10:
        # Further check for circle by matching shape
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "unknown"
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity > 0.8:  # A threshold for circularity, close to 1 for a perfect circle
            return "circle"
        else:
            return "ellipse"
    else:
        return "star"  # Assuming shapes with more than 6 but less than 10 vertices are stars

# Generate reference shape with a black outline and white background
def generate_shape_img_with_outline(shape, img_size=(128, 128)):
    img = np.ones(img_size, dtype=np.uint8) * 255  # Initialize with white background
    
    if shape == "circle":
        center = (img_size[0]//2, img_size[1]//2)
        radius = min(img_size)//3
        cv2.circle(img, center, radius, 255, -1)  # White filled circle
        cv2.circle(img, center, radius, 0, 2)    # Black outline
    elif shape == "square":
        side = int(min(img_size) // 1.5)
        start = int((img_size[0] - side) // 2)
        end = int(start + side)
        cv2.rectangle(img, (start, start), (end, end), 255, -1)  # White filled square
        cv2.rectangle(img, (start, start), (end, end), 0, 2)    # Black outline
    elif shape == "rectangle":
        width = int(img_size[0] // 1.5)
        height = int(img_size[1] // 2)
        start_x = int((img_size[0] - width) // 2)
        start_y = int((img_size[1] - height) // 2)
        cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), 255, -1)  # White filled rectangle
        cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), 0, 2)    # Black outline
    elif shape == "star":
        # Draw a star shape
        points = np.array([[64, 10], [74, 40], [104, 40], [80, 60], [90, 90], [64, 70], [38, 90], [48, 60], [24, 40], [54, 40]], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [points], 255)
        cv2.polylines(img, [points], True, 0, 2)
    elif shape == "pentagon":
        points = np.array([[64, 10], [104, 40], [84, 90], [44, 90], [24, 40]], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [points], 255)
        cv2.polylines(img, [points], True, 0, 2)
    elif shape == "hexagon":
        points = np.array([[64, 10], [104, 40], [104, 80], [64, 110], [24, 80], [24, 40]], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [points], 255)
        cv2.polylines(img, [points], True, 0, 2)
    elif shape == "ellipse":
        center = (img_size[0]//2, img_size[1]//2)
        axes = (min(img_size)//3, min(img_size)//4)
        cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)  # White filled ellipse
        cv2.ellipse(img, center, axes, 0, 0, 360, 0, 2)    # Black outline
    
    return img

# User input
user_input_shape = detect_shape(img)  # Change this to the shape you expect, e.g., "star"

# Detect the shape in the uploaded img
detected_shape = detect_shape(img)

# Check if the detected shape matches the user's input
if detected_shape == user_input_shape:
    corrected_img = generate_shape_img_with_outline(detected_shape)
    print(f"Detected shape: {detected_shape}. Generating {detected_shape} with outline.")
    
    # Save the corrected img
    corrected_img_path = "corrected_shape_with_outline.jpg"
    cv2.imwrite(corrected_img_path, corrected_img)
    print(f"Corrected img saved at: {corrected_img_path}")
else:
    print(f"Detected shape '{detected_shape}' does not match the user's input '{user_input_shape}'.")
    # Handle this situation as you prefer, e.g., generate the user-specified shape:
    corrected_img = generate_shape_img_with_outline(user_input_shape)
    corrected_img_path = f"{user_input_shape}_with_outline.jpg"
    cv2.imwrite(corrected_img_path, corrected_img)
    print(f"User-specified shape '{user_input_shape}' generated and saved at: {corrected_img_path}")

# Visualize the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Uploaded img")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Shape img ({user_input_shape}) with Outline")
plt.imshow(corrected_img, cmap='gray')

plt.show()

# Save the final img
output_img_path = "corrected_shape_with_outline.png"
cv2.imwrite(output_img_path, corrected_img)
print(f"Corrected shape img with outline saved at: {output_img_path}")