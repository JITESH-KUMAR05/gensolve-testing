import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Load the uploaded image
img_path = "star.jpg"  # Use the correct path of your uploaded image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Unable to open image file '{img_path}'. Please check the file path and try again.")
    exit()

img = cv2.resize(img, (128, 128))  # Resize the image for consistent processing

# Preprocess the image
def preprocess_image(img):
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresholded, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    return edges

# Shape detection function
def detect_shape(img):
    edges = preprocess_image(img)
    
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
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
    elif num_vertices == 5:
        return "pentagon"
    elif num_vertices == 6:
        return "hexagon"
    else:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity > 0.8:
            return "circle"
        elif circularity > 0.6:
            return "ellipse"
        else:
            return "star"

# Generate shape image using OpenCV functions
def generate_shape_image(shape, img_size=(128, 128)):
    img = np.ones(img_size, dtype=np.uint8) * 255  # Initialize with white background
    
    if shape == "star":
        control_points = np.array([[64, 10], [74, 40], [104, 40], [80, 60], [90, 90], [64, 70], [38, 90], [48, 60], [24, 40], [54, 40]])
        cv2.polylines(img, [control_points], True, 0, 2)
    elif shape == "pentagon":
        control_points = np.array([[64, 10], [104, 40], [84, 90], [44, 90], [24, 40]])
        cv2.polylines(img, [control_points], True, 0, 2)
    elif shape == "hexagon":
        control_points = np.array([[64, 10], [104, 40], [104, 80], [64, 110], [24, 80], [24, 40]])
        cv2.polylines(img, [control_points], True, 0, 2)
    elif shape == "circle":
        center = (img_size[0]//2, img_size[1]//2)
        radius = min(img_size)//3
        cv2.circle(img, center, radius, 0, 2)
    elif shape == "square":
        side = int(min(img_size) // 1.5)
        start = int((img_size[0] - side) // 2)
        end = int(start + side)
        cv2.rectangle(img, (start, start), (end, end), 0, 2)
    elif shape == "rectangle":
        width = int(img_size[0] // 1.5)
        height = int(img_size[1] // 2)
        start_x = int((img_size[0] - width) // 2)
        start_y = int((img_size[1] - height) // 2)
        cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), 0, 2)
    elif shape == "ellipse":
        center = (img_size[0]//2, img_size[1]//2)
        axes = (min(img_size)//3, min(img_size)//4)
        cv2.ellipse(img, center, axes, 0, 0, 360, 0, 2)
    
    return img

# Detect the shape in the uploaded image
detected_shape = detect_shape(img)

# Generate and visualize the shape image with an outline
corrected_img = generate_shape_image(detected_shape)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Uploaded Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Detected Shape: {detected_shape} with Outline")
plt.imshow(corrected_img, cmap='gray')

plt.show()
