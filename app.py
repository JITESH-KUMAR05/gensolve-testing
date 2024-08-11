import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocess the image: thresholding, blurring, and edge detection
def preprocess_image(img):
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresholded, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    return edges

# Function to detect symmetry lines
def detect_symmetry_axes(img, shape):
    edges = preprocess_image(img)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return [], img
    
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate the moments of the contour to find the centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return [], img
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    img_with_axes = img.copy()
    color = (128,)  # Gray color for the dotted line
    axes = []

    # Symmetry logic based on detected shape
    if shape in ["square", "rectangle", "ellipse"]:
        # For shapes like squares, rectangles, and ellipses, draw both horizontal and vertical symmetry axes
        vertical_axis = ((cx, 0), (cx, img.shape[0]))
        horizontal_axis = ((0, cy), (img.shape[1], cy))
        
        axes.append(("vertical", vertical_axis))
        axes.append(("horizontal", horizontal_axis))
        
        img_with_axes = draw_dotted_line(img_with_axes, *vertical_axis, color)
        img_with_axes = draw_dotted_line(img_with_axes, *horizontal_axis, color)
        
    elif shape == "circle":
        # For a circle, we draw several symmetry lines: horizontal, vertical, and diagonal
        axes.append(("vertical", ((cx, 0), (cx, img.shape[0]))))
        axes.append(("horizontal", ((0, cy), (img.shape[1], cy))))
        axes.append(("diagonal1", ((0, 0), (img.shape[1], img.shape[0]))))
        axes.append(("diagonal2", ((0, img.shape[0]), (img.shape[1], 0))))

        for _, axis in axes:
            img_with_axes = draw_dotted_line(img_with_axes, *axis, color)
        
    elif shape in ["pentagon", "hexagon"]:
        # For pentagon or hexagon, we will approximate symmetry lines from centroid to vertices
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        for vertex in approx:
            axis = ((cx, cy), (vertex[0][0], vertex[0][1]))
            axes.append(("rotational", axis))
            img_with_axes = draw_dotted_line(img_with_axes, *axis, color)
    
    elif shape == "star":
        # A star typically has rotational symmetry, not reflection symmetry, so draw lines connecting opposite vertices
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_points = len(approx)
        
        for i in range(num_points):
            opposite_index = (i + num_points // 2) % num_points
            axis = ((approx[i][0][0], approx[i][0][1]), (approx[opposite_index][0][0], approx[opposite_index][0][1]))
            axes.append(("rotational", axis))
            img_with_axes = draw_dotted_line(img_with_axes, *axis, color)
    
    return axes, img_with_axes

# Function to draw precise dotted lines
def draw_dotted_line(img, start_point, end_point, color, thickness=1, gap=5):
    dist = np.linalg.norm(np.array(end_point) - np.array(start_point))
    points = int(dist // gap)
    
    for i in range(points + 1):
        point = (
            int(start_point[0] + i * (end_point[0] - start_point[0]) / points),
            int(start_point[1] + i * (end_point[1] - start_point[1]) / points),
        )
        if i % 2 == 0:
            cv2.circle(img, point, thickness, color, -1)
    
    return img

# Shape detection function
def detect_shape(img):
    edges = preprocess_image(img)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return "unknown"
    
    # Assume the largest contour is the shape
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate the bounding rectangle to distinguish between circle and ellipse
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    
    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        return "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
    elif num_vertices == 5:
        return "pentagon"
    elif num_vertices == 6:
        return "hexagon"
    else:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        if 0.8 <= circularity <= 1.0 and 0.9 <= aspect_ratio <= 1.1:
            return "circle"
        else:
            return "ellipse" if aspect_ratio < 0.9 or aspect_ratio > 1.1 else "star"

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

# Load and preprocess the uploaded image
img_path = "circle.jpg"  # Use the correct path of your uploaded image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Unable to open image file '{img_path}'. Please check the file path and try again.")
    exit()

img = cv2.resize(img, (128, 128))  # Resize the image for consistent processing

# Detect the shape in the uploaded image
detected_shape = detect_shape(img)

# Generate and visualize the shape image with an outline
corrected_img = generate_shape_image(detected_shape)

# Detect symmetry axes on the detected shape
axes, corrected_img_with_axes = detect_symmetry_axes(corrected_img, detected_shape)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title(f"Detected Shape: {detected_shape}")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Corrected Image with Symmetry Axes")
plt.imshow(corrected_img_with_axes, cmap='gray')
plt.axis('off')

plt.show()


# 3rd part


# Preprocess the image: thresholding, blurring, and edge detection
def preprocess_image(img):
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresholded, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    return edges

# Function to interpolate and complete the curve
def complete_curve(img, contour):
    # Fit an ellipse around the contour to approximate the shape
    if len(contour) >= 5:  # FitEllipse needs at least 5 points
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(img, ellipse, (128,), 1)

    # Attempt to close any open contours by connecting endpoints
    if cv2.isContourConvex(contour):
        hull = cv2.convexHull(contour)
        cv2.drawContours(img, [hull], -1, (128,), 1)
    else:
        # Find endpoints of the curve and connect them
        start_point = tuple(contour[0][0])
        end_point = tuple(contour[-1][0])
        cv2.line(img, start_point, end_point, (128,), 1)

    return img

# Function to complete and correct curves
def complete_incomplete_curves(img):
    edges = preprocess_image(img)
    
    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return img  # No contours found
    
    img_completed = img.copy()
    
    for contour in contours:
        img_completed = complete_curve(img_completed, contour)
    
    return img_completed

# Load and preprocess the uploaded image
img_path = "incomplete-circle.png"  # Use the correct path of your uploaded image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Unable to open image file '{img_path}'. Please check the file path and try again.")
    exit()

img = cv2.resize(img, (128, 128))  # Resize the image for consistent processing

# Complete the incomplete curves
completed_img = complete_incomplete_curves(img)

# Display the original and completed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Incomplete Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Completed Image")
plt.imshow(completed_img, cmap='gray')
plt.axis('off')

plt.show()