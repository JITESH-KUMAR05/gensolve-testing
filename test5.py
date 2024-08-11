import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
img_path = "circle.jpg"  # Update this to the path of your input image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Unable to open image file '{img_path}'. Please check the file path and try again.")
    exit()

img = cv2.resize(img, (128, 128))  # Resize to a smaller resolution

# Display the original image
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Preprocess the image
def preprocess_img(img):
    # Use CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    plt.imshow(img, cmap='gray')
    plt.title('After CLAHE')
    plt.axis('off')
    plt.show()
    
    # Apply median blur to reduce noise
    img = cv2.medianBlur(img, 5)
    plt.imshow(img, cmap='gray')
    plt.title('After Median Blur')
    plt.axis('off')
    plt.show()
    
    # Apply Canny edge detection with adjusted parameters
    edges = cv2.Canny(img, 30, 100)
    plt.imshow(edges, cmap='gray')
    plt.title('After Canny Edge Detection')
    plt.axis('off')
    plt.show()
    
    # Apply a simple dilation to make edges thicker
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    plt.imshow(edges, cmap='gray')
    plt.title('After Dilation')
    plt.axis('off')
    plt.show()
    
    return edges

# Shape detection function
def detect_shape(img):
    edges = preprocess_img(img)
    
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contours found.")
        return "unknown"
    
    # Display contours for debugging
    contour_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    plt.imshow(contour_img)
    plt.title('Contours')
    plt.axis('off')
    plt.show()
    
    # Assume the largest contour is the shape
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    num_vertices = len(approx)
    print(f"Number of vertices detected: {num_vertices}")
    
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
    elif num_vertices == 8:
        return "octagon"
    elif num_vertices > 8:
        # Check for circular or elliptical shape
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity > 0.75:
            return "circle"
        else:
            return "ellipse"
    else:
        return "unknown"

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
    elif shape == "octagon":
        points = np.array([[64, 10], [94, 20], [104, 50], [94, 80], [64, 90], [34, 80], [24, 50], [34, 20]], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [points], 255)
        cv2.polylines(img, [points], True, 0, 2)
    elif shape == "ellipse":
        center = (img_size[0]//2, img_size[1]//2)
        axes = (min(img_size)//3, min(img_size)//4)
        cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)  # White filled ellipse
        cv2.ellipse(img, center, axes, 0, 0, 360, 0, 2)    # Black outline
    
    return img

# Detect shape in the input image
detected_shape = detect_shape(img)
print(f"Detected Shape: {detected_shape}")

# Generate reference shape image with outline based on detected shape
output_img = generate_shape_img_with_outline(detected_shape)

# Display the input and output images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Output Image')
plt.imshow(output_img, cmap='gray')
plt.axis('off')

plt.show()
