import sys
import pyautogui
from PIL import Image
import cv2
import numpy as np

# Split the single comma-separated string into individual values
outer_bounds_str = sys.argv[1]
inner_bounds_str = sys.argv[2]

# Convert the comma-separated strings into lists of integers
outer_bounds = list(map(int, outer_bounds_str.split(',')))
inner_bounds = list(map(int, inner_bounds_str.split(',')))

print("Outer Bounds:", outer_bounds)
print("Inner Bounds:", inner_bounds)

# Define the bounds of the outer area
x1, y1, x2, y2 = outer_bounds

# Capture the screenshot within the outer bounds
screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))

# Convert the screenshot to a format usable by OpenCV
screenshot_np = np.array(screenshot)
screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

# Load template images
template1 = cv2.imread('templates/top_left.png', cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread('templates/bottom_right.png', cv2.IMREAD_GRAYSCALE)

# Perform template matching for the first template
result1 = cv2.matchTemplate(screenshot_gray, template1, cv2.TM_CCOEFF_NORMED)
min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(result1)

# Perform template matching for the second template
result2 = cv2.matchTemplate(screenshot_gray, template2, cv2.TM_CCOEFF_NORMED)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)

# Define a threshold for detecting a match
threshold = 0.5

# Check if templates match within the threshold
if max_val1 >= threshold:
    print(f"Template 1 found at {max_loc1} with confidence {max_val1:.2f}")
    # Draw a rectangle around the detected template
    h1, w1 = template1.shape
    cv2.rectangle(screenshot_np, max_loc1, (max_loc1[0] + w1, max_loc1[1] + h1), (255, 0, 0), 2)

if max_val2 >= threshold:
    print(f"Template 2 found at {max_loc2} with confidence {max_val2:.2f}")
    # Draw a rectangle around the detected template
    h2, w2 = template2.shape
    cv2.rectangle(screenshot_np, max_loc2, (max_loc2[0] + w2, max_loc2[1] + h2), (0, 255, 0), 2)

# Convert back to RGB for displaying with OpenCV
screenshot_display = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)

# Optionally save the result
cv2.imwrite('detected_templates.png', screenshot_np)

# Perform template matching
template1_found = max_val1 >= threshold
template2_found = max_val2 >= threshold

if template1_found or template2_found:
    print("Template detected. Capturing inner area screenshot...")

    # Capture the inner area screenshot
    x1_inner, y1_inner, x2_inner, y2_inner = inner_bounds  # Coordinates of the inner area
    inner_screenshot = pyautogui.screenshot(region=(x1_inner, y1_inner, x2_inner - x1_inner, y2_inner - y1_inner))

    # Convert to a format usable by OpenCV
    inner_screenshot_np = np.array(inner_screenshot)

    # Optionally, save the screenshot
    cv2.imwrite('inner_area_screenshot.png', cv2.cvtColor(inner_screenshot_np, cv2.COLOR_RGB2BGR))
    print("Inner area screenshot saved as 'inner_area_screenshot.png'.")

    # Optionally display the screenshot
    cv2.imshow('Inner Area Screenshot', cv2.cvtColor(inner_screenshot_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No template detected in the outer area. No inner area screenshot captured.")