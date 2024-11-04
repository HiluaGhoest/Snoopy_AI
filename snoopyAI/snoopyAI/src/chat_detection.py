import cv2
import numpy as np
import pyautogui
import time
import os

def init_opencl():
    """Initialize OpenCL context"""
    try:
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            print("OpenCL is available")
            print(f"OpenCL device: {cv2.ocl.Device.getDefault().name()}")
            return True
        else:
            print("OpenCL is not available")
            return False
    except Exception as e:
        print(f"OpenCL initialization failed: {e}")
        return False

def preprocess_image_opencl(image):
    """OpenCL-accelerated image preprocessing"""
    gpu_image = cv2.UMat(image)
    gpu_gray = cv2.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
    gpu_blur = cv2.GaussianBlur(gpu_gray, (3, 3), 0)
    edges = cv2.Canny(gpu_blur, 30, 100)
    kernel = np.ones((2, 2), np.uint8)
    gpu_dilated = cv2.dilate(edges, kernel)
    return gpu_dilated

def match_template_opencl(image, template, threshold=0.5):
    """OpenCL-accelerated template matching"""
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return result

def find_chat_boundaries(image, x, y, w, h, w_scale, h_scale):
    """Find the actual boundaries of the chat window"""
    # Convert region to grayscale
    region = image[max(0, y-10):min(image.shape[0], y+h+10), max(0, x-10):min(image.shape[1], x+w+10)]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect nearby lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x_offset, y_offset, w_expanded, h_expanded = cv2.boundingRect(largest_contour)
        
        # Adjust coordinates back to full image space and apply scaling
        return (
            max(0, x-10) + x_offset,
            max(0, y-10) + y_offset,
            int(w_expanded * w_scale),  # Scale the width
            int(h_expanded * h_scale)   # Scale the height
        )
    
    return x, y, w, h

def calculate_correction(scale):
    """Calculate correction factor and offset based on scale."""
    # Define correction constants
    default_correction = 1.1  # Correction factor for less than 100%
    max_correction = 0.70    # Correction factor for 140% and above

    if scale < 100:  # For scales less than 100%
        correction_factor = default_correction  # Default correction factor
        x_offset = -20
        y_offset = 0  # Assuming no offset for height in this case
    elif 100 <= scale <= 140:  # For scales between 100% and 140%
        # Linear interpolation between 100% and 140%
        correction_factor = default_correction - ((default_correction - max_correction) / (140 - 100)) * (scale - 100)
        x_offset = -20 + ((0 - (-20)) / (140 - 100)) * (scale - 100)
        y_offset = 0  # Assuming no offset for height
    else:  # For scales greater than 140%
        # Extrapolate for scales above 140%
        correction_factor = max_correction + (max_correction - default_correction) / (140 - 100) * (scale - 140)
        x_offset = 20 * ((scale - 140) / 100)  # Example extrapolation for positive offset
        y_offset = 0  # Adjust as needed for height offset above 140%

    return correction_factor, x_offset, y_offset


def find_chat_on_screen(template_path, output_path, threshold=0.5):
    try:
        if not os.path.exists(template_path):
            print(f"Template file not found: {template_path}")
            return None

        # Capture screenshot
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        screenshot_orig = screenshot.copy()
        screen_height, screen_width = screenshot.shape[:2]

        # Load template
        template = cv2.imread(template_path)
        if template is None:
            print(f"Failed to load template: {template_path}")
            return None
        template_height, template_width = template.shape[:2]

        # Process images with OpenCL
        gpu_screenshot_proc = preprocess_image_opencl(screenshot)
        gpu_template_proc = preprocess_image_opencl(template)

        # Define scaling ranges
        width_scales = np.linspace(0.5, 2.0, 8)
        height_scales = np.linspace(0.5, 2.0, 8)

        matches = []

        for w_scale in width_scales:
            for h_scale in height_scales:
                resized_width = int(template_width * w_scale)
                resized_height = int(template_height * h_scale)

                if resized_height > screen_height or resized_width > screen_width:
                    continue

                try:
                    # Resize template
                    gpu_resized_template = cv2.resize(
                        gpu_template_proc, 
                        (resized_width, resized_height)
                    )

                    # Match template
                    result = match_template_opencl(
                        gpu_screenshot_proc,
                        gpu_resized_template,
                        threshold
                    )
                    # Initialize locations to an empty tuple
                    locations = (np.array([]), np.array([]))

                    # Match template
                    result = match_template_opencl(
                        gpu_screenshot_proc,
                        gpu_resized_template,
                        threshold
                    )

                    result_cpu = result.get()
                    locations = np.where(result_cpu >= threshold)

                    # Check if there are any locations found
                    if locations[0].size > 0:  # If there are any matches
                        for y, x in zip(*locations):
                            # Find actual chat boundaries with scaling factors
                            actual_x, actual_y, actual_w, actual_h = find_chat_boundaries(
                                screenshot_orig,
                                x,
                                y,
                                resized_width,
                                resized_height,
                                w_scale,
                                h_scale
                            )

                            matches.append({
                                'position': (actual_x, actual_y),
                                'width_scale': w_scale,
                                'height_scale': h_scale,
                                'confidence': result_cpu[y, x],
                                'size': (actual_w, actual_h)
                            })

                except cv2.error as e:
                    print(f"OpenCV error during template matching: {e}")
                    continue

        # In the find_chat_on_screen function
        if matches:
            matches.sort(key=lambda x: x['confidence'], reverse=True)

            match = matches[0]
            actual_x, actual_y = match['position']
            actual_w, actual_h = match['size']

            # Get the correction factor and offsets based on the last scales used
            current_scale = int(match['width_scale'] * 100)  # Use the last matched width scale
            correction_factor, x_offset, y_offset = calculate_correction(current_scale)

            # Calculate adjusted width and height using the correction factor
            adjusted_w = int(actual_w * correction_factor)
            adjusted_h = int(actual_h * correction_factor)

            # Adjust the position with the x and y offsets
            actual_x = int(actual_x + x_offset)
            actual_y = int(actual_y + y_offset)

            # Ensure the rectangle does not exceed image boundaries
            adjusted_w = min(adjusted_w, screenshot_orig.shape[1] - actual_x)
            adjusted_h = min(adjusted_h, screenshot_orig.shape[0] - actual_y)

            # Ensure that all values used are integers
            actual_x = max(0, actual_x)
            actual_y = max(0, actual_y)
            adjusted_w = max(1, adjusted_w)  # Minimum width of 1 pixel
            adjusted_h = max(1, adjusted_h)  # Minimum height of 1 pixel

            # Draw the adjusted rectangle
            cv2.rectangle(
                screenshot_orig,
                (actual_x, actual_y),
                (actual_x + adjusted_w, actual_y + adjusted_h),
                (0, 255, 0),
                2
            )

            # Add confidence text
            cv2.putText(
                screenshot_orig,
                f"Conf: {match['confidence']:.2f}",
                (actual_x, actual_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Print the adjusted values for debugging
            print(f"Scale: {current_scale}%")
            print(f"Actual Width: {actual_w}, Actual Height: {actual_h}")
            print(f"Adjusted Width: {adjusted_w}, Adjusted Height: {adjusted_h}")
            print(f"Adjusted X Offset: {x_offset}, Adjusted Y Offset: {y_offset}")

            cv2.imwrite(output_path, screenshot_orig)
            print(f"Found match with confidence: {match['confidence']:.2f}")




        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    if not init_opencl():
        print("OpenCL acceleration not available. Exiting...")
        return

    if not os.path.exists('templates'):
        os.makedirs('templates')

    template_paths = {
        'chat_window': 'templates/chat_template.png',
    }

    output_path = 'chat_detection_output.png'

    print("Starting OpenCL-accelerated chat detection...")
    try:
        while True:
            for template_name, template_path in template_paths.items():
                start_time = time.time()
                match = find_chat_on_screen(template_path, output_path)
                end_time = time.time()
                
                if match:
                    print(f"Found {template_name}! Processing time: {(end_time - start_time) * 1000:.2f}ms")
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
