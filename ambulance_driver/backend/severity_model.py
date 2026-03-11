import cv2
import numpy as np

def analyze_image(img):

    severity = 0

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect red regions (simulate bleeding detection)
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 + mask2

    red_pixels = cv2.countNonZero(red_mask)

    total_pixels = img.shape[0] * img.shape[1]

    red_ratio = red_pixels / total_pixels

    # Simulated severity logic
    if red_ratio > 0.10:
        severity += 5

    elif red_ratio > 0.05:
        severity += 3

    else:
        severity += 1

    # Brightness check (simulate unconscious / dark scene)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 50:
        severity += 3

    # Texture/edge intensity (simulate trauma complexity)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / total_pixels

    if edge_density > 20:
        severity += 2

    return min(severity, 10)