import cv2
import numpy as np
import time
import math

# ---------------------------
# Utility functions
# ---------------------------

def clamp(value, low, high):
    return max(low, min(value, high))

def distance_point_to_rect(px, py, rx1, ry1, rx2, ry2):
    """
    Compute shortest distance from point (px, py) to axis-aligned rectangle
    with corners (rx1, ry1) top-left and (rx2, ry2) bottom-right.
    """
    qx = clamp(px, rx1, rx2)
    qy = clamp(py, ry1, ry2)
    return math.hypot(px - qx, py - qy)

# ---------------------------
# Initialize camera
# ---------------------------

cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Read one frame to get dimensions and define virtual object
ret, frame = cap.read()
if not ret:
    print("Error: Could not read from camera.")
    cap.release()
    exit(1)

h, w, _ = frame.shape

# Define virtual object (rectangle) in the middle of the screen
rect_width = int(w * 0.25)
rect_height = int(h * 0.25)
rect_x1 = w // 2 - rect_width // 2
rect_y1 = h // 2 - rect_height // 2
rect_x2 = rect_x1 + rect_width
rect_y2 = rect_y1 + rect_height

# Distance thresholds (in pixels) – you may tune these
SAFE_THRESHOLD = int(min(w, h) * 0.35)
DANGER_THRESHOLD = int(min(w, h) * 0.15)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural mirror-like interaction
    frame = cv2.flip(frame, 1)

    # ---------------------------
    # 1. Skin color segmentation
    # ---------------------------
    # Convert to HSV (you may also try YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rough skin color range in HSV (needs tuning per lighting/skin tone)
    # These are example values for lighter to medium skin in normal lighting.
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Noise reduction
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # ---------------------------
    # 2. Find largest contour (hand)
    # ---------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand_point = None  # (x, y) of fingertip / topmost point

    if contours:
        # Largest contour by area
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Filter tiny noisy blobs
        if area > 2000:
            # Draw contour for visualization
            cv2.drawContours(frame, [largest], -1, (255, 0, 0), 2)

            # Topmost point (minimum y) -> approximated fingertip / closest to top
            topmost = tuple(largest[largest[:, :, 1].argmin()][0])
            hand_point = topmost

            cv2.circle(frame, hand_point, 8, (0, 255, 255), -1)
            cv2.putText(frame, "Hand", (hand_point[0] + 10, hand_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ---------------------------
    # 3. Draw virtual object (rectangle)
    # ---------------------------
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
    cv2.putText(frame, "VIRTUAL OBJECT", (rect_x1, rect_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------------------------
    # 4. Distance + state logic
    # ---------------------------
    state = "NO HAND"
    state_color = (255, 255, 255)

    if hand_point is not None:
        hx, hy = hand_point

        d = distance_point_to_rect(hx, hy, rect_x1, rect_y1, rect_x2, rect_y2)

        # Classify state based on distance
        if d > SAFE_THRESHOLD:
            state = "SAFE"
            state_color = (0, 255, 0)
        elif d > DANGER_THRESHOLD:
            state = "WARNING"
            state_color = (0, 255, 255)
        else:
            state = "DANGER"
            state_color = (0, 0, 255)

        # Optionally show distance
        cv2.putText(frame, f"d = {int(d)}", (hx + 10, hy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

    # ---------------------------
    # 5. Overlay state + DANGER message
    # ---------------------------
    cv2.putText(frame, f"STATE: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)

    if state == "DANGER":
        cv2.putText(frame, "DANGER DANGER", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ---------------------------
    # 6. FPS estimation
    # ---------------------------
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-8)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # ---------------------------
    # Show windows
    # ---------------------------
    cv2.imshow("Hand Tracking - POC", frame)
    cv2.imshow("Skin Mask (Debug)", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
