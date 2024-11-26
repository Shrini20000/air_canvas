import cv2
import numpy as np
import mediapipe as mp

from collections import deque
import tkinter as tk
from tkinter import simpledialog

# Initialize the Tkinter root
root = tk.Tk()
root.withdraw()

# Ask for initial brush size
brush_size = simpledialog.askinteger("Brush Size", "Enter initial brush size:", minvalue=1, maxvalue=20)

# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Variables for rectangle drawing
rectangle_mode = False
rectangle_start = None
rectangle_end = None

# Here is code for Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (1, 40), (65, 140), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (1, 160), (65, 255), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (1, 275), (65, 370), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (1, 390), (65, 485), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (1, 505), (65, 600), (0, 255, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (571, 1), (635, 65), (0, 0, 0), 2)

cv2.putText(paintWindow, "CLEAR", (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (10, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (10, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RECT", (576, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (1, 40), (65, 140), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (1, 160), (65, 255), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (1, 275), (65, 370), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (1, 390), (65, 485), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (1, 505), (65, 600), (0, 255, 255), 2)
    frame = cv2.rectangle(frame, (571, 1), (635, 65), (0, 0, 0), 2)
    cv2.putText(frame, "CLEAR", (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (10, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (10, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RECT" if rectangle_mode else "FREE", (576, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        if thumb[1] - center[1] < 30:
            if rectangle_mode:
                if rectangle_start is None:
                    rectangle_start = center
                else:
                    rectangle_end = center
                    cv2.rectangle(frame, rectangle_start, rectangle_end, colors[colorIndex], 2)
                    cv2.rectangle(paintWindow, rectangle_start, rectangle_end, colors[colorIndex], 2)
            else:
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

        elif center[0] <= 65:
            if 40 <= center[1] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:, :, :] = 255
            elif 160 <= center[1] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[1] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[1] <= 485:
                colorIndex = 2  # Red
            elif 505 <= center[1] <= 600:
                colorIndex = 3  # Yellow
        elif 571 <= center[0] <= 635 and 1 <= center[1] <= 65:
            rectangle_mode = not rectangle_mode
            if not rectangle_mode:
                rectangle_start = None
                rectangle_end = None
        else:
            if rectangle_mode:
                if rectangle_start and rectangle_end:
                    cv2.rectangle(frame, rectangle_start, rectangle_end, colors[colorIndex], 2)
                    cv2.rectangle(paintWindow, rectangle_start, rectangle_end, colors[colorIndex], 2)
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], brush_size)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], brush_size)

    # Display the frame and the paint window
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Key bindings
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the current paint window
        filename = simpledialog.askstring("Save", "Enter the filename to save as (with .jpg or .png):")
        if filename:
            cv2.imwrite(filename, paintWindow)
    elif key == ord('+'):
        brush_size += 1
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)
    elif key == ord('r'):
        rectangle_mode = not rectangle_mode
        if not rectangle_mode:
            rectangle_start = None
            rectangle_end = None

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()