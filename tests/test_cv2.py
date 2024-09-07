import time

import cv2

# Open the camera (0 is the default camera)
cap = cv2.VideoCapture(0)
# Set the resolution to 640x360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture frames in a loop
while True:
    # Capture frame-by-frame
    t1 = time.time()
    ret, frame = cap.read()

    # # resize frame to 171x96
    frame = cv2.resize(frame, (171, 96))
    # # crop center part of frame to 96x96
    frame = frame[0:96, 38:134]

    # If frame was not captured successfully, break the loop
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    # Display the resulting frame
    cv2.imshow("Camera Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    t2 = time.time()
    print(f"Capture frame: {t2 - t1:.2f} s")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
