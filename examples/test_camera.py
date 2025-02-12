import cv2
import numpy as np

from toddlerbot.sensing.camera import Camera

# This script captures video from the two fisheye cameras and displays them side by side in a single window.
# It also saves the combined video to a file.

# If you want to view the camera feed from the Jetson Orin through an ssh session, you can set up X11 forwarding
# following this guide: https://www.businessnewsdaily.com/11035-how-to-use-x11-forwarding.html.

if __name__ == "__main__":
    # Initialize two Camera instances with different camera IDs
    camera1 = Camera("left")
    camera2 = Camera("right")

    # Define video writer properties
    output_combined_path = "combined_output.mp4"

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30  # Frames per second
    frame_size = (640, 360)  # Ensure frames are resized to this size

    video_writer_combined = cv2.VideoWriter(
        output_combined_path, fourcc, fps, (frame_size[0] * 2, frame_size[1])
    )

    try:
        while True:
            # Get frames from both cameras
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame()

            # Resize frames to the same size if necessary
            frame1_bgr = cv2.resize(frame1, frame_size)
            frame2_bgr = cv2.resize(frame2, frame_size)

            # Concatenate frames horizontally
            combined_frame = np.hstack((frame1_bgr, frame2_bgr))

            # Write the combined frame to its video file
            video_writer_combined.write(combined_frame)

            # Display the combined frame in a single window
            cv2.imshow("Dual Camera Stream", combined_frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Release the cameras and close OpenCV windows
        camera1.close()
        camera2.close()
        video_writer_combined.release()
        cv2.destroyAllWindows()
