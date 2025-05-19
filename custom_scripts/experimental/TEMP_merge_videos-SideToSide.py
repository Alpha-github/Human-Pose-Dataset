import cv2
import numpy as np

# Paths to input videos
video_paths = [r'Recordings\Pranav\color_video_04-08-25_15-20-54.avi', r'Recordings\Pranav\ir_video_04-08-25_15-20-54.avi', r'Custom_Results/overlay.avi']

# Open the videos
caps = [cv2.VideoCapture(path) for path in video_paths]

# Get the minimum FPS from all videos
fps = min([cap.get(cv2.CAP_PROP_FPS) for cap in caps])

# Target dimensions
target_height = 1080
target_width = 640  # Each video will be resized to 640x1080

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('merged_output_1920x1080.avi', fourcc, fps, (1920, 1080))

# Reset video capture positions
for cap in caps:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 640x1080
        resized = cv2.resize(frame, (target_width, target_height))
        frames.append(resized)

    # If any video ends early, break
    if len(frames) != len(caps):
        print("Issue")
        break

    # Stack horizontally
    merged_frame = np.hstack(frames)
    cv2.imshow('Merged Video', merged_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(merged_frame)

# Release resources
for cap in caps:
    cap.release()
out.release()
cv2.destroyAllWindows()
