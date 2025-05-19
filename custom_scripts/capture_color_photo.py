import cv2
import os
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from pyorbbecsdk.examples.utils import frame_to_bgr_image  # Ensure utils.py is available

ESC_KEY = 27
SAVE_FOLDER = "captured_color_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)  # Ensure folder exists

def capture_color_images():
    config = Config()
    pipeline = Pipeline()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("Using default color profile:", color_profile)
        
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    pipeline.start(config)
    image_counter = 1  # Track number of images captured

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            # Convert frame to BGR image
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert frame to image")
                continue

            # Show the live color feed
            cv2.imshow("Color Viewer - Press SPACE to capture", color_image)

            key = cv2.waitKey(1)

            if key == ord(' '):  # Press SPACE to save an image
                img_path = os.path.join(SAVE_FOLDER, f"image_{image_counter}.png")
                cv2.imwrite(img_path, color_image)
                print(f"Captured: {img_path}")
                image_counter += 1

            elif key == ESC_KEY or key == ord('q'):  # Exit
                print("Exiting...")
                break

        except KeyboardInterrupt:
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_color_images()
