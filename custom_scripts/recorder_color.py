import cv2
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image

ESC_KEY = 27
OUTPUT_FILENAME = "output.avi"  # Change to .avi if needed
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720  # Set based on your camera
FPS = 30

def main():
    config = Config()
    pipeline = Pipeline()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(FRAME_WIDTH, 0, OBFormat.RGB, FPS)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("Using default color profile:", color_profile)

        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    pipeline.start(config)

    # Try MP4 first (H.264), otherwise fallback to AVI (MJPEG)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' for .avi if needed
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"Recording started. Saving as {OUTPUT_FILENAME}. Press 'q' or ESC to stop.")

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert frame to image")
                continue

            print(color_image.shape)

            out.write(color_image)
            cv2.imshow("Color Viewer", color_image)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    print(f"Recording stopped. Video saved as {OUTPUT_FILENAME}")

    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
