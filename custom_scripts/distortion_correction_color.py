import cv2
import numpy as np
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image

# Load saved calibration parameters
camera_matrix = np.array([[1.12152278e+03,0.00000000e+00,9.82250784e+02],
 [0.00000000e+00,1.11430049e+03,5.39308610e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist_coeffs = np.array([[ 0.12010729,-0.24954568,0.00151861,0.01197723,0.18355586]])

def undistort_image(image, camera_matrix, dist_coeffs):
    """Applies undistortion using the saved calibration parameters."""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the valid region
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def main():
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

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            # Convert to BGR format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert frame to image")
                continue

            # Resize undistorted image to match original image height
            undistorted_image = cv2.resize(color_image, (color_image.shape[1], color_image.shape[0]))

            # Stack original and undistorted images side by side
            stacked_image = np.hstack((color_image, undistorted_image))


            # Show both images
            cv2.imshow("Femto Camera - Original (Left) | Undistorted (Right)", stacked_image)

            key = cv2.waitKey(1)
            if key == 27:  # Press 'ESC' to exit
                break

        except KeyboardInterrupt:
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
