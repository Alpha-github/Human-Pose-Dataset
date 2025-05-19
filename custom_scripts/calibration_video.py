import cv2
import numpy as np

# Chessboard settings
CHESSBOARD_SIZE = (4, 6)  # (Columns, Rows)
SQUARE_SIZE = 20  # Real-world square size in mm or cm

# Video source (0 for webcam, or provide a video file path)
VIDEO_SOURCE = "infrared_record.avi"  # Change to "video.mp4" for a recorded video
STORE_PICS = 'captured_ir_images'  # Folder to store captured images
# Number of frames to capture before calibrating


def calibrate_camera(color_capture=False,ir_capture=False):
    """Perform camera calibration using frames from a video source."""
    obj_points = []  # 3D real-world points
    img_points = []  # 2D image points
    frame_count = 0

    # Prepare object points (Z=0 plane)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    # MAX_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)//2)
    MAX_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("❌ Error: Unable to open video source!")
        return

    print("🎥 Capturing frames for calibration... Press 'q' to stop.")

    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(frame, CHESSBOARD_SIZE, None)

        if found:
            obj_points.append(objp)
            img_points.append(corners)
            frame_count += 1

            if color_capture:
                cv2.imwrite(f"{color_capture}/color_{frame_count}.jpg", frame)
            if ir_capture:
                cv2.imwrite(f"{ir_capture}/ir_{frame_count}.jpg", frame)

            # Draw corners
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, found)
            

        cv2.putText(frame, f"Frames Captured: {frame_count}/{MAX_FRAMES}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 1:
        print("❌ Not enough frames for calibration!")
        return

    # Perform calibration
    print("🔧 Calibrating camera...")
    print(frame.shape[::-1])
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (frame.shape[1], frame.shape[0]), None, None
    )

    if ret:
        print("\n🎯 Camera Calibration Successful!")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", distortion_coeffs)
    else:
        print("❌ Calibration failed!")

# def calibration(obj_points, img)

if __name__ == "__main__":
    calibrate_camera(ir_capture=STORE_PICS)
