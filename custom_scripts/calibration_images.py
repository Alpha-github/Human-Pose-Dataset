import cv2
import numpy as np
import glob
# import yaml

# Chessboard settings (modify based on your calibration board)
CHESSBOARD_SIZE = (4, 6)  # (Columns, Rows)
SQUARE_SIZE = 20  # Size of a square in mm or cm (for real-world scaling)

# Path to the folder containing calibration images
IMAGE_FOLDER = "calibration_ir_images-unique"

def calibrate_camera():
    """Perform camera calibration using images from a folder"""
    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image plane

    # Prepare a grid of 3D object points (assuming Z=0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale points to real-world units

    # Get all image file paths
    images = glob.glob(f"{IMAGE_FOLDER}/*.jpg")
    print(len(images))

    if not images:
        print("❌ No calibration images found!")
        return

    print(f"📸 Found {len(images)} images for calibration.")

    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
            cv2.imshow("Calibration", img)
            cv2.waitKey(500)
        else:
            print(f"⚠️ Chessboard not found in {img_path}")

    cv2.destroyAllWindows()

    if len(obj_points) < 1:
        print("❌ Not enough valid images for calibration!")
        return

    # Calibrate camera
    print("🔧 Calibrating camera...")
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    if ret:
        print("\n🎯 Camera Calibration Successful!")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", distortion_coeffs)

        # Save calibration parameters
        calib_data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coeffs": distortion_coeffs.tolist(),
            "rvecs": [r.tolist() for r in rvecs],
            "tvecs": [t.tolist() for t in tvecs],
        }


        print(f"✅ Calibration parameters saved to '{calib_data}'")
    else:
        print("❌ Calibration failed!")

if __name__ == "__main__":
    calibrate_camera()
