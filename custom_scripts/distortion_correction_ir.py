import cv2
import numpy as np
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile

# Load saved calibration parameters
camera_matrix = np.array([[1.80376226e+03,0.00000000e+00,3.16914171e+02],
 [0.00000000e+00,4.31043121e+03,2.87652674e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
 )
dist_coeffs = np.array([[-3.49027957e+00,3.13640856e+01,8.12342618e-02,3.06527531e-02,-3.86407135e+02]])

def undistort_image(image, camera_matrix, dist_coeffs):
    """Applies undistortion using the saved calibration parameters."""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the valid region
    x, y, w, h = roi
    print(w,h)
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def preprocess_ir_data(ir_data, max_data):
    """Preprocess IR data for better visualization."""
    ir_data = ir_data.astype(np.float32)
    ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
    ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    ir_data = clahe.apply(ir_data)
    
    blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)
    ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)
    
    return ir_data

def main():
    config = Config()
    pipeline = Pipeline()
    
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
        try:
            ir_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
        except OBError as e:
            print(e)
            ir_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(ir_profile)
    except Exception as e:
        print(e)
        return
    
    pipeline.start(config)
    
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            ir_frame = frames.get_ir_frame()
            if ir_frame is None:
                continue
            
            ir_data = np.asanyarray(ir_frame.get_data())
            width = ir_frame.get_width()
            height = ir_frame.get_height()
            # print(width, height)
            ir_format = ir_frame.get_format()
            
            if ir_format == OBFormat.Y8:
                ir_data = np.resize(ir_data, (height, width, 1))
                data_type = np.uint8
                image_dtype = cv2.CV_8UC1
                max_data = 255
            elif ir_format == OBFormat.MJPG:
                ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
                data_type = np.uint8
                image_dtype = cv2.CV_8UC1
                max_data = 255
                if ir_data is None:
                    print("decode mjpeg failed")
                    continue
                ir_data = np.resize(ir_data, (height, width, 1))
            else:
                ir_data = np.frombuffer(ir_data, dtype=np.uint16)
                data_type = np.uint16
                image_dtype = cv2.CV_16UC1
                max_data = 65535
                ir_data = np.resize(ir_data, (height, width, 1))
            
            cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
            ir_image = preprocess_ir_data(ir_data, max_data)
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
            
            undistorted_image = undistort_image(ir_image, camera_matrix, dist_coeffs)
            # stacked_image = np.hstack((ir_image, undistorted_image))
            
            cv2.imshow("Infrared - Original (Left) | Undistorted (Right)", undistorted_image)
            
            key = cv2.waitKey(1)
            if key == 27:  # Press 'ESC' to exit
                break
        
        except KeyboardInterrupt:
            break
    
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
