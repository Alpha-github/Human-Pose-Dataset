import cv2
import numpy as np
import os
from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline

ESC_KEY = 27
SAVE_PATH = "captured_ir_images"
os.makedirs(SAVE_PATH, exist_ok=True)

def preprocess_ir_data(ir_data, max_data):
    # Convert IR data to float for processing
    ir_data = ir_data.astype(np.float32)

    # Apply logarithmic compression to reduce brightness extremes
    ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
    ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)

    # Apply CLAHE to enhance dim regions and suppress bright ones
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    ir_data = clahe.apply(ir_data)

    # Apply sharpening (Unsharp Masking)
    blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)  # Gaussian blur
    ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)  # Sharpening formula

    return ir_data

def main():
    config = Config()
    pipeline = Pipeline()
    image_count = 0
    
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
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            
            ir_frame = frames.get_ir_frame()
            if ir_frame is None:
                continue
            
            ir_data = np.asanyarray(ir_frame.get_data())
            width = ir_frame.get_width()
            height = ir_frame.get_height()
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
            ir_data = ir_data.astype(data_type)
            ir_data = preprocess_ir_data(ir_data, max_data)
            ir_image = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)
            
            cv2.imshow("Infrared Viewer", ir_image)
            key = cv2.waitKey(1)
            
            if key == ord('q') or key == ESC_KEY:
                break
            elif key == ord(' '):  # Space bar pressed
                image_path = os.path.join(SAVE_PATH, f"capture_{image_count}.png")
                cv2.imwrite(image_path, ir_image)
                print(f"Saved {image_path}")
                image_count += 1
                
        except KeyboardInterrupt:
            break
    
    pipeline.stop()

if __name__ == "__main__":
    main()