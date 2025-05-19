import cv2
import numpy as np
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline


ESC_KEY = 27

def preprocess_ir_data(ir_data, max_data, enhance=True):
    # Convert IR data to float for processing
    ir_data = ir_data.astype(np.float32)

    if enhance:
        # Apply logarithmic compression to reduce brightness extremes
        ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
        ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)

        # Apply CLAHE to enhance dim regions and suppress bright ones
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
        ir_data = clahe.apply(ir_data)

        # Apply sharpening (Unsharp Masking)
        blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)  # Gaussian blur
        ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)  # Sharpening formula

    else:
        ir_data = ir_data/max_data * 255
        print("in")
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

    VIDEO_OUTPUT = "TEMPORARY_IR_VIDEO.avi"
    FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 576
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for .avi files
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

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
            ir_image = ir_data.astype(data_type)
            # print(np.max(ir_image), np.min(ir_image))
            # ir_image = preprocess_ir_data(ir_image, max_data)
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
            
            cv2.imshow("Infrared Viewer", ir_image)
            print(ir_image.shape)

            out.write(ir_image)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()