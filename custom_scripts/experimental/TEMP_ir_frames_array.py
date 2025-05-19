import cv2
import numpy as np
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline

ESC_KEY = 27

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

    frames_list = []  # List to store captured frames

    print("Capturing IR frames... Press 'q' or ESC to stop.")

    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            ir_frame = frames.get_ir_frame()
            if ir_frame is None:
                continue

            # Get IR frame data
            ir_data = np.asanyarray(ir_frame.get_data())
            width, height = ir_frame.get_width(), ir_frame.get_height()
            ir_format = ir_frame.get_format()

            if ir_format == OBFormat.Y8:
                ir_data = np.resize(ir_data, (height, width)).astype(np.uint8)
            elif ir_format == OBFormat.MJPG:
                ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED).astype(np.uint8)
                if ir_data is None:
                    print("Decode MJPEG failed")
                    continue
                ir_data = np.resize(ir_data, (height, width))
            else:
                ir_data = np.frombuffer(ir_data, dtype=np.uint16).reshape((height, width))

            frames_list.append(ir_data)  # Store the raw frame in a list

            # Display the raw IR image
            ir_display = cv2.convertScaleAbs(ir_data, alpha=(255.0/65535.0))  # Normalize for visualization
            cv2.imshow("Infrared Viewer", ir_display)

            # Exit condition
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()

    # Convert list of frames to NumPy array and save
    frames_array = np.array(frames_list)
    np.save("ir_frames.npy", frames_array)

    print(f"Saved {len(frames_list)} frames as 'ir_frames.npy'.")

if __name__ == "__main__":
    main()
