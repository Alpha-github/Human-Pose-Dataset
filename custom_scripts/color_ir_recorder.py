import cv2
import numpy as np
import threading
import queue
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image
import datetime
import os

ESC_KEY = 27
color_queue = queue.Queue()
ir_queue = queue.Queue()
stop_event = threading.Event()

datetime_str = datetime.datetime.now().strftime("%x_%X").replace("/", "-").replace(":", "-")
print("Recording started at:", datetime_str)

def preprocess_ir_data(ir_data, max_data):
    ir_data = ir_data.astype(np.float32)
    ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
    ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    ir_data = clahe.apply(ir_data)
    blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)
    ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)
    return ir_data

def record_color(folder,datetime_str):
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1920, 0, OBFormat.RGB, 30)
        # color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    pipeline.start(config)
    out = cv2.VideoWriter(f'.\\Recordings\\{folder}\\color_video_{datetime_str}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,
                           (color_profile.get_width(), color_profile.get_height()))

    while not stop_event.is_set():
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
            color_queue.put(color_image)
            out.write(color_image)
        except Exception as e:
            print(f"Error in color thread: {e}")
            break

    out.release()
    pipeline.stop()


def record_ir(folder,datetime_str):
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

    VIDEO_OUTPUT = f".\\Recordings\\{folder}\\ir_video_{datetime_str}.avi"
    FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 576
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for .avi files
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    while not stop_event.is_set():
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
            ir_image = preprocess_ir_data(ir_image, max_data)
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
            ir_queue.put(ir_image)
            out.write(ir_image)
        except Exception as e:
            print(f"Error in IR thread: {e}")
            break

    out.release()
    pipeline.stop()

def display_frames():
    while not stop_event.is_set():
        if not color_queue.empty():
            color_frame = color_queue.get()
            cv2.imshow("Color Viewer", color_frame)
        if not ir_queue.empty():
            ir_frame = ir_queue.get()
            cv2.imshow("Infrared Viewer", ir_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    subject = input("Enter subject name: ")
    if subject not in os.listdir(".\\Recordings"):
        create_folder = input("Folder NOT FOUND! Do you want to create subject folder (Y/N):")
        if create_folder.lower() == "y":
            os.mkdir(f".\\Recordings\{subject}")
        else:
            print("Exiting...")
            exit()
        
    t1 = threading.Thread(target=record_color, args=(subject,datetime_str,))
    t2 = threading.Thread(target=record_ir, args=(subject,datetime_str,))
    t3 = threading.Thread(target=display_frames)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()



# import cv2
# import numpy as np
# import threading
# import queue
# from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
# from utils import frame_to_bgr_image
# import datetime
# import os
# import time

# ESC_KEY = 27
# color_queue = queue.Queue()
# ir_queue = queue.Queue()
# stop_event = threading.Event()

# # Timestamped queues for sync
# synced_color_queue = queue.Queue()
# synced_ir_queue = queue.Queue()

# datetime_str = datetime.datetime.now().strftime("%x_%X").replace("/", "-").replace(":", "-")
# print("Recording started at:", datetime_str)

# SYNC_TOLERANCE = 0.015  # 15ms
# # start_time = time.time()

# def preprocess_ir_data(ir_data, max_data):
#     ir_data = ir_data.astype(np.float32)
#     ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
#     ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
#     ir_data = clahe.apply(ir_data)
#     blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)
#     ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)
#     return ir_data

# def record_color():
#     config = Config()
#     pipeline = Pipeline()
#     try:
#         profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
#         color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1920, 0, OBFormat.RGB, 30)
#         config.enable_stream(color_profile)
#     except Exception as e:
#         print(e)
#         return

#     pipeline.start(config)

#     while not stop_event.is_set():
#         try:
#             frames: FrameSet = pipeline.wait_for_frames(100)
#             if frames is None:
#                 continue
#             color_frame = frames.get_color_frame()
#             if color_frame is None:
#                 continue
#             color_image = frame_to_bgr_image(color_frame)
#             if color_image is None:
#                 continue
#             timestamp = time.time()
#             color_queue.put((timestamp, color_image))
#         except Exception as e:
#             print(f"Error in color thread: {e}")
#             break

#     pipeline.stop()

# def record_ir():
#     config = Config()
#     pipeline = Pipeline()
#     try:
#         profile_list = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
#         try:
#             ir_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
#         except OBError as e:
#             print(e)
#             ir_profile = profile_list.get_default_video_stream_profile()
#         config.enable_stream(ir_profile)
#     except Exception as e:
#         print(e)
#         return

#     pipeline.start(config)

#     while not stop_event.is_set():
#         try:
#             frames = pipeline.wait_for_frames(100)
#             if frames is None:
#                 continue
#             ir_frame = frames.get_ir_frame()
#             if ir_frame is None:
#                 continue
#             ir_data = np.asanyarray(ir_frame.get_data())
#             width = ir_frame.get_width()
#             height = ir_frame.get_height()
#             ir_format = ir_frame.get_format()
#             if ir_format == OBFormat.Y8:
#                 ir_data = np.resize(ir_data, (height, width, 1))
#                 max_data = 255
#             elif ir_format == OBFormat.MJPG:
#                 ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
#                 if ir_data is None:
#                     continue
#                 ir_data = np.resize(ir_data, (height, width, 1))
#                 max_data = 255
#             else:
#                 ir_data = np.frombuffer(ir_data, dtype=np.uint16)
#                 ir_data = np.resize(ir_data, (height, width, 1))
#                 max_data = 65535

#             ir_image = preprocess_ir_data(ir_data, max_data)
#             ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
#             timestamp = time.time()
#             ir_queue.put((timestamp, ir_image))
#         except Exception as e:
#             print(f"Error in IR thread: {e}")
#             break

#     pipeline.stop()

# def sync_and_display_and_record(folder, datetime_str, counter):
#     color_buffer = []
#     ir_buffer = []

#     color_out = cv2.VideoWriter(f'.\\Recordings\\{folder}\\color_video_{datetime_str}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
#     ir_out = cv2.VideoWriter(f'.\\Recordings\\{folder}\\ir_video_{datetime_str}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 576))

#     start_time = time.time()

#     while not stop_event.is_set():
#         while not color_queue.empty():
#             color_buffer.append(color_queue.get())
#         while not ir_queue.empty():
#             ir_buffer.append(ir_queue.get())

#         while color_buffer and ir_buffer:

#             t_color, frame_color = color_buffer[0]      
#             t_ir, frame_ir = ir_buffer[0]
#             if abs(t_color - t_ir) < SYNC_TOLERANCE:
#                 counter+=1
#                 elapsed_time = counter/30
#                 time_text = f"Time: {elapsed_time:.2f}s"
                
#                 cv2.putText(frame_color, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#                 cv2.putText(frame_ir, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#                 cv2.imshow("Color Viewer", frame_color)
#                 cv2.imshow("Infrared Viewer", frame_ir)
#                 color_out.write(frame_color)
#                 ir_out.write(frame_ir)
#                 color_buffer.pop(0)
#                 ir_buffer.pop(0)
#             elif t_color < t_ir:
#                 color_buffer.pop(0)
#             else:
#                 ir_buffer.pop(0)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Exiting...", counter)
#             stop_event.set()
#             break

#     color_out.release()
#     ir_out.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     subject = input("Enter subject name: ")
#     if subject not in os.listdir(".\\Recordings"):
#         create_folder = input("Folder NOT FOUND! Do you want to create subject folder (Y/N):")
#         if create_folder.lower() == "y":
#             os.mkdir(f".\\Recordings\\{subject}")
#         else:
#             print("Exiting...")
#             exit()

#     counter = 0
#     t1 = threading.Thread(target=record_color)
#     t2 = threading.Thread(target=record_ir)
#     t3 = threading.Thread(target=sync_and_display_and_record, args=(subject, datetime_str,counter,))

#     t1.start()
#     t2.start()
#     t3.start()

#     t1.join()
#     t2.join()
#     t3.join()