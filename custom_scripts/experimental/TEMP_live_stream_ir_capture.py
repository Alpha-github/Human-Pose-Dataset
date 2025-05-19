# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http:# www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

import cv2
import numpy as np
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline

ESC_KEY = 27

class IRViewer:
    def __init__(self):
        self.config = Config()
        self.pipeline = Pipeline()
        self._setup_stream()

    def _setup_stream(self):
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
            try:
                ir_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
            except OBError as e:
                print(e)
                ir_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(ir_profile)
        except Exception as e:
            print(f"Error setting up IR stream: {e}")
            raise e

    def start(self):
        self.pipeline.start(self.config)

    def stop(self):
        self.pipeline.stop()

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                return None
            ir_frame = frames.get_ir_frame()
            if ir_frame is None:
                return None

            ir_data = np.asanyarray(ir_frame.get_data())
            width, height, ir_format = ir_frame.get_width(), ir_frame.get_height(), ir_frame.get_format()
            if ir_format == OBFormat.Y8:
                ir_data = np.resize(ir_data, (height, width, 1))
                data_type, image_dtype, max_data = np.uint8, cv2.CV_8UC1, 255
            elif ir_format == OBFormat.MJPG:
                ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
                if ir_data is None:
                    print("Failed to decode MJPEG data.")
                    return None
                ir_data = np.resize(ir_data, (height, width, 1))
                data_type, image_dtype, max_data = np.uint8, cv2.CV_8UC1, 255
            else:
                ir_data = np.frombuffer(ir_data, dtype=np.uint16)
                ir_data = np.resize(ir_data, (height, width, 1))
                data_type, image_dtype, max_data = np.uint16, cv2.CV_16UC1, 65535

            cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
            ir_data = ir_data.astype(data_type)
            ir_image = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)
            return ir_image

        except Exception as e:
            print(f"Error capturing IR frame: {e}")
            return None


# def run_ir_viewer():
#     viewer = IRViewer()
#     viewer.start()
#     print("Press 'q' or ESC to exit.")

#     while True:
#         ir_image = viewer.get_frame()
#         if ir_image is not None:
#             cv2.imshow("Infrared Viewer", ir_image)
#         key = cv2.waitKey(1)
#         if key == ord('q') or key == ESC_KEY:
#             break

#     viewer.stop()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     run_ir_viewer()
