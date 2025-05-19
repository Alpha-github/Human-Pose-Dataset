# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  Licensed under the Apache License, Version 2.0 (the "License");
#  You may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ******************************************************************************

import cv2
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image

ESC_KEY = 27

class LiveFeed:
    def __init__(self, width=640, height=0, format=OBFormat.RGB, fps=30):
        self.pipeline = Pipeline()
        self.config = Config()
        self.width = width
        self.height = height
        self.format = format
        self.fps = fps
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile = profile_list.get_video_stream_profile(self.width, self.height, self.format, self.fps)
            except OBError as e:
                print(f"Error: {e}")
                color_profile = profile_list.get_default_video_stream_profile()
                print("Using default color profile:", color_profile)
            self.config.enable_stream(color_profile)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
            raise

    def get_frame(self):
        try:
            frames: FrameSet = self.pipeline.wait_for_frames(100)
            if frames is None:
                return None
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None
            return frame_to_bgr_image(color_frame)
        except Exception as e:
            print(f"Error while capturing frame: {e}")
            return None

    def release(self):
        self.pipeline.stop()


# if __name__ == "__main__":
#     feed = LiveFeed()
#     try:
#         while True:
#             frame = feed.get_frame()
#             if frame is not None:
#                 cv2.imshow("Color Viewer", frame)
#             key = cv2.waitKey(1)
#             if key == ord('q') or key == ESC_KEY:
#                 break
#     except KeyboardInterrupt:
#         print("Interrupted.")
#     finally:
#         feed.release()
#         cv2.destroyAllWindows()
