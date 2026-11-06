"""Render MuJoCo depth into a producer-owned float32 shared-memory block.

G1 retains the legacy Gymnasium renderer and cv2 resize path. X2 uses native
metric depth with clip-before-resize antialiased bicubic processing. Both publish
[N, 1, H, W] images with configurable frame delay and coordinated shutdown.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import ExitStack
from multiprocessing import shared_memory
from threading import Event

import cv2
import numpy as np
from loop_rate_limiters import RateLimiter


class MujocoCamera:
    def __init__(self, mj_model, mj_data, cam_id, img_shape, native_renderer=False):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.img_shape = img_shape
        self.cam_id = cam_id
        self.native_renderer = native_renderer
        if cam_id == -1:
            raise ValueError("Camera not found in the model.")
        # Created lazily on the rendering thread: with MUJOCO_GL=egl the GL context is bound to the
        # thread that creates it (creating it on the main thread gives EGL_BAD_ACCESS).
        self.cam_renderer = None

    def _ensure_renderer(self):
        if self.cam_renderer is None:
            if self.native_renderer:
                import mujoco

                self.cam_renderer = mujoco.Renderer(self.mj_model, height=self.img_shape[0], width=self.img_shape[1])
                self.cam_renderer.enable_depth_rendering()
                return
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

            self.cam_renderer = MujocoRenderer(
                self.mj_model, self.mj_data, camera_id=self.cam_id, width=self.img_shape[1], height=self.img_shape[0]
            )

    def get_frame(self):
        self._ensure_renderer()
        if self.native_renderer:
            self.cam_renderer.update_scene(self.mj_data, camera=self.cam_id)
            return self.cam_renderer.render().copy()
        return self._depth_2_meters(self.cam_renderer.render("depth_array"))

    def _depth_2_meters(self, depth):
        extend = self.mj_model.stat.extent
        near = self.mj_model.vis.map.znear * extend
        far = self.mj_model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))

    def release(self):
        if self.cam_renderer is not None:
            self.cam_renderer.close()


class ImageServer:
    def __init__(self, config, img_shm_name):
        print(f"[Image Server] Configuration: {config}")
        self.near_clip = config.get("near_clip", 0.1)
        self.far_clip = config.get("far_clip", 2.5)
        self.if_resize = config.get("if_resize", False)
        self.expected_shape = (
            config.get("resized_image_shape", [480, 640]) if self.if_resize else config.get("image_shape", [480, 640])
        )
        self.depth_delay = config.get("depth_delay", 0)
        self.clip_before_resize = config.get("clip_before_resize", False)
        self.stop_event = Event()
        self.error = None
        self.depth_buffer = deque(maxlen=self.depth_delay + 1)
        print(f"[Image Server] Depth delay: {self.depth_delay} frames")

        self.cameras = []
        self.camera_names = []
        image_shape = config.get("image_shape", [480, 640])
        fps = config.get("fps", 30)
        for camera_name, cam_id in config["id_numbers"].items():
            print(f"[Image Server] Initializing {camera_name}: shape={image_shape}, ids={cam_id}")
            self.cameras.append(
                MujocoCamera(
                    config["mj_model"],
                    config["mj_data"],
                    cam_id,
                    image_shape,
                    native_renderer=config.get("native_renderer", False),
                )
            )
            self.camera_names.append(camera_name)
        if not self.cameras:
            raise ValueError("No valid cameras found")

        expected_shape = [len(self.cameras), 1, self.expected_shape[0], self.expected_shape[1]]
        memory_size = int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        try:
            self.image_shm = shared_memory.SharedMemory(create=True, size=memory_size, name=img_shm_name)
            print(f"[Image Server] Created new shared memory: {img_shm_name}")
        except FileExistsError as error:
            raise FileExistsError(
                f"Depth producer block {img_shm_name!r} already exists; stop the existing simulator "
                "or choose another --shm-name."
            ) from error
        self.img_array = np.ndarray(expected_shape, dtype=np.float32, buffer=self.image_shm.buf)
        print(f"[Image Server] Shared memory: shape={expected_shape}, dtype=float32")

        self.rate = RateLimiter(fps)
        self.frame_count = 0
        self.frame_times: deque = deque()

    def _post_process_frame(self, frame):
        if getattr(self, "clip_before_resize", False):
            frame = np.clip(frame, self.near_clip, self.far_clip)
            if self.if_resize:
                from rpl_sim2sim.depth_processing import resize_bicubic_antialias

                frame = resize_bicubic_antialias(frame, self.expected_shape[0], self.expected_shape[1])
            # Match training: keep bicubic overshoot after the initial clipping.
            return (frame - self.near_clip) / (self.far_clip - self.near_clip) - 0.5
        if self.if_resize:
            frame = cv2.resize(frame, (self.expected_shape[1], self.expected_shape[0]), interpolation=cv2.INTER_CUBIC)
        frame = np.clip(frame, self.near_clip, self.far_clip)
        return (frame - self.near_clip) / (self.far_clip - self.near_clip) - 0.5

    def _close(self):
        # Run every cleanup even if one renderer fails to close. Callbacks run
        # in reverse order: release renderers, close memory, then unlink it.
        with ExitStack() as cleanup:
            cleanup.callback(self.image_shm.unlink)
            cleanup.callback(self.image_shm.close)
            for camera in self.cameras:
                cleanup.callback(camera.release)
        print("[Image Server] Closed.")

    def send_process(self):
        try:
            while not self.stop_event.is_set():
                frames = []
                for camera in self.cameras:
                    frame = camera.get_frame()
                    if frame is None:
                        raise RuntimeError(f"No frame from depth camera {camera.cam_id}")
                    frame = self._post_process_frame(frame)
                    frames.append(frame[None])  # [H, W] -> [C=1, H, W]
                full_image = np.stack(frames, axis=0)  # [N, C, H, W]

                if self.depth_delay > 0:
                    self.depth_buffer.append(full_image.copy())
                    if len(self.depth_buffer) > self.depth_delay:
                        full_image = self.depth_buffer[-(self.depth_delay + 1)]
                    else:
                        full_image = self.depth_buffer[0]

                np.copyto(self.img_array, full_image.astype(np.float32))

                current_time = time.time()
                self.frame_times.append(current_time)
                while self.frame_times and self.frame_times[0] < current_time - 1.0:
                    self.frame_times.popleft()
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    print(f"[Image Server] FPS: {len(self.frame_times):.2f}, Frames: {self.frame_count}")
                self.rate.sleep()
        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        except Exception as error:
            self.error = error
        finally:
            self.stop_event.set()
            try:
                self._close()
            except Exception as error:
                if self.error is None:
                    self.error = error
