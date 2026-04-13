import os
import sys
import imageio.v2 as imageio

import numpy as np

from shutil import rmtree

class VideoRecorder:
    def __init__(self, root_dir, fps=1, dm_env=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.dm_env = dm_env
        self.fps = 20 if self.dm_env else fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if self.dm_env:
                if hasattr(env, 'physics'):
                    frame = env.physics.render(height=256,width=256, camera_id=0)
                else:
                    frame = env.get_pixels_with_width_height(256, 256)
                self.frames.append(frame)
            else:
                env.render()

    def save(self, save_name):
        if self.enabled:
            if self.dm_env:
                path = self.save_dir / save_name
                if "adroit" in str(self.save_dir):
                    self.frames = np.array(self.frames, dtype=np.uint8).transpose(0, 2, 3, 1)
                imageio.mimsave(str(path), self.frames, fps=self.fps)

# class TrainVideoRecorder:
#     def __init__(self, root_dir, render_size=256, fps=2):
#         if root_dir is not None:
#             self.save_dir = root_dir / 'train_video'
#             self.save_dir.mkdir(exist_ok=True)
#         else:
#             self.save_dir = None

#         self.render_size = render_size
#         self.fps = fps
#         self.frames = []

#     def init(self, obs, enabled=True):
#         self.frames = []
#         self.enabled = self.save_dir is not None and enabled
#         self.record(obs)

#     def record(self, obs):
#         if self.enabled:
#             frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
#                                dsize=(self.render_size, self.render_size),
#                                interpolation=cv2.INTER_CUBIC)
#             self.frames.append(frame)

#     def save(self, file_name):
#         if self.enabled:
#             path = self.save_dir / file_name
#             imageio.mimsave(str(path), self.frames, fps=self.fps)
