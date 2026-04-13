from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_env import StepType, specs
from dm_control import suite, manipulation
from dm_control.suite.wrappers import pixels, action_scale

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_shape, dm_env):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._dm_env = dm_env

        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        
        pixel_spec = specs.BoundedArray(
            shape=np.concatenate([[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.float32,
            minimum=0.,
            maximum=1.,
            name='pixels'
        )

        if self._dm_env:
            self._obs_spec = pixel_spec
        else:
            self._obs_spec = {
                'pixels': pixel_spec,
                'state': self._env.observation_spec()['state']
            }

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        if self._dm_env:
            return time_step._replace(observation=obs)
        else:
            return time_step._replace(observation={'pixels': obs.astype(np.float32), 'state': time_step.observation['state'].astype(np.float32)})

    def _extract_pixels(self, time_step):
        if self._dm_env:
            pixels = (time_step.observation['pixels'] / 255.).astype(np.float32)
        else:
            pixels = time_step.observation['pixels']
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.copy() # [CHANNEL, NX, NY]

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class StateObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        time_step = self._env.reset()
        n_states = self._extract_states(time_step).size
        self._obs_spec = specs.BoundedArray(
            shape=(n_states,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name='observation'
        )

    def _extract_states(self, time_step):
        return np.concatenate([d[1] for d in time_step.observation.items()], axis=0).astype(np.float32)

    def reset(self):
        time_step = self._env.reset()
        states = self._extract_states(time_step)
        return time_step._replace(observation=states)

    def step(self, action):
        time_step = self._env.step(action)
        states = self._extract_states(time_step)
        return time_step._replace(observation=states)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make_dm_env(state_mode, domain, task, frame_stack, action_repeat, seed):
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    if (domain, task) in suite.ALL_TASKS:
        if state_mode:
            env = StateObservationWrapper(env)
        else:
            # add renderings for classical tasks
            # zoom in camera for quadruped
            camera_id = dict(quadruped=2).get(domain, 0)
            render_kwargs = dict(height=84, width=84, camera_id=camera_id)
            env = pixels.Wrapper(env,
                                 pixels_only=True,
                                 render_kwargs=render_kwargs, 
                                 observation_key='pixels')
            # stack several frames
            env = FrameStackWrapper(env, frame_stack, (84,84,3), True)            

    env = ExtendedTimeStepWrapper(env)
    return env

if __name__ == "__main__":

    import sys
    # sys.path.append("C:/dev/Laue_RL/src/drm")

    from laue_rl.drm.video import VideoRecorder
    from pathlib import Path

    env = make_dm_env(
        state_mode=False, 
        domain="cartpole", 
        task="balance", 
        frame_stack=2,
        action_repeat=2, 
        seed=0
    )

    print(env.observation_spec())

    video_recorder = VideoRecorder(
        root_dir=Path("C:/temporary/laue_rl_models/_testrun/eval_video"),
        fps=25,
        dm_env=True
    )

    n_frames = 1000

    timestep = env.reset()
    video_recorder.init(env, enabled=True)

    for _ in range(n_frames):
        timestep = env.step(action=0.1)
        video_recorder.record(env)
        if timestep.step_type == StepType.LAST:
            break

    video_recorder.save("test_video.mp4")
