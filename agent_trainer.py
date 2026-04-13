#!/usr/bin/env python3
import os
import json
import torch

import numpy as np

from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from pprint import pprint
from copy import deepcopy
from torchinfo import summary

from rl import utils, dmc
from rl.agents import drm, sac
from rl.logger import Logger
from rl.replay_buffer import ReplayBufferStorage, make_replay_loader
from rl.video import VideoRecorder

from dm_env import specs
from dm_control.suite.wrappers import action_scale

from laue_utils import sample_laue_parameters, get_timestamp, JSONEncoder, max_spots_sample_weights
from laue_simulation import map_structures
from laue_environment import LaueEnvMeca500

torch.backends.cudnn.benchmark = True

class AgentTrainer(ABC):
    """
    Base class of Agent trainer.
    """
    def __init__(self, algo, save_dir, config_file, **kwargs):
        """
        Class constructor.

        Parameters
        ----------
        algo : str
            Reinforcement learning algorithm ('drm' or 'sac')
        save_dir : str
            Save directory
        config_file : str
            Configuration file containing the run parameters
        kwargs : dict
            Dictionary with run parameters to be overwritten
        """
        # Setting up save directory and configuration
        self.save_dir = save_dir
        self.config_file = config_file

        # Read and copy config file
        with open(self.config_file, 'r') as fid:
            self.cfg = json.load(fid)

        self._update_cfg(**kwargs)

        # Check for potential save directory that already exists
        _dir_counter = 1
        _base, _current = os.path.split(self.save_dir)

        while True:
            try:
                os.makedirs(self.save_dir, exist_ok=False) # throws error if directory already exists
                break
            except FileExistsError:
                self.save_dir = os.path.join(_base, f"{_current}_{_dir_counter}") # update save directory
                _dir_counter += 1

        print(f"{get_timestamp()} - Using save directory '{self.save_dir}'.")

        _extended_cfg = deepcopy(self.cfg)
        _extended_cfg['base']['save_directory'] = self.save_dir
        _extended_cfg['base']['agent_algo'] = algo.lower()

        with open(os.path.join(self.save_dir, os.path.split(self.config_file)[1]), 'w') as fid:
            json.dump(_extended_cfg, fid, sort_keys=False, cls=JSONEncoder, indent=4)

        print(f"{get_timestamp()} - Using configuration parameters:")
        pprint(self.cfg, indent=4, width=160, sort_dicts=False)

        # Set class attributes regarding the environment
        self.reset_laue_pars_period = int(self.cfg['task']['reset_laue_pars_period'])

        # Set or sample Laue parameters
        self._laue_base_pars = self.cfg['task']['laue_parameters']
        self._laue_crystal_structure = self._laue_base_pars['crystal_structure']

        _cs, _hkl_targets = map_structures(
            crystal_structure=self._laue_base_pars['crystal_structure'],
            hkl_targets=self._laue_base_pars['hkl_targets']
        )

        # Update Laue parameters (replace str or list[str] with class/Enum instance)
        self._laue_base_pars['crystal_structure'] = _cs
        self._laue_base_pars['hkl_targets'] = _hkl_targets
        
        # Update Laue parameters with max. spots sample weights
        self._laue_base_pars['max_spots_weights'] = max_spots_sample_weights(
            min_value=min(self._laue_base_pars['max_spots_range']),
            max_value=max(self._laue_base_pars['max_spots_range']),
            weights_type=self._laue_base_pars['max_spots_weights']
        )

        for s in ['max_action_change', 'max_target_distance', 'max_target_rotation']:
            if not isinstance(self.cfg['task'][s], list):
                self.cfg['task'][s] = [self.cfg['task'][s]]

        # Setup curriculum learning values
        self._cl_episodes = int(self.cfg['task']['curriculum_learning_episodes'])
        self._cl_success_rate = float(self.cfg['task']['curriculum_learning_success_rate'])
        self._max_actions = int(self.cfg['task']['max_actions']) # NOTE constant for now

        # Convert to lists with appropriate data types
        convert_to_list = lambda x: [x] if not isinstance(x, list) else x
        convert_to_nested_list = lambda x: [x] if not any(isinstance(i, list) for i in x) else x

        # Number of actions, max. target distance and rotation
        self._max_action_changes = convert_to_list(self.cfg['task']['max_action_change'])
        self._max_target_distances = convert_to_list(self.cfg['task']['max_target_distance'])
        self._max_target_rotations = convert_to_list(self.cfg['task']['max_target_rotation'])

        # Populate angular ranges
        self._chi_ranges = {'action': [], 'initial': []}
        self._phi_ranges = {'action': [], 'initial': []}
        self._theta_ranges = {'action': [], 'initial': []}

        for full, ang_key in zip([self._chi_ranges, self._phi_ranges, self._theta_ranges], ['chi_range', 'phi_range', 'theta_range']):
            for dict_key in ['action', 'initial']:
                full[dict_key] = convert_to_nested_list(self.cfg['task'][ang_key][dict_key])

        # Set initial values (for curriculum learning)
        self._current_max_action_change = self._max_action_changes.pop(0)
        self._current_max_target_distance = self._max_target_distances.pop(0)
        self._current_max_target_rotation = self._max_target_rotations.pop(0)            
        self._current_chi_ranges = {'action': [], 'initial': []}
        self._current_phi_ranges = {'action': [], 'initial': []}
        self._current_theta_ranges = {'action': [], 'initial': []}
        _ = self._cl_sample_next_angular_ranges(update_env=False) # environments are not created yet

    @abstractmethod
    def _create_env(self):
        """
        Creation of environment.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def train(self):
        raise NotImplementedError()

    def _update_cfg(self, **kwargs):
        """
        Updates/overwrites configuration parameters.
        """
        for key, new_val in kwargs.items():
            for main_key in self.cfg.keys():
                for sub_key in self.cfg[main_key].keys():
                    if key == sub_key:
                        self.cfg[main_key][sub_key] = new_val # update value

    def _log_statement(self):
        """
        Logging statement for printing to console.

        Returns
        -------
        msg : str
            Logging statement
        """
        msg = f"max. action change: {self._current_max_action_change} | " + \
              f"max. target distance: {self._current_max_target_distance} | " + \
              f"max. target rotation: {self._current_max_target_rotation} | " + \
              f"chi range: {self._current_chi_ranges} | " + \
              f"phi range: {self._current_phi_ranges} | " + \
              f"theta range: {self._current_theta_ranges}"
        
        return msg
    
    def _get_snapshot_name(self):
        """
        Snapshot name based on the current (curriculum learning) parameters.

        Returns
        -------
        fname : str
            Snapshot name (without .pt ending)
        """
        if isinstance(self._current_max_action_change, list):
            mac = '_'.join([f'{int(x)}' for x in self._current_max_action_change])
        else:
            mac = int(self._current_max_action_change)

        mtd = int(self._current_max_target_distance)
        mtr = int(self._current_max_target_rotation)
        achi = 'to'.join([str(int(i)) for i in self._current_chi_ranges['action']])
        aphi = 'to'.join([str(int(i)) for i in self._current_phi_ranges['action']])
        atheta = 'to'.join([str(int(i)) for i in self._current_theta_ranges['action']])
        ichi = 'to'.join([str(int(i)) for i in self._current_chi_ranges['initial']])
        iphi = 'to'.join([str(int(i)) for i in self._current_phi_ranges['initial']])
        itheta = 'to'.join([str(int(i)) for i in self._current_theta_ranges['initial']])
        fname = f"snapshot_{self._laue_crystal_structure.lower()}_mac_{mac}_mtd_{mtd}_mtr_{mtr}_actions_{achi}_{aphi}_{atheta}_initial_{ichi}_{iphi}_{itheta}"

        return fname
    
    def _cl_sample_next_angular_ranges(self, update_env=False):
        """
        Samples the next iteration of angular ranges.

        Parameters
        ----------
        update_env : bool
            Whether to update the environment angular ranges

        Returns
        -------
        changed : bool
            Whether any angular range has been updated
        """
        c_list = [self._current_chi_ranges, self._current_phi_ranges, self._current_theta_ranges]
        f_list = [self._chi_ranges, self._phi_ranges, self._theta_ranges]
        n_list = ['chi', 'phi', 'theta']
        changed = False

        for current, full, name in zip(c_list, f_list, n_list):
            for dict_key in ['action', 'initial']:
                if len(full[dict_key]) > 0:
                    current[dict_key] = full[dict_key].pop(0)
                    changed = True
                if update_env:
                    self.train_env.set_angular_ranges(**{name: current})
                    self.eval_env.set_angular_ranges(**{name: current})
        
        return changed

class AgentTrainerOffPolicy(AgentTrainer):
    """
    Class for handling the training of off-policy model-free RL algorithms based on soft-actor-critic methods.
    """
    def __init__(self, algo, save_dir, config_file, **kwargs):
        """
        Class constructor.

        Parameters
        ----------
        algo : str
            Reinforcement learning algorithm ('drm' or 'sac')
        save_dir : str
            Save directory
        config_file : str
            Configuration file containing the run parameters
        kwargs : dict
            Dictionary with run parameters to be overwritten
        """
        agent_algo = algo.lower()

        # Check whether algorithm-choice is allowed for task
        if agent_algo not in ["drm", "sac"]:
            raise Exception(f"Only 'drm', 'sac' are allowed.")
        
        super().__init__(algo=agent_algo, save_dir=save_dir, config_file=config_file, **kwargs)

        # Set a few internal class attributes
        self._frame_stack = int(self.cfg['base']['frame_stack'])
        self._action_repeat = int(self.cfg['base']['action_repeat'])
        self._num_eval_episodes = int(self.cfg['base']['num_eval_episodes'])
        self._num_train_frames = int(self.cfg['base']['num_train_frames'])
        self._num_seed_frames = int(self.cfg['base']['num_seed_frames'])
        self._eval_every_frames = int(self.cfg['base']['eval_every_frames'])
        self._update_every_steps = int(self.cfg['base']['update_every_steps'])
        self._log_metrics_every_steps = int(self.cfg['base']['log_metrics_every_steps'])
        self.device = torch.device(self.cfg['base']['device'])
        utils.set_seed_everywhere(int(self.cfg['base']['seed']))

        # Create logger
        self.logger = Logger(log_dir=Path(self.save_dir), use_tb=True, use_wandb=False)
    
        # Create environments
        print(f"{get_timestamp()} - Creating environments ...")

        # Instantiate training and evaluation environments
        self.train_env : LaueEnvMeca500 = self._create_env()
        self.eval_env : LaueEnvMeca500 = self._create_env(os.path.join(self.save_dir, 'eval_video'))

        # Create replay buffer
        print(f"{get_timestamp()} - Initializing replay buffer (n_workers={self.cfg['base']['replay_buffer_num_workers']}) ...")
        data_specs = (
            self.train_env.observation_spec()['pixels'],
            self.train_env.observation_spec()['state'],
            self.train_env.action_spec(),
            specs.Array((1, ), np.float32, 'reward'),
            specs.Array((1, ), np.float32, 'discount')
        )

        replay_dir = Path(self.save_dir) / 'buffer'

        self.replay_storage = ReplayBufferStorage(data_specs=data_specs, replay_dir=replay_dir)
        self.replay_loader, self.buffer = make_replay_loader(
            replay_dir=replay_dir, 
            max_size=int(self.cfg['base']['replay_buffer_size']),
            batch_size=int(self.cfg['base']['batch_size']),
            num_workers=int(self.cfg['base']['replay_buffer_num_workers']), 
            save_snapshot=True,
            nstep=int(self.cfg['base']['nstep']),
            discount=float(self.cfg['base']['discount'])
        )
        self._replay_iter = None

        # Create video recorder (evaluation environment)
        self.video_recorder = VideoRecorder(root_dir=Path(self.save_dir), dm_env=False)

        # Create agent
        self.state_observation = bool(self.cfg['task']['state_observation'])

        if agent_algo == "drm":
            self.agent = drm.DrMAgent(
                obs_shape=self.train_env.observation_spec()['pixels'].shape,
                state_dim=self.train_env.observation_spec()['state'].shape[0] if self.state_observation else 0,
                action_dim=self.train_env.action_spec().shape[0],
                device=self.device,
                **self.cfg['agent']
            )
        elif agent_algo == "sac":
            self.agent = sac.SACPixelAgent(
                obs_shape=self.train_env.observation_spec()['pixels'].shape,
                action_dim=self.train_env.action_spec().shape[0],
                action_range=[-1.0, 1.0],
                device=self.device,
                **self.cfg['agent']
            )
        
        # Set final internal attributes
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_stddev_step = 0
        self._global_episode = 0
        self._global_success_rate = 0
        self._best_eval_reward = -np.inf

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_stddev_step(self):
        return self._global_stddev_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self._action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        """
        Evaluating the agent.
        """
        step, episode, total_reward, no_successes, no_out_of_range = 0, 0, 0, 0, 0
        eval_until_episode = utils.Until(self._num_eval_episodes)
        
        # Create save directory
        save_dir = os.path.join(self.video_recorder.save_dir, f"global_frame_{self.global_frame:07d}")
        os.makedirs(save_dir, exist_ok=True)

        while eval_until_episode(episode):
            # Sample simulation parameters
            _laue_pars = sample_laue_parameters(self._laue_base_pars)
            self.eval_env.set_laue_sim(_laue_pars)

            # Reset environment
            time_step = self.eval_env.reset()
            
            if episode == 0:
                self.eval_env.set_save_folder(save_dir) # update save directory
                with open(os.path.join(save_dir, 'laue_pars.json'), 'w') as fid:
                    json.dump(_laue_pars, fid, sort_keys=False, cls=JSONEncoder, indent=4) # save first set of Laue parameters

            self.video_recorder.init(self.eval_env, enabled=(episode == 0))

            # Episode loop
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, self.global_stddev_step, eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                if episode == 0:
                    self.eval_env.update_history(
                        dist=self.eval_env.currentdist, 
                        state=self.eval_env.state.copy(), 
                        reward=time_step.reward
                    )
                total_reward += time_step.reward
                step += 1

            no_successes += 1 if self.eval_env.done else 0
            no_out_of_range += 1 if self.eval_env.out_of_range else 0

            if episode == 0:
                self.eval_env.history_plot(self.eval_env.done)
                self.eval_env.reset_history()

            episode += 1

        # Calculate average metrics
        mean_reward = total_reward / episode
        mean_episode_length = step * self._action_repeat / episode
        success_rate = no_successes / episode
        out_of_range_rate = no_out_of_range / episode

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', mean_reward)
            log('episode_length', mean_episode_length)
            log('success_rate', success_rate)
            log('out_of_range_rate', out_of_range_rate)
            log('episode', self.global_episode)
            log('step', self.global_step)

        # try to save snapshot
        if mean_reward > self._best_eval_reward:
            print(f"{get_timestamp()} - New best mean evaluation reward ({mean_reward:.2f}). Saving snapshot ...")
            fname = f"{self._get_snapshot_name().replace('.', 'p')}"
            self.save_snapshot(os.path.join(self.save_dir, fname))
            self._best_eval_reward = mean_reward # update best mean evaluation reward

    def train(self):
        """
        Training the agent.
        """
        # predicates
        train_until_step = utils.Until(self._num_train_frames, self._action_repeat)
        seed_until_step = utils.Until(self._num_seed_frames, self._action_repeat)
        eval_every_step = utils.Every(self._eval_every_frames, self._action_repeat)

        episode_step, episode_reward = 0, 0
        successes = deque(maxlen=self._cl_episodes) # size-limited list used for curriculum learning
        log_successes = deque(maxlen=self._cl_episodes) # size-limited list used for logging the moving average success rate

        # Reset environment
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = None

        print(f"{get_timestamp()} - Starting training using {self.agent.__class__.__name__} ({self._log_statement()}) ...")
        print(f"{get_timestamp()} - Encoder structure:")

        try:
            summary(
                model=self.agent.encoder, 
                input_size=(1,*self.train_env.observation_spec()['pixels'].shape),
                col_names=['input_size', 'output_size', 'num_params']
            )
        except:
            print(f"Cannot print model summary for '{self.agent.encoder}'")

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1

                # check whether agent found high-symmetry point (training env!)
                success = 1 if self.train_env.done else 0
                successes.append(success)
                log_successes.append(success)
            
                # Check whether to increase task difficulty (avg. success rate >= 95% for N consecutive episodes)
                success_rate = np.mean(successes) if len(successes) == self._cl_episodes else 0 # wait till averaged long enough
                
                if success_rate >= self._cl_success_rate:

                    increased_difficulty = False 

                    # Check whether to increase task difficulty
                    if len(self._max_action_changes) > 0:
                        self._current_max_action_change = self._max_action_changes.pop(0)
                        self.train_env.set_max_action_change(self._current_max_action_change)
                        self.eval_env.set_max_action_change(self._current_max_action_change)
                        increased_difficulty = True
                    if len(self._max_target_distances) > 0:
                        self._current_max_target_distance = self._max_target_distances.pop(0)
                        self.train_env.set_max_target_distance(self._current_max_target_distance)
                        self.eval_env.set_max_target_distance(self._current_max_target_distance)
                        increased_difficulty = True
                    if len(self._max_target_rotations) > 0:
                        self._current_max_target_rotation = self._max_target_rotations.pop(0)
                        self.train_env.set_max_target_rotation(self._current_max_target_rotation)
                        self.eval_env.set_max_target_rotation(self._current_max_target_rotation)
                        increased_difficulty = True
                    
                    increased_difficulty = self._cl_sample_next_angular_ranges(update_env=True)

                    if increased_difficulty:
                        print(f"{get_timestamp()} - Success rate of {(success_rate*100):.1f} has been reached.")
                        print(f"{get_timestamp()} - Increasing task difficulty ({self._log_statement()}) ...")
                        successes = deque(maxlen=self._cl_episodes) # reset successes
                        self._best_eval_reward = -np.inf # reset best mean evaluation success rate

                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self._action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        log('success_rate', np.mean(log_successes))

                # Check whether to resample Laue parameters
                if self.reset_laue_pars_period != -1 and self.global_episode % self.reset_laue_pars_period == 0:
                    _laue_pars = sample_laue_parameters(self._laue_base_pars)
                    self.train_env.set_laue_sim(_laue_pars)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                # self.train_video_recorder.init(time_step.observation)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation, self.global_stddev_step, eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                ) if self.global_step % self._update_every_steps == 0 else dict()

                if self.global_step % self._log_metrics_every_steps == 0:
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            # self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            self._global_stddev_step += 1

    def save_snapshot(self, snapshot_fn):
        """
        Saving model weights.

        Parameters
        ----------
        snapshot_fn : str
            Snapshot name
        """
        keys_to_save = ['timer', '_global_step', '_global_episode']
        keys_to_save.extend([
            '_current_max_action_change', '_current_max_target_distance', '_current_max_target_rotation',
            '_current_chi_ranges', '_current_phi_ranges', '_current_theta_ranges'
        ])
        meta_kwargs = {k: self.__dict__[k] for k in keys_to_save}
        self.agent.save(snapshot_fn, **meta_kwargs)        

    def _update_cfg(self, **kwargs):
        """
        Updates/overwrites configuration parameters.
        """
        for key, new_val in kwargs.items():
            for main_key in self.cfg.keys():
                for sub_key in self.cfg[main_key].keys():
                    if key == sub_key:
                        self.cfg[main_key][sub_key] = new_val # update value
        
    def _create_env(self, video_save_dir=None):
        """
        Create environment.

        Parameters
        ----------
        video_save_dir : str or None
            Whether and where to save episode observations.

        Returns
        -------
        env : dm_env.Environment
            Environment instance
        """
        cryst_angles = self.cfg['task']['fixed_crystal_orientation_angles']
        obs_shape = tuple([int(x) for x in self.cfg['base']['obs_shape']])
        render_shape = tuple([int(x) for x in self.cfg['base']['render_shape']])
        
        # Sample the Laue parameters
        _laue_pars = sample_laue_parameters(self._laue_base_pars)

        env = LaueEnvMeca500(
            laue_pars=_laue_pars,
            obs_shape=obs_shape,
            render_shape=render_shape,
            max_actions=self._max_actions,
            max_action_change=self._current_max_action_change,
            max_target_distance=self._current_max_target_distance,
            max_target_rotation=self._current_max_target_rotation,
            chi_range=self._current_chi_ranges,
            phi_range=self._current_phi_ranges,
            theta_range=self._current_theta_ranges,
            hkl_targets=self._laue_base_pars['hkl_targets'],
            save_folder=video_save_dir,
            fixed_initial_crystal_angles=cryst_angles,
            fix_closest_hkl_target=self.cfg['task']['fix_closest_hkl_target'],
            reset_orientation_period=self.cfg['task']['reset_orientation_period'],
            sparse_rewards=self.cfg['task']['sparse_rewards'],
            coupled_actions=self.cfg['task']['coupled_actions'],
            domain_randomization=self.cfg['task']['domain_randomization']
        )

        env = dmc.ActionDTypeWrapper(env, np.float32)
        env = dmc.ActionRepeatWrapper(env, self._action_repeat)
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

        # stack several frames
        env = dmc.FrameStackWrapper(env, self._frame_stack, pixels_shape=(1,*obs_shape)[::-1], dm_env=False)
        env = dmc.ExtendedTimeStepWrapper(env)

        return env

if __name__ == "__main__":
    
    # Testing the agent trainer
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning) # TODO: fix the deprecation warning related to matplotlib (see github log)

    trainer = AgentTrainerOffPolicy(
        algo="drm",
        save_dir="C:/temporary/laue_rl_models/_testrun",
        config_file="C:/temporary/laue_rl_models/drm_config_test.json"
    )

    trainer.train()
