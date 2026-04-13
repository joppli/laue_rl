#!/usr/bin/env python3
import os
import dm_env
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from skimage.draw import disk
from scipy.ndimage import gaussian_filter
from PIL import Image

from laue_utils import rot_matrix_x, rot_matrix_y, rot_matrix_z
from laue_simulation import LaueSimulator

class LaueEnvMeca500(dm_env.Environment):
    """"
    Custom class based on dm_env.Environment for single crystal alignment using reinforcement learning.
    Angle convention according to the Mecademic Meca500 robotarm (chi, phi, theta angles).
    
    Based on DeepMind RL enviroment API: https://github.com/google-deepmind/dm_env/tree/master
    """
    def __init__(
            self, 
            laue_pars, 
            obs_shape,
            render_shape,
            max_actions, 
            max_action_change, 
            max_target_distance,
            max_target_rotation,
            chi_range,
            phi_range,
            theta_range,
            hkl_targets,
            save_folder,
            domain_randomization,
            fixed_initial_crystal_angles=False,
            fix_closest_hkl_target=False,
            reset_orientation_period=1,
            sparse_rewards=False,
            coupled_actions=False
        ):
        """
        Class constructor.

        Parameters
        ----------
        laue_pars : dict
            (Constant and range) parameters for the Laue simulation
        obs_shape : tuple[int]
            Shape of the observations
        render_shape : tuple[int]
            Shape of the renders (for e.g. video creations)
        max_actions : int
            Number of maximum actions for the Laue agent
        max_action_change : list[float]
            Maximum angular changes step for the Laue agent (in degrees)
        max_target_distance : float
            Maximum reciprocal distance to symmetry target for pattern to be labeled as 'final state' (in degrees)
        max_target_rotation : float
            Maximum rotation of Laue pattern around symmetry target (in degrees) for pattern to be labeled as 'final state' 
            (if target_distance requirement is satisfied)
        chi_range : dict
            Dictionary with different ranges of the chi angle (in degrees):
                'action' (angular range of the robot arm)
                'initial' (angular range of the initial crystal orientation angles)
        phi_range : dict
            Dictionary with different ranges of the chi angle (in degrees) (see chi_range)
        theta_range : dict
            Dictionary with different ranges of the chi angle (in degrees) (see chi_range)
        hkl_targets : list[str]
            List of (hkl) symmetry targets
        save_folder : str
            Used for saving the renders
        domain_randomization : dict
            Domain randomization parameters 
            (keys: 'gauss_sigma_range', 'remove_fraction', 'distortion_fraction', 'spot_shift_sigma')
        fixed_initial_crystal_angles : list[float]
            Rotation angles (chi->phi->theta) describing the initial crystal orientation (in degrees)
        fix_closest_hkl_target : bool
            Whether the agent should target the initially closest high-symmetry point
        reset_orientation_period : int
            After how many episodes the crystal orientation (crystal + beam axis) should be randomly reset
            (ignored if the crystal- and/or beam axis are fixed)
        sparse_rewards : bool
            Whether to use only sparse rewards (i.e. found high-symmetry point), penalty of -1 per step
        coupled_actions : bool
            Whether actions are coupled, i.e. later actions depend on previous action ('true' goniometer setup)
        """
        super().__init__()

        # Iinitialization of simulation
        self.laue_pars = laue_pars

        # Check whether space group is allowed for given crystal structure
        for sg in self.laue_pars['space_groups']:
            if sg not in self.laue_pars['crystal_structure'].space_groups:
                raise Exception(f"Space group not allowed for '{self.laue_pars['crystal_structure'].__class__.__name__}'. Allowed are {self.laue_pars['crystal_structure'].space_groups}.")

        self.sim = LaueSimulator(
            space_groups=self.laue_pars['space_groups'],
            space_groups_weights=self.laue_pars['space_groups_weights'],
            lat_constants=self.laue_pars['lat_constants'],
            d_film=self.laue_pars['d_film'],
            y_lim=self.laue_pars['y_lim'],
            z_lim=self.laue_pars['z_lim'],
            beam_yz=self.laue_pars['beam_yz'],
            energy_beam=self.laue_pars['energy_beam'],
            max_plane_index=self.laue_pars['max_planes']
        )

        # Environment parameters
        self.obs_shape = tuple(obs_shape)
        self.render_shape = tuple(render_shape)
        self.step_nr = 0 # number of environment steps
        self.episode_nr = 0 # number of episodes
        self.chi_rotation = False if np.all(np.array(chi_range['action']) == 0) else True # whether to enable chi rotation
        self.done = self.out_of_range = self.out_of_actions = False
        self.found_center = False
        self.intrinsic_render_shape = (1284,1284) # Laue patterns are rendered at this resolution first and then downsampled
        self.coupled_actions = bool(coupled_actions)

        # Domain randomization
        self.domain_rnd_dict = {}
        self.domain_rnd_dict['gauss_sigma_range'] = np.array(domain_randomization['gauss_sigma_range'], dtype=np.float32)
        self.domain_rnd_dict['remove_fraction'] = float(domain_randomization['remove_fraction'])
        self.domain_rnd_dict['spot_shift_sigma'] = float(domain_randomization['spot_shift_sigma'])
        self.domain_rnd_dict['max_fraction_random_spots'] = float(domain_randomization['max_fraction_random_spots'])
        self.domain_rnd_dict['constant_scale_range'] = np.array(domain_randomization['constant_scale_range'], dtype=np.float32)
        
        self.normalize_pixels = lambda x: x * self.domain_rnd_dict['constant_scale']

        # Define the (fixed) beam axis
        self.beam_axis_ref = np.array([1.0, 0.0, 0.0]) # reference beam axis

        # The initial state / goniometer angles should be zero while the initial crystal axis orientation is randomly set.
        self.state = np.array([0., 0., 0.]) if self.chi_rotation else np.array([0., 0.])

        # Environment parameters via setter methods
        self.set_max_actions(max_actions)
        self.set_max_action_change(max_action_change)
        self.set_max_target_distance(max_target_distance, calibrate_radius=False)
        self.set_max_target_rotation(max_target_rotation)
        self.chi_range = {'action': [], 'initial': []}
        self.phi_range = {'action': [], 'initial': []}
        self.theta_range = {'action': [], 'initial': []}
        self.set_angular_ranges(chi_range, phi_range, theta_range)

        # Define the action space to be in the range [-1,1] for stability
        if self.chi_rotation:
            self.action_space = dm_env.specs.BoundedArray(minimum=-1, maximum=1, shape=(3,), dtype=np.float32, name='action')
            self._rotation_matrix = self._rotation_matrix_chi
        else:
            self.action_space = dm_env.specs.BoundedArray(minimum=-1, maximum=1, shape=(2,), dtype=np.float32, name='action')
            self._rotation_matrix = self._rotation_matrix_no_chi

        # Extract symmetry targets
        self.hkl_xax = None # cartesian (h,k,l) values of the current state (plane normal to lattice vectors)
        self.hkl_target = None # cartesian (h,k,l) target value
        self.hkl_possible_targets = list(hkl_targets)
        
        if len(self.hkl_possible_targets) == 0:
            raise Exception(f"At least one (h,k,l) target is required.")
        
        self._hkl_targets_dict = {}
        self._cart_targets_dict = {}
        allowed_hkl_targets = self.laue_pars['crystal_structure'].symmetries.keys()
        for t in self.hkl_possible_targets:
            if t.upper() not in allowed_hkl_targets:
                raise Exception(f"Unkown or forbidden symmetry target '{t}' provided for {self.laue_pars['crystal_structure']}. Allowed are {allowed_hkl_targets}.")
            # Define actual (h,k,l) values
            _hkl = self.laue_pars['crystal_structure'].symmetries[t]['hkl'] # get array values
            _hkl /= np.linalg.norm(_hkl, axis=1, keepdims=True) # normalize vectors
            # Calculate cartesian (h,k,l) values (identical to actual (h,k,l) values if lattice angles are perpendicular)
            cart = (self.sim.basis_star @ _hkl.T).T # apply reciprocal basis transformation to get cartesian coordinates
            cart = np.where(np.abs(cart) < 1e-6, 0, cart) # set small values to zero
            cart /= np.linalg.norm(cart, axis=1, keepdims=True)
            self._hkl_targets_dict[t] = _hkl
            self._cart_targets_dict[t] = cart

        # Define the possible reference axes to calculate the angle of rotation
        self._rot_ref_axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        if fixed_initial_crystal_angles is not False:
            assert np.size(fixed_initial_crystal_angles) == 3
            self.fixed_initial_crystal_angles = np.deg2rad(fixed_initial_crystal_angles)
        else:
            self.fixed_initial_crystal_angles = None
            self.initial_crystal_angles = None

        # Calibrate the reciprocal radius for rendering
        self._calibrate_reciprocal_radius(self.max_target_distance)

        # Set remaining timers and flags
        self.reset_orientation_period = int(reset_orientation_period)
        self.reset_next_step = True

        # Define the observation space (normalized images and state (might not be used))
        self.observation_space = {
            'pixels': dm_env.specs.BoundedArray(
                shape=(1,*self.obs_shape), minimum=0., maximum=1., dtype=np.float32, name='pixels'
            ),
            'state': dm_env.specs.BoundedArray(
                shape=(3 if self.chi_rotation else 2,), minimum=-1., maximum=1., dtype=np.float32, name='state'
            )
        }

        # Initialization of remaining attributes
        self.save_folder = save_folder
        self.fix_closest_hkl_target = bool(fix_closest_hkl_target)
        self.sparse_rewards = bool(sparse_rewards)

        # Define final and penalty rewards
        self.reached_target_reward = 100.0 # only once per episode
        self.step_penalty = -1.0
        self.out_of_range_penalty = 0.0

        # Initialize containers for history metrics
        self.reset_history()

    def action_spec(self):
        """
        Action specs that should be provided to 'step'.

        Returns
        -------
        spec[BoundedArray]
        """
        return self.action_space
    
    def observation_spec(self):
        """
        Observation specs provided by the environment.

        Returns
        -------
        spec[BoundedArray]
        """
        return self.observation_space
    
    def set_save_folder(self, save_folder):
        """
        Setter method for setting the save directory for renders.

        Parameters
        ----------
        save_folder : str
            Save directory
        """
        self.save_folder = save_folder

    def set_max_actions(self, max_actions):
        """
        Setter method for maximum number of actions.

        Parameters
        ----------
        max_actions : int
            Maximum number of actions
        """
        self.max_actions = int(max_actions)

    def set_max_action_change(self, max_action_change):
        """
        Setter method for maximum action change.

        Parameters
        ----------
        max_action_change : list[float]
            Maximum angular change(s) in degrees
        """
        self.max_angles_change = np.deg2rad(max_action_change)

    def set_max_target_distance(self, max_target_distance, calibrate_radius=True):
        """
        Setter method for maximum target distance.

        Parameters
        ----------
        max_target_distance : float
            Maximum target distance (in degrees)
        """
        self.max_target_distance = np.deg2rad(max_target_distance)
        if calibrate_radius:
            self._calibrate_reciprocal_radius(self.max_target_distance) # re-calibrate the reciprocal radius for rendering

    def set_max_target_rotation(self, max_target_rotation):
        """
        Setter method for maximum target rotation (phi angle).

        Parameters
        ----------
        max_target_rotation : float
            Maximum target rotation (in degrees)
        """
        self.max_target_rotation = float(max_target_rotation)

    def set_angular_ranges(self, chi=None, phi=None, theta=None):
        """
        Setter method for angular ranges (chi,phi,theta). Used to check for 'out-of-boundaries'.

        Parameters
        ----------
        chi : dict
            Angular chi ranges (in degrees)
        phi : dict
            Angular phi ranges (in degrees)
        theta : dict
            Angular theta ranges (in degrees)
        """
        for d, ang in zip([self.chi_range, self.phi_range, self.theta_range], [chi, phi, theta]):
            for key in ['action', 'initial']:
                if ang is not None:
                    d[key] = np.deg2rad(ang[key])

        if self.chi_rotation:
            self.angular_action_ranges = np.array([
                self.chi_range['action'], 
                self.phi_range['action'], 
                self.theta_range['action']
            ])
        else:
            self.angular_action_ranges = np.array([
                self.phi_range['action'], 
                self.theta_range['action']
            ])

    def set_laue_sim(self, laue_pars):
        """
        Setter method for Laue simulation instance.

        Parameters
        ----------
        laue_pars : dict
            Dictionary with Laue parameters
        """
        self.laue_pars = laue_pars
        self.sim = LaueSimulator(
            space_groups=self.laue_pars['space_groups'],
            space_groups_weights=self.laue_pars['space_groups_weights'],
            lat_constants=self.laue_pars['lat_constants'],
            d_film=self.laue_pars['d_film'],
            y_lim=self.laue_pars['y_lim'],
            z_lim=self.laue_pars['z_lim'],
            beam_yz=self.laue_pars['beam_yz'],
            energy_beam=self.laue_pars['energy_beam'],
            max_plane_index=self.laue_pars['max_planes']
        )
        self._calibrate_reciprocal_radius(self.max_target_distance)

    def set_reset_next_step(self, reset):
        """
        Whether to reset the environment with the next step.

        Parameters
        ----------
        reset : bool
        """
        self.reset_next_step = reset

    def distance_reward(self, dist):
        """
        Reward based on reciprocal distance change.

        Parameters
        ----------
        dist : float
            New reciprocal distance (to be compared with previous distance)

        Returns
        -------
        float
            Distance reward            
        """
        return (self.currentdist - dist) / (self.startdist * np.sqrt(self.step_nr)) * self.reached_target_reward

    def step(self, action):
        """
        Simulates new Laue pattern, detects symmetry and calculates reward by a given action.

        Parameters
        ----------
        action : space[BoundedArray]
            Chosen action by randomness or policy to change state and thus the observation

        Returns
        -------
        TimeStep object containing:
            - TimeStepType
            - reward
            - discount
            - observation
        """
        if self.reset_next_step:
            return self.reset()
        
        # Update the number of environment steps
        self.step_nr += 1

        # Rescale the action from the [-1,1] range to the range defined by max_angles_change
        action = self.max_angles_change * np.array(action)

        # Apply goniometer rotation to the angles (conversion of angles to crystal- and z-axis)
        self._angles_to_hkl(action)

        # Update state
        self.state += action

        # Apply goniometer rotation and run pattern simulation
        obs = {
            'pixels': (self._simulate_observation(shape=self.obs_shape)[np.newaxis,...]).astype(np.float32), # with initial channel dimension
            'state': (self.state / self.angular_action_ranges[:,1]).astype(np.float32) # normalize to max. angular range
        }

        # Define the reward and check conditions
        if not self.fix_closest_hkl_target:
            # Re-calculate the closest symmetry target based on the current orientation
            self._update_closest_target()

        # Calculate new reciprocal distance and rotation
        new_dist = self._get_distance()
        new_rot = self._get_rotation()

        # Check for out-of-range condition (end episode with no additional reward)
        if np.any((self.state < self.angular_action_ranges[:,0]) | (self.state > self.angular_action_ranges[:,1])):
            self.episode_nr += 1
            self.out_of_range = True
            self.reset_next_step = True
            return dm_env.TimeStep(step_type=dm_env.StepType.LAST, reward=self.out_of_range_penalty, discount=0.0, observation=obs)

        # Intermediate reward calculation
        if self.sparse_rewards:
            reward = self.step_penalty
        else:
            # Calculate distance reward: positive/negative reward if agents gets closer/further away 
            # (reduced reward for higher number of steps taken and normalized by start distance)
            reward = self.distance_reward(new_dist)

        if new_dist <= self.max_target_distance:
            # Close to high-symmetry point (potentially remaining chi offset)
            if not self.found_center:
                reward += self.reached_target_reward if not self.chi_rotation else 0.0 # reward for correct phi + theta angles
                self.found_center = True

            if new_rot <= self.max_target_rotation:
                # Fully aligned high-symmetry point
                reward += self.reached_target_reward if self.chi_rotation else 0.0 # reward for correct chi angle
                self.done = True

        # Update distance and rotation variables
        self.currentdist = new_dist
        self.currentrot = new_rot

        if self.step_nr >= self.max_actions: # used up all actions (end episode)
            self.out_of_actions = True
        
        if self.done or self.out_of_actions:
            self.episode_nr += 1
            self.reset_next_step = True
            return dm_env.TimeStep(step_type=dm_env.StepType.LAST, reward=reward, discount=0.0, observation=obs)
        else:
            return dm_env.TimeStep(step_type=dm_env.StepType.MID, reward=reward, discount=1.0, observation=obs)

    def reset(self):
        """
        Resets attributes to initial conditions.

        Returns
        -------
        TimeStep object containing:
            - TimeStepType
            - reward
            - discount
            - observation
        """
        # Potentially set random initial crystal- and beam orientation       
        zero_state = np.array([0., 0., 0.]) if self.chi_rotation else np.array([0., 0.])

        # Sample domain randomization parameters
        self.domain_rnd_dict['gauss_sigma'] = np.random.uniform(*self.domain_rnd_dict['gauss_sigma_range'])
        self.domain_rnd_dict['constant_scale'] = np.random.uniform(*self.domain_rnd_dict['constant_scale_range'])

        if self.episode_nr % self.reset_orientation_period == 0:

            while True:
                self._update_crystal_orientation(self.fixed_initial_crystal_angles) # set initial crystal orientation
                self._angles_to_hkl(zero_state.copy()) # set initial goniometer orientation

                # Identify the closest high-symmetry target
                self._update_closest_target()
                new_dist = self._get_distance()
                new_rot = self._get_rotation()

                if new_dist > self.max_target_distance: # make sure that agent isn't placed on target
                    self.startdist = new_dist + 1e-8 # add small epsilon to avoid division by zero
                    self.startrot = new_rot + 1e-8
                    self.currentdist = self.startdist
                    self.currentrot = self.startrot
                    break

        # Reset flags and counters
        self.done = self.out_of_range = self.out_of_actions = False
        self.found_center = False
        self.reset_next_step = False
        self.step_nr = self.step_nr_chi = 0
        self.state = zero_state.copy()

        # Simulate initial observation
        obs = {
            'pixels': (self._simulate_observation(shape=self.obs_shape)[np.newaxis,...]).astype(np.float32), # with initial channel dimension
            'state': (self.state / self.angular_action_ranges[:,1]).astype(np.float32)
        }
        
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=None, discount=None, observation=obs)

    def render(self, domain_rnd=False):
        """
        Saves the current observation (state) to a .png file.

        Parameters
        ----------
        domain_rnd : bool
            Whether to enable or disable domain randomization when rendering
        """
        os.makedirs(self.save_folder, exist_ok=True)

        # Render observation
        obs = self._simulate_observation(shape=self.render_shape, domain_rnd=domain_rnd)

        fig = plt.figure(figsize=(5.12,5.12)) # 5.12 to have 512 pixels in total
        ax = fig.add_subplot(1,1,1)
        ax.imshow(obs, vmin=0.0, vmax=1.0, cmap=plt.cm.gray)

        line_ofs = int(0.1*max(self.render_shape))

        ax.vlines(
            x=[self.render_shape[1]/2,self.render_shape[1]/2], 
            ymin=line_ofs, ymax=self.render_shape[0]-line_ofs, 
            colors='red', linewidth=1, alpha=0.25
        )
        ax.hlines(
            y=[self.render_shape[0]/2,self.render_shape[0]/2], 
            xmin=line_ofs, xmax=self.render_shape[1]-line_ofs, 
            colors='red', linewidth=1, alpha=0.25
        )

        circle = plt.Circle(
            xy=(self.render_shape[0]/2,self.render_shape[1]/2), 
            radius=self._proj_target_radius, 
            color='red', fill=False, linewidth=2, alpha=0.25)
        
        ax.add_patch(circle)

        if self.out_of_actions and not self.done:
            ax.text(0.5, 0.925, "OUT OF ACTIONS", transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='red')
        
        if self.done:
            ax.text(0.5, 0.925, "SUCCESS", transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='yellow')
        elif self.out_of_range:
            ax.text(0.5, 0.925, "OUT OF RANGE", transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='red')
        
        title_step = f"Step: {self.step_nr:03d}"
        title_angles = f"Angles: ({', '.join(f'{d:.1f}' for d in np.rad2deg(self.state))}) deg"
        title_state = f"Reciprocal coord.: ({', '.join(f'{d:.2f}' for d in self.hkl_xax)}) r.l.u." # NOTE: these are the "cartesian" (h,k,l) values
        title_stats = f"Distance (rotation): {np.rad2deg(self.currentdist):.1f} ({self.currentrot:.1f}) deg"

        ax.set(
            xticks=[], yticks=[], xlim=[0,self.render_shape[1]-1], ylim=[self.render_shape[0]-1,0],
            title=f"{title_step}\n{title_angles}\n{title_state}\n{title_stats}"
        )

        fig.savefig(os.path.join(self.save_folder, f"laue_step_{self.step_nr:03d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def reset_history(self):
        """
        Resets the history metrics.
        """
        self._history_distances = []
        self._history_states = []
        self._history_rewards = []

    def update_history(self, dist, state, reward):
        """
        Updates the current history metrics.

        Parameters
        ----------
        dist : float
            Reciprocal distance to target
        state : 1D-array[float]
            State angles
        reward : float
            Step reward
        """
        self._history_distances.append(dist)
        self._history_states.append(state)
        self._history_rewards.append(np.nan if reward is None else reward)

    def history_plot(self, success, show=False):
        """
        Creates a history plot containing the reciprocal distance, the states, and the rewards.

        Parameters
        ----------
        success : bool
            Whether the goal has been reached
        show : bool
            Whether to show the plot
        """
        if len(self._history_distances) == 0 or len(self._history_states) == 0 or len(self._history_rewards) == 0:
            raise Exception(f"Cannot create history plot without a history. Call update_history() at least once to populate history.")
        
        os.makedirs(self.save_folder, exist_ok=True)

        n_steps = len(self._history_distances)
        step_numbers = np.arange(n_steps)
        distances = np.rad2deg(np.array(self._history_distances, dtype=np.float32))
        states = np.rad2deg(np.array(self._history_states, dtype=np.float32))
        rewards = np.array(self._history_rewards, dtype=np.float32)
        cummulative_rewards = np.nancumsum(rewards)

        out_of_range = rewards[-1] == 0.0
        if success or out_of_range:
            rewards[-1] = np.nan
            cummulative_rewards[-1] = np.nan

        if success:
            status = "SUCCESS"
        elif out_of_range:
            status = "OUT OF ANGULAR RANGE"
        else:
            status = "OUT OF ACTIONS"

        xticks_locator = MaxNLocator(integer=True)

        plt.figure(figsize=(10,6))
        plt.suptitle(f"Evaluation with status [{status}]")
        plt.subplot(2,2,1)
        plt.plot(step_numbers, distances)
        plt.axhline(y=np.rad2deg(self.max_target_distance), xmin=0.0, xmax=n_steps-1, color='red', linestyle='--', alpha=0.75)
        plt.xlabel(f"Episode step")
        plt.ylabel(f"Distance to target (deg)")
        plt.gca().xaxis.set_major_locator(xticks_locator)
        plt.subplot(2,2,2)
        for j, action_label in enumerate(['chi', 'phi', 'theta'] if self.chi_rotation else ['phi', 'theta']):
            plt.plot(step_numbers, states[:,j], label=action_label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=3, frameon=False)
        plt.xlabel(f"Episode step")
        plt.ylabel(f"State (deg)")
        plt.gca().xaxis.set_major_locator(xticks_locator)
        plt.subplot(2,2,3)
        plt.plot(step_numbers, rewards)
        if success or out_of_range:
            color = 'green' if success else 'red'
            mk = 'o' if success else 'X'
            try:
                plt.scatter(step_numbers[-1], rewards[-2], s=40, c=color, marker=mk)
            except: # probably an IndexError has arised
                pass
        plt.xlabel(f"Episode step")
        plt.ylabel(f"Reward")
        plt.gca().xaxis.set_major_locator(xticks_locator)
        plt.subplot(2,2,4)
        plt.plot(step_numbers, cummulative_rewards)
        if success or out_of_range:
            color = 'green' if success else 'red'
            mk = 'o' if success else 'X'
            try:
                plt.scatter(step_numbers[-1], cummulative_rewards[-2], s=40, c=color, marker=mk)
            except: # probably an IndexError has arised
                pass
        plt.xlabel(f"Episode step")
        plt.ylabel(f"Cummulative reward")
        plt.gca().xaxis.set_major_locator(xticks_locator)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, "history.png"), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def _rotation_matrix_chi(self, angles):
        """
        Generates a rotation matrix for the Mecademic Meca500 robot arm convention.

        Parameters
        ----------
        angles : 1D-array[float]
            (chi,phi,theta) angles (in radians)
            1) chi (rotates the pattern around the x-axis), i.e. pattern rotation
            2) phi (rotates pattern around the new z-axis), i.e. "up-down" movement
            3) theta (rotates the pattern around the new y-axis), i.e. "left-right" movement

        Returns
        -------
        2D-array[float]
            3x3 rotation matrix
        """
        chi, phi, theta = angles
        return rot_matrix_y(theta) @ rot_matrix_z(phi) @ rot_matrix_x(chi)
    
    def _rotation_matrix_no_chi(self, angles):
        """
        Generates a rotation matrix for the Mecademic Meca500 robot arm convention (without chi rotation).

        Parameters
        ----------
        angles : 1D-array[float]
            (phi,theta) angles (in radians)
            1) phi (rotates pattern around the z-axis), i.e. "up-down" movement
            2) theta (rotates the pattern around the new y-axis), i.e. "left-right" movement

        Returns
        -------
        2D-array[float]
            3x3 rotation matrix
        """
        phi, theta = angles
        return rot_matrix_y(theta) @ rot_matrix_z(phi) 

    def _angles_to_hkl(self, angles):
        """
        Transformation of angles to (h,k,l) values. Runs the Laue spot simulation.
        
        Parameters
        ----------
        angles : 1D-array[float]
            (chi,phi,theta) angles (in radians)
        """
        # Generate full rotation matrix
        R = self.Rt @ self._rotation_matrix(angles) @ self.Rt.T

        # Update crystal and beam axis
        self.crystal_axis = R @ self.crystal_axis
        self.beam_axis = R @ self.beam_axis

        # Set crystal- and z-axis vectors (use of cartesian (orthogonal) vectors)
        self.sim.set_crystal_axis(self.crystal_axis, from_hkl=False)
        self.sim.set_z_axis(self.beam_axis, from_hkl=False)
        self.sim.run_simulation()

        # Update reciprocal position
        self.hkl_xax = self.sim.x_axis.copy()

        if not self.coupled_actions:
            # Update new orthonormal crystal orientation (sets center of pattern rotation to center of detector)
            self.Rt = self._gram_schmidt(self.crystal_axis, self.beam_axis)

    def _update_crystal_orientation(self, initial_angles=None):
        """
        Transforms reference coordinate frame to a new crystal orientation.
        Additionally, incorporates initial orientation angles.

        Parameters
        ----------
        initial_angles : list[float] or None
            Initial crystal orientation angles (in radians)
        """
        if initial_angles is None: # crystal can have any orientation
            self.initial_crystal_angles = np.random.uniform(
                low=[self.chi_range['initial'][0], self.phi_range['initial'][0], self.theta_range['initial'][0]],
                high=[self.chi_range['initial'][1], self.phi_range['initial'][1], self.theta_range['initial'][1]]
            )
        else:
            self.initial_crystal_angles = np.array(initial_angles, dtype=np.float32)

        # Sample a new "perfect" crystal orientation
        hkl_targets = self._hkl_targets_dict[np.random.choice(self.hkl_possible_targets)]

        # Sample the initial u and b vectors (as actual (h,k,l) values)
        self.u_hkl = hkl_targets[np.random.randint(0, hkl_targets.shape[0]),:].copy() # make sure to copy to avoid in-place modification (bad)
        self.b_hkl = self.beam_axis_ref.copy() # the reference beam axis is always along (1,0,0)

        # If beam axis is parallel to (h,k,l) plane normal, change the beam axis
        if np.all(np.cross(self.u_hkl, self.b_hkl) == np.zeros((3,))):
            self.b_hkl = np.roll(self.b_hkl, 1)

        # Map (h,k,l) to Cartesian via reciprocal space basis
        u_cart = self.sim.basis_star @ self.u_hkl
        u_cart /= np.linalg.norm(u_cart)  

        # Map (h,k,l) of beam axis via reciprocal space basis
        b_cart = self.sim.basis_star @ self.b_hkl
        b_cart /= np.linalg.norm(b_cart)

        # Perform first Gram-Schmidt orthonormalization
        self.Rt = self._gram_schmidt(u_cart, b_cart)

        # Create the initial crystal orientation matrix
        R_crystal = self._rotation_matrix_chi(self.initial_crystal_angles)

        # Create initial crystal and beam axis vectors
        crystal_ax = self.Rt @ R_crystal @ self.Rt.T @ self.Rt[:,0]
        beam_ax = self.Rt @ R_crystal @ self.Rt.T @ self.Rt[:,1]

        # Perform second Gram-Schmidt orthonormalization (centered on initial crystal orientation vectors)
        self.Rt = self._gram_schmidt(crystal_ax, beam_ax)

        # Construct orthonormal crystal and beam axis vectors
        self.crystal_axis = self.Rt[:,0]
        self.beam_axis = self.Rt[:,1]

    def _simulate_observation(self, shape, domain_rnd=True):
        """
        Simulates a new (normalized) Laue diffraction pattern given the current Laue instance.

        Parameters
        ----------
        shape : tuple[int]
            Shape of discretized 2D-array
        domain_rnd : bool
            Whether to enable domain randomization or not

        Returns
        -------
        observation : 2D-array[float]
            Laue diffraction pattern
        """
        # The pattern is first rendered at a high resolution (e.g. 1024 x 1024 pixels)
        observation = self.sim.generate_pattern(
            shape=self.intrinsic_render_shape, 
            max_spots=self.laue_pars['max_spots'],
            remove_fraction=self.domain_rnd_dict['remove_fraction'] if domain_rnd else 0.0,
            spot_shift_sigma=self.domain_rnd_dict['spot_shift_sigma'] if domain_rnd else 0.0,
            spot_radius=10.0
        )

        # Add random spots
        if domain_rnd:
            n_random_spots = np.random.uniform(
                low=0.0, high=self.domain_rnd_dict['max_fraction_random_spots']*self.laue_pars['max_spots']
            )
            random_spot_centers = np.unravel_index(
                np.random.choice(observation.size, size=(int(n_random_spots),), replace=False), shape=self.intrinsic_render_shape
            )
            for center in np.transpose(random_spot_centers):
                r, c = disk(center, radius=10.0, shape=self.intrinsic_render_shape)
                observation[r, c] = 1.0

        # The pattern is then downsampled to be used by the agent encoder
        observation = np.array(Image.fromarray(observation).resize(size=shape, resample=Image.Resampling.BOX))

        # And then convoluted with a Gaussian
        observation = gaussian_filter(observation, sigma=self.domain_rnd_dict['gauss_sigma'])

        # Potentially remove camera center spot
        if self.laue_pars['r_camera'] > 0:
            r_cut, c_cut = disk((shape[0]//2, shape[1]//2), radius=int(self.laue_pars['r_camera']*min(shape)), shape=shape)
            observation[r_cut, c_cut] = 0.0

        # Normalization and clipping of pixel observations
        observation = np.clip(self.normalize_pixels(observation), a_min=0.0, a_max=1.0)

        return observation

    def _get_distance(self):
        """
        Calculates the angular separation distance of the Laue diffraction pattern to the closest high-symmetry target.

        Returns
        -------
        float
            Angular distance to the high-symmetry target
        """
        return np.arccos(np.clip(np.dot(self.hkl_target, self.hkl_xax), a_min=-1.0, a_max=1.0))

    def _get_rotation(self):
        """
        Calculate the (chi) rotation of the Laue diffraction pattern w.r.t to the rotational symmetry.

        Returns
        -------
        ang : float
            Rotation angle (deg)
        """
        # Pick a lab‐axis that is least parallel to the current crystal axis
        # (we use the convention that the detector horizontal/vertical axes correspond to either (100), (010), or (001) direction)
        dots = np.abs(self._rot_ref_axes.dot(self.hkl_xax))
        ref_axis = self._rot_ref_axes[np.argmin(dots)]
             
        # Make the lab-axis perpendicular to the beam
        v = ref_axis - (ref_axis.dot(self.hkl_xax))*self.hkl_xax
        v /= np.linalg.norm(v)

        # Project through R‐matrix on the Laue film
        v_cam = self.sim.R @ v
        ang = np.rad2deg(np.arctan2(v_cam[2], v_cam[1]))
        ang = np.abs(ang - self.rot_sym_ndeg * np.round(ang / self.rot_sym_ndeg)) # account for rotational symmetry
        return ang
                
    def _update_closest_target(self):
        """
        Determines the closest high-symmetry target and the coresponding n-fold rotational symmetry.
        """
        min_dist = np.inf

        for t in self.hkl_possible_targets: # NOTE hkl_targets must be normed for consistent results
            cand_cart = self._cart_targets_dict[t]
            cand_cart /= np.linalg.norm(cand_cart, axis=1, keepdims=True)
            dists = np.arccos(np.clip(cand_cart @ self.hkl_xax, a_min=-1.0, a_max=1.0))
            ct = np.argmin(dists)
            if dists[ct] < min_dist:
                min_dist = dists[ct]
                self.hkl_target = cand_cart[ct]
                self.rot_sym_ndeg = 180 / self.laue_pars['crystal_structure'].symmetries[t]['C_n']

    def _calibrate_reciprocal_radius(self, radius):
        """
        Calibrates the reciprocal space radius used for rendering.

        Parameters
        ----------
        radius : float
            Reciprocal radius (in radians)
        """
        self._proj_target_radius = self.sim.d_film * np.tan(2 * radius) * (self.render_shape[0] / self.sim.y_phys_size)

    @staticmethod
    def _gram_schmidt(u, b):
        """
        Constructs a new orthonormal basis using Gram-Schmidt.

        Parameters
        ----------
        u : 1D-array[float]
            First vector
        b : 1D-array[float]
            Second vector (orthogonal to u)

        Returns
        -------
        2D-array[float]
            Orthonormal basis (3x3 matrix)
        """
        # Normalize vectors
        u /= np.linalg.norm(u)
        b /= np.linalg.norm(b)
        
        # Apply Gram–Schmidt orthonormalization b_cart w.r.t. u_cart in true space
        v = b - np.dot(b, u)*u
        v /= np.linalg.norm(v)

        # Complete orthonormal triad
        w = np.cross(u, v)

        return np.column_stack([u, v, w])

if __name__ == "__main__":

    # Testing the Laue environment
    from time import perf_counter
    from laue_simulation import CubicCrystalStructure

    np.random.seed(0)

    laue_pars = {
        'crystal_structure': CubicCrystalStructure(),
        'y_lim': [-50,50],
        'z_lim': [-50,50],
        'beam_yz': [0,0],
        'energy_beam': 50000,
        'space_groups': [221],
        'space_groups_weights': [1],
        'd_film': 40,
        'lat_constants': [3.905, 3.905, 3.905, 90, 90, 90],
        'max_planes': 30,
        'max_spots': 90,
        'r_camera': 0.1
    }

    domain_rnd = {
        'gauss_sigma_range': [1.0, 1.0],
        'remove_fraction': 0.2,
        'spot_shift_sigma': 1.0,
        'max_fraction_random_spots': 0.1,
        'constant_scale_range': [1.0, 1.0]
    }

    laue_env = LaueEnvMeca500(
        laue_pars=laue_pars,
        obs_shape=(84,84),
        render_shape=(1284,1284),
        max_actions=20,
        max_action_change=5, # degrees
        max_target_distance=5, # degrees
        max_target_rotation=360,
        chi_range={'action': [-180, 180], 'initial': [-90,90]},
        phi_range={'action': [-180, 180], 'initial': [-90,90]},
        theta_range={'action': [-180, 180], 'initial': [-90,90]},
        hkl_targets=['001'],
        save_folder="C:/temporary/laue_rl_models/_test",
        domain_randomization=domain_rnd,
        fixed_initial_crystal_angles=np.array([-17, 15, -35]),
        reset_orientation_period=1,
        sparse_rewards=False,
        fix_closest_hkl_target=False,
        coupled_actions=False
    )

    done = False
    total_reward = 0
    distances = []
    rotations = []
    rewards = []
    actions = []

    s = laue_env.reset() # initial state
    laue_env.render()
    scale = 1

    # Environment loop
    while not done:

        t0 = perf_counter()
        
        action = scale*np.array([0., 1., 0.]) if laue_env.chi_rotation else scale*np.array([1., 0.])

        s = laue_env.step(action=action)

        if laue_env.step_nr > 0 and s.reward is not None:
            total_reward += s.reward

        laue_env.update_history(dist=laue_env.currentdist, state=laue_env.state.copy(), reward=s.reward)
        laue_env.render()

        distances.append(laue_env.currentdist)
        rotations.append(laue_env.currentrot)
        rewards.append(s.reward)
        actions.append(action)
        done = laue_env.reset_next_step

        print(f"Step {laue_env.step_nr:02d} performed in {(perf_counter()-t0):.2f} seconds with reward {s.reward:.1f} (summed: {np.sum(rewards):.1f}) ({s.observation['pixels'].min(), s.observation['pixels'].max()})")

    laue_env.history_plot(success=laue_env.done)
    laue_env.reset_history() 
