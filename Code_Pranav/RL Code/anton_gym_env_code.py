import genesis as gs
import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env import VecEnv

class PendulumBalanceEnv(VecEnv):

    ##### VecEnv Overrides ####################
    def __init__(self, vis, device='cpu',
                 max_steps=2000,
                 max_torque=5, max_speed=15, num_envs=10,
                 reset_on_completion=True,
                 dt=0.1):

        if str.lower(device) == 'cpu':
            gs.init(backend=gs.cpu, precision="32", logging_level='warning')
        else:
            print("ERROR! Current no other device than CPU supported")
        self.vis = vis

        self.max_torque = max_torque
        self.max_speed = max_speed
        
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))  # Torque applied at the pivot
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,)) # Angle (theta) and angular velocity (theta_dot)

        super().__init__(num_envs,
                         observation_space,
                         action_space)
        
        self.scene = gs.Scene(
            viewer_options=
            gs.options.ViewerOptions(
                camera_pos=(0.0, 20.0, 5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=dt),
            show_viewer=vis
        )

        self.cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (0.0, 20.0, 5),
                lookat = (0.0, 0.0, 0.5),
                fov    = 30,
                GUI    = False
            )

        self.pendulum = self.scene.add_entity(
            gs.morphs.URDF(
                file="./assets/pendulum.urdf",  # Path to your URDF file
                fixed=True
            )
        )


        plane = self.scene.add_entity(gs.morphs.Plane())

        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        self.actions = np.zeros([num_envs, 1], dtype=np.float32)
        self.step_counts = np.zeros(num_envs, dtype=np.int32)

        self.MAX_STEPS = max_steps
        self.POS_THRESHOLD = np.deg2rad([1.0])
        self.VEL_THRESHOLD = np.deg2rad([0.5])
        self.RESET_ON_COMPLETION = reset_on_completion

        self.set_positions()

    def reset_(self, dones):
        
        num_resets = dones.sum()

        position = np.pi * np.ones([num_resets, 1])
        velocity = np.zeros([num_resets, 1])
        # Set angular position
        self.pendulum.set_dofs_position(position, envs_idx=self.envs_idx[dones])
        # Set angular velocity
        self.pendulum.set_dofs_velocity(velocity, envs_idx=self.envs_idx[dones])
        self.step_counts[dones] = np.zeros(num_resets, dtype=np.int32) 

        return self._get_observation()

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):

        self.step_counts += 1

        self.actions = np.clip(
            self.max_torque * self.actions,
            a_min=-self.max_torque * np.ones_like(self.actions),
            a_max= self.max_torque * np.ones_like(self.actions)
        ).reshape([self.num_envs, 1])  
        # Apply torque within limits
        self.pendulum.control_dofs_force(self.actions)
        self.scene.step()

        # Get pendulum state
        state = self._get_state()
        theta = state[:, 0]
        theta_dot = state[:, 1]

        # Episode ends if the pendulum falls
        max_steps_reached = self.step_counts > self.MAX_STEPS
        reached_goal = (  (np.abs(theta) < self.POS_THRESHOLD)
                        & (np.abs(theta_dot) < self.VEL_THRESHOLD))
        dones = (max_steps_reached | (reached_goal & self.RESET_ON_COMPLETION) )

        # Compute reward
        upright_bonus = -np.square(theta)                                       # Encourage theta close to 0
        velocity_bonus = -0.1 * np.square(theta_dot)                            # Discourage high velocities
        torque_bonus = -0.001 * np.square(self.actions.flatten())               # Penalize high actuation
        step_cost = 0.0                                                         # Encourage fast solutions
        reached_goal_bonus = reached_goal * 1e2
        rewards = (upright_bonus + velocity_bonus                               # Combine Bonuses
                    + torque_bonus + reached_goal_bonus + step_cost)
        # Write info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self._get_observation()
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True

        # Reset done environments
        self.reset_(dones=dones)

        return self._get_observation(), rewards, dones, infos
    
    def close(self):
        pass
    
    def seed(self):
        pass

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in range(self.num_envs)]
    
    def set_attr(self, attr_name, value, indices=None):
        pass
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]*self.num_envs
    

    ##### Additional methods ###################

    def set_positions(self, spacing=1.5):
        num_per_side = int(np.ceil(np.sqrt(self.num_envs)))  # Number of rows/cols
        x_idx, y_idx = np.meshgrid(range(num_per_side), range(num_per_side))
        
        # Select the first num_envs positions
        grid_positions = np.stack([x_idx.ravel(), y_idx.ravel()], axis=1)[:self.num_envs]
        
        # Center the grid at the origin
        grid_positions = (grid_positions - np.mean(grid_positions, axis=0)) * spacing
        positions = np.column_stack([grid_positions, np.ones(self.num_envs) * 1.1])  # Add Z coordinate
        
        self.pendulum.set_pos(positions)
        self.scene.step()

    def simulate(self, steps=100, control_inputs=None):
        """
        Simulates the pendulum environment for a given number of steps.
        
        Args:
            steps (int): Number of simulation steps to run.
            control_inputs (np.array): Control torques to apply (optional, shape: [steps, num_envs, 1]).
        """
        for t in range(steps):
            # Apply control input if provided
            if control_inputs is not None:
                self.pendulum.control_dofs_force(control_inputs[t, :, :])

            # Step the simulation
            self.scene.step()

            # Retrieve and print the state for debugging (optional)
            angular_pos = self.pendulum.get_dofs_position()
            angular_vel = self.pendulum.get_dofs_velocity()
            print(f"Step {t}: Position={angular_pos}, Velocity={angular_vel}")

    def _set_state(self, theta, theta_dot=None):

        if theta_dot is None:
            theta_dot = np.zeros(self.num_envs)

        assert len(theta) == self.num_envs, "Wrong number of positions given!"
        assert len(theta_dot) == self.num_envs, "Wrong number of velocities given!"

        # Set angular position
        self.pendulum.set_dofs_position(theta, envs_idx=self.envs_idx)
        # Set angular velocity
        self.pendulum.set_dofs_velocity(theta_dot, envs_idx=self.envs_idx)

    def _get_state(self):
        theta = self.pendulum.get_dofs_position()
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        theta_dot = self.pendulum.get_dofs_velocity()
        return np.concatenate((theta, theta_dot), axis=1)
    
    def _get_observation(self):
            angle = self.pendulum.get_dofs_position()
            return np.concatenate([
                 np.cos(angle),
                 np.sin(angle),
                 self.pendulum.get_dofs_velocity() / self.max_speed], 
                 axis=1)
    