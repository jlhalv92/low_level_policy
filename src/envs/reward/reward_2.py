import numpy as np


class RudinOwnVarReward:
    def __init__(self, env,
                 tracking_xy_velocity_command_coeff_start=5.0, tracking_xy_velocity_command_coeff_end=2.0,
                 tracking_yaw_velocity_command_coeff=2.0,
                 curriculum_end_steps=10000000, tracking_temperature=0.25,
                 z_velocity_coeff=2e0, pitch_roll_coeff=5e-2, joint_position_limit_coeff=1e1, soft_joint_position_limit=0.9,
                 joint_velocity_coeff=0.01, joint_acceleration_coeff=2.5e-7, joint_torque_coeff=2e-4, action_rate_coeff=1e-2,
                 collision_coeff=6e0, air_time_coeff=1e0, air_time_max=0.5):
        self.env = env
        self.tracking_xy_velocity_command_coeff_start = tracking_xy_velocity_command_coeff_start * self.env.dt
        self.tracking_xy_velocity_command_coeff_end = tracking_xy_velocity_command_coeff_end * self.env.dt
        self.tracking_yaw_velocity_command_coeff = tracking_yaw_velocity_command_coeff * self.env.dt
        self.curriculum_end_steps = curriculum_end_steps
        self.tracking_temperature = tracking_temperature
        self.base_height_coeff = 0
        self.nominal_trunk_z = 0.325
        self.z_velocity_coeff = z_velocity_coeff * self.env.dt
        self.pitch_roll_coeff = pitch_roll_coeff * self.env.dt
        self.joint_position_limit_coeff = joint_position_limit_coeff * self.env.dt
        self.joint_velocity_coeff = joint_velocity_coeff * self.env.dt
        self.joint_acceleration_coeff = joint_acceleration_coeff * self.env.dt
        self.joint_torque_coeff = joint_torque_coeff * self.env.dt
        self.action_rate_coeff = action_rate_coeff * self.env.dt
        self.collision_coeff = collision_coeff * self.env.dt
        self.air_time_coeff = air_time_coeff * self.env.dt
        self.air_time_max = air_time_max

        upper_joint_limits = self.env.default_dof_pos + np.array([0.5, 0.9, 0.9] * 4)
        lower_joint_limits = self.env.default_dof_pos - np.array([0.5, 0.9, 0.9] * 4)
        limits_np = np.array([lower_joint_limits, upper_joint_limits])
        self.joint_limits = limits_np.T
        joint_limits_midpoint = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        joint_limits_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.joint_limits[:, 0] = joint_limits_midpoint - joint_limits_range / 2 * soft_joint_position_limit
        self.joint_limits[:, 1] = joint_limits_midpoint + joint_limits_range / 2 * soft_joint_position_limit

        self._cycles_since_last_touchdown_fr = None
        self._cycles_since_last_touchdown_fl = None
        self._cycles_since_last_touchdown_rr = None
        self._cycles_since_last_touchdown_rl = None
        self._max_step_length_fr = None
        self._max_step_length_fl = None
        self._max_step_length_rr = None
        self._max_step_length_rl = None
        self._prev_joint_vel = None

    def setup(self):
        self.time_since_last_touchdown_fr = 0
        self.time_since_last_touchdown_fl = 0
        self.time_since_last_touchdown_rr = 0
        self.time_since_last_touchdown_rl = 0

        self._prev_joint_vel = np.zeros(12)

    def step(self, action):
        self.time_since_last_touchdown_fr = 0 if self.env.check_collision("floor", "FR_foot") else self.time_since_last_touchdown_fr + self.env.dt
        self.time_since_last_touchdown_fl = 0 if self.env.check_collision("floor", "FL_foot") else self.time_since_last_touchdown_fl + self.env.dt
        self.time_since_last_touchdown_rr = 0 if self.env.check_collision("floor", "RR_foot") else self.time_since_last_touchdown_rr + self.env.dt
        self.time_since_last_touchdown_rl = 0 if self.env.check_collision("floor", "RL_foot") else self.time_since_last_touchdown_rl + self.env.dt

        self._prev_joint_vel = self.env._data.qvel[6:]

    def reward(self, action):
        # Tracking velocity command reward

        current_global_velocity_xy = self.env._data.qvel[:2]
        desired_local_velocity_x = self.env.target_vx
        desired_local_velocity_y = self.env.target_vy
        rotation_angle = self.env.orientation_euler[2]
        desired_global_velocity_x = desired_local_velocity_x * np.cos(rotation_angle) - desired_local_velocity_y * np.sin(rotation_angle)
        desired_global_velocity_y = desired_local_velocity_x * np.sin(rotation_angle) + desired_local_velocity_y * np.cos(rotation_angle)
        desired_global_velocity_xy = np.array([desired_global_velocity_x, desired_global_velocity_y])
        xy_velocity_difference_norm = np.sum(np.square(desired_global_velocity_xy - current_global_velocity_xy))

        tracking_xy_velocity_command_coeff = self.tracking_xy_velocity_command_coeff_start + \
                                                (self.tracking_xy_velocity_command_coeff_end - self.tracking_xy_velocity_command_coeff_start) * \
                                                min(self.env.total_timesteps / self.curriculum_end_steps, 1.0)
        tracking_xy_velocity_command_reward = (tracking_xy_velocity_command_coeff *
                                               np.exp(-xy_velocity_difference_norm / self.tracking_temperature))

        # Tracking angular velocity command reward
        current_local_angular_velocity = self.env._data.qvel[3:6]
        desired_local_yaw_velocity = self.env.target_yaw
        yaw_velocity_difference_norm = np.sum(np.square(current_local_angular_velocity[2] - desired_local_yaw_velocity))
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * np.exp(-yaw_velocity_difference_norm / self.tracking_temperature)

        # Linear velocity reward

        rot_mat_torso = self.env._data.body("trunk").xmat.reshape(3, 3)
        current_global_trunk_velocity = self.env._data.qvel[:3]
        current_trunk_velocity = np.matmul(rot_mat_torso.T, current_global_trunk_velocity)
        z_velocity_squared = current_trunk_velocity[2] ** 2
        linear_velocity_reward = self.z_velocity_coeff * -z_velocity_squared

        # Angular velocity reward

        current_local_angular_velocity = self.env._data.qvel[3:6][:2]
        angular_velocity_norm = np.sum(np.square(current_local_angular_velocity))
        angular_velocity_reward = self.pitch_roll_coeff * -angular_velocity_norm

        # Joint position limit reward
        joint_positions = np.array(self.env._data.qpos[7:])
        lower_limit_penalty = -np.minimum(joint_positions - self.joint_limits[:, 0], 0.0).sum()
        upper_limit_penalty = np.maximum(joint_positions - self.joint_limits[:, 1], 0.0).sum()
        joint_position_limit_reward = self.joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

        # Joint velocity reward
        velocity_norm = np.sum(np.square(np.array(self.env._data.qvel[6:])))
        velocity_reward = self.joint_velocity_coeff * -velocity_norm

        # Joint acceleration reward
        acceleration_norm = np.sum(np.square(self._prev_joint_vel - self.env._data.qvel[6:]/ self.env.dt))
        acceleration_reward = self.joint_acceleration_coeff * -acceleration_norm

        # Joint torque reward
        torque_norm = np.sum(np.square(self.env._data.qfrc_actuator[6:]))
        torque_reward = self.joint_torque_coeff * -torque_norm

        # Action rate reward
        action_rate_norm = np.sum(np.square(action - self.env.last_action))
        action_rate_reward = self.action_rate_coeff * -action_rate_norm

        # Collision reward
        collisions = self.env.check_any_collision_for_all(["FR_calf1", "FL_calf1", "RR_calf1", "RL_calf1" ])
        trunk_collision = 1 if self.env.check_any_collision(["trunk_1", "trunk_2", "trunk_3", "trunk_4", "trunk_5", "trunk_6", "trunk_7", "trunk_8"]) else 0

        nr_collisions = sum(collisions.values()) + trunk_collision
        collision_reward = self.collision_coeff * -nr_collisions


        # Air time reward

        air_time_reward = 0.0
        foot_fr_on_ground = self.env.check_collision("floor", "FR_foot")
        foot_fl_on_ground = self.env.check_collision("floor", "FL_foot")
        foot_rr_on_ground = self.env.check_collision("floor", "RR_foot")
        foot_rl_on_ground = self.env.check_collision("floor", "RL_foot")
        if foot_fr_on_ground:
            air_time_reward += self.time_since_last_touchdown_fr - self.air_time_max
        if foot_fl_on_ground:
            air_time_reward += self.time_since_last_touchdown_fl - self.air_time_max
        if foot_rr_on_ground:
            air_time_reward += self.time_since_last_touchdown_rr - self.air_time_max
        if foot_rl_on_ground:
            air_time_reward += self.time_since_last_touchdown_rl - self.air_time_max
        air_time_reward = self.air_time_coeff * air_time_reward


        # Walking height
        trunk_z = self.env._data.qpos[2]
        height_difference_squared = (trunk_z - self.nominal_trunk_z) ** 2
        base_height_reward =  self.base_height_coeff * -height_difference_squared

        # Total reward
        reward_move = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward

        penalty =  + linear_velocity_reward + \
                 angular_velocity_reward + joint_position_limit_reward + velocity_reward + acceleration_reward + \
                 torque_reward + action_rate_reward + collision_reward + air_time_reward + base_height_reward

        reward = reward_move + penalty

        reward = max(reward, 0.0)

        return reward
