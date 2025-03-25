import numpy as np


class Reward:
    def __init__(self, env,
                 curriculum_steps=0,
                 tracking_xy_velocity_command_coeff= 1,
                 tracking_yaw_velocity_command_coeff= 0.5,
                 pose_coeff= 0, #0.
                 z_velocity_coeff= -2,
                 pitch_roll_vel_coeff= -0.05,
                 pitch_roll_pos_coeff= -5, # 0.2
                 joint_position_limit_coeff= -10.0, #10
                 joint_torque_coeff= -0.0001,
                 joint_acceleration_coeff= -2.5e-7,
                 joint_velocity_coeff = -0, # 0
                 action_rate_coeff= -0.02, # 0.02
                 collision_coeff= -1,
                 base_height_coeff= -30, #30
                 air_time_coeff= 1, # 0.1
                 stand_still_coeff= -0.25,
                 xy_tracking_temperature= 0.25,
                 yaw_tracking_temperature= 0.25,
                 soft_joint_position_limit= 0.9,
                 max_airtime =0.5):

        self.env = env
        self.tracking_xy_vel_reward = 0
        self.tracking_yaw_vel_reward = 0
        # Tracking rewards
        self.joint_velocity_coeff = joint_velocity_coeff* self.env.dt

        self.joint_acceleration_coeff = joint_acceleration_coeff* self.env.dt
        self.tracking_xy_velocity_command_coeff = tracking_xy_velocity_command_coeff * self.env.dt
        self.tracking_yaw_velocity_command_coeff = tracking_yaw_velocity_command_coeff * self.env.dt
        self.curriculum_steps = curriculum_steps
        self.xy_tracking_temperature = xy_tracking_temperature
        self.yaw_tracking_temperature = yaw_tracking_temperature
        # Cost Coefficients
        self.z_velocity_coeff = z_velocity_coeff * self.env.dt
        self.pitch_roll_vel_coeff = pitch_roll_vel_coeff * self.env.dt
        self.pitch_roll_pos_coeff = pitch_roll_pos_coeff * self.env.dt
        self.pose_coeff = pose_coeff * self.env.dt
        self.joint_position_limit_coeff = joint_position_limit_coeff * self.env.dt
        self.soft_joint_position_limit = soft_joint_position_limit
        self.joint_torque_coeff = joint_torque_coeff * self.env.dt
        self.action_rate_coeff = action_rate_coeff * self.env.dt
        self.collision_coeff = collision_coeff * self.env.dt
        self.base_height_coeff = base_height_coeff * self.env.dt
        self.nominal_trunk_z = self.env.nominal_height
        self.air_time_coeff = air_time_coeff * self.env.dt
        self.stand_still_coeff = stand_still_coeff * self.env.dt
        self._prev_joint_vel = None
        self.max_airtime = max_airtime

    def init(self):
        self.joint_limits = self.env.model.jnt_range[1:].copy()
        joint_limits_midpoint = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        joint_limits_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.joint_limits[:, 0] = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        self.joint_limits[:, 1] = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit

    def setup(self):
        self._prev_joint_vel = np.zeros(self.env.model.nu)
        self.sum_tracking_performance_percentage = 0.0

    def step(self, action):
      self._prev_joint_vel = np.array(self.env.data.qvel[6:])

    def reward_and_info(self, info, done, air_time, first_contact):

        commands = np.array([self.env.goal_x_velocity,self.env.goal_y_velocity,self.env.goal_yaw_velocity])
        cmd_norm = np.linalg.norm(commands)


        # total_timesteps = self.env.total_timesteps * self.env.nr_envs_per_type
        # #
        # curriculum_coeff = min(total_timesteps / self.curriculum_steps, 1.0)
        # if self.env.eval:
        #     curriculum_coeff = 1.0

        curriculum_coeff = 1
        # Tracking velocity command reward
        current_global_linear_velocity = self.env.data.qvel[:3]
        current_local_linear_velocity = self.env.orientation_quat_inv.apply(current_global_linear_velocity)
        desired_local_linear_velocity_xy = np.array([self.env.goal_x_velocity, self.env.goal_y_velocity])
        xy_velocity_difference_norm = np.sum(np.square(desired_local_linear_velocity_xy - current_local_linear_velocity[:2]))
        linear_vel_reward_tracking = np.exp(-xy_velocity_difference_norm / self.xy_tracking_temperature)
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * linear_vel_reward_tracking
        self.tracking_xy_vel_reward = linear_vel_reward_tracking.copy()


        # Tracking angular velocity command reward
        current_local_angular_velocity = self.env.data.qvel[3:6]
        desired_local_yaw_velocity = self.env.goal_yaw_velocity
        yaw_velocity_difference_norm = np.sum(np.square(current_local_angular_velocity[2] - desired_local_yaw_velocity))
        yaw_vel_reward_tracking = np.exp(-yaw_velocity_difference_norm / self.yaw_tracking_temperature)
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * yaw_vel_reward_tracking

        self.tracking_yaw_vel_reward = yaw_vel_reward_tracking.copy()

        # default Pose
        qpos = np.array(self.env.data.qpos[7:]).copy()
        default_pose = self.env.nominal_joint_positions
        weight = np.array([1.0, 1.0, 0.1] * 4)
        pos_reward = curriculum_coeff*self.pose_coeff*np.exp(-np.sum(np.square(qpos - default_pose) * weight))

        # Air time reward

        air_time_reward = np.sum((air_time - self.max_airtime) * first_contact)
        commands = np.array([self.env.goal_x_velocity,self.env.goal_y_velocity,self.env.goal_yaw_velocity])
        cmd_norm = np.linalg.norm(commands[:2])
        air_time_reward *= (cmd_norm > 0.1)
        air_time_reward = curriculum_coeff * self.air_time_coeff * air_time_reward

        # Linear velocity reward
        z_velocity_squared = current_local_linear_velocity[2] ** 2
        linear_velocity_penalty = curriculum_coeff * self.z_velocity_coeff * z_velocity_squared

        # Angular velocity reward
        angular_velocity_norm = np.sum(np.square(current_local_angular_velocity[:2]))
        angular_velocity_penalty = curriculum_coeff * self.pitch_roll_vel_coeff * angular_velocity_norm

        # Angular position reward
        pitch_roll_position_norm = np.sum(np.square(self.env.orientation_euler[:2]))

        angular_position_penalty = curriculum_coeff * self.pitch_roll_pos_coeff * pitch_roll_position_norm

        # print("---------------------------------------------------------------------")
        # print("angular_position_penalty: ", pitch_roll_position_norm, "angular_velocity_penalty: ",angular_velocity_norm)
        # print("---------------------------------------------------------------------")

        # Joint position limit reward
        joint_positions = np.array(self.env.data.qpos[7:])
        lower_limit_penalty = -np.minimum(joint_positions - self.joint_limits[:, 0], 0.0).sum()
        upper_limit_penalty = np.maximum(joint_positions - self.joint_limits[:, 1], 0.0).sum()
        joint_position_limit_penalty = curriculum_coeff * self.joint_position_limit_coeff * (lower_limit_penalty + upper_limit_penalty)

        # Joint   # joint vel limit
        joint_velocities = np.array(self.env.data.qvel[6:])
        lower_limit = -0.9*self.env.max_joint_velocities.copy()
        upper_limit = 0.9*self.env.max_joint_velocities.copy()

        lower_limit_penalty = -np.minimum(joint_velocities - lower_limit, 0.0).sum()
        upper_limit_penalty = np.maximum(joint_velocities - upper_limit, 0.0).sum()
        joint_vel_limit_penalty = curriculum_coeff * self.joint_position_limit_coeff * (
                    lower_limit_penalty + upper_limit_penalty)


        # Joint torques limit
        joint_torques = self.env.data.qfrc_actuator[6:].copy()
        lower_limit = -0.85*self.env.default_torque_limit.copy()
        upper_limit = 0.85*self.env.default_torque_limit.copy()

        lower_limit_penalty = -np.minimum(joint_torques - lower_limit, 0.0).sum()
        upper_limit_penalty = np.maximum(joint_torques - upper_limit, 0.0).sum()
        joint_torques_limits = curriculum_coeff * self.joint_position_limit_coeff * (
                    lower_limit_penalty + upper_limit_penalty)



        velocity_norm = np.sum(np.square(self._prev_joint_vel - np.array(self.env.data.qvel[6:])))
        q_velocity_penalty = curriculum_coeff * self.joint_velocity_coeff * velocity_norm

        # Joint acceleration reward
        acceleration_norm = np.sum(np.square(self._prev_joint_vel - np.array(self.env.data.qvel[6:]) / self.env.dt))
        q_acceleration_penalty = curriculum_coeff * self.joint_acceleration_coeff * acceleration_norm


        # Joint torque reward
        torque_norm = np.sum(np.square(np.array(self.env.data.qfrc_actuator[6:])))
        torque_penalty = curriculum_coeff * self.joint_torque_coeff * torque_norm

        # Action rate reward
        action_rate_norm = np.sum(np.square(self.env.current_action - self.env.last_action))
        action_rate_penalty = curriculum_coeff * self.action_rate_coeff * action_rate_norm

        # Collision reward
        collisions = self.env.check_any_collision_for_all([
            "FR_calf1", "FL_calf1", "RR_calf1", "RL_calf1"
        ])
        trunk_collision = 1 if self.env.check_any_collision(["trunk_1", "trunk_2", "trunk_3"]) else 0
        nr_collisions = sum(collisions.values()) + trunk_collision
        collision_penalty = curriculum_coeff * self.collision_coeff * nr_collisions

        # Walking height
        trunk_z = self.env.data.qpos[2]

        height_difference_squared = (trunk_z - self.nominal_trunk_z) ** 2
        base_height_penalty = curriculum_coeff * self.base_height_coeff * height_difference_squared


        # Stand still
        commands = np.array([self.env.goal_x_velocity,self.env.goal_y_velocity,self.env.goal_yaw_velocity])
        cmd_norm = np.linalg.norm(commands)
        qpos = np.array(self.env.data.qpos[7:]).copy()
        default_pose = self.env.nominal_joint_positions
        stand_still =  np.sum(np.abs(qpos - default_pose)) * (cmd_norm < 0.2)
        stand_still_penalty = curriculum_coeff*self.stand_still_coeff *stand_still

        # # trotting low speeds
        # contacts = np.array(
        #     [self.env.check_collision("floor", foot_name) for foot_name in self.env.foot_names])
        # diagonal_contact = (contacts[0] == contacts[3]) & (contacts[1] == contacts[2])  # (FR == RL) & (FL == RR)
        # contacts_flag = contacts.sum() < 4
        # trot_reward = self.trot_coeff* contacts_flag*diagonal_contact* (cmd_norm > 0.1)
        # if cmd_norm < 0.1:
        #     trot_reward = -self.trot_coeff * (contacts.sum() < 4)

        # Total reward
        tracking_reward = (tracking_xy_velocity_command_reward +
                           tracking_yaw_velocity_command_reward +
                           pos_reward +
                           air_time_reward)

        reward_penalty = (linear_velocity_penalty +
                          angular_velocity_penalty +
                          angular_position_penalty +
                         joint_position_limit_penalty +
                          q_acceleration_penalty +
                          q_velocity_penalty +
                          torque_penalty +
                          action_rate_penalty +
                         collision_penalty +
                          base_height_penalty  + stand_still_penalty)
        # print(f"linear_velocity_penalty: {linear_velocity_penalty/(self.z_velocity_coeff)} \n",
        #        f"angular_velocity_penalty: {angular_velocity_penalty/(self.pitch_roll_vel_coeff )}  \n",
        #         f"angular_position_penalty: {angular_position_penalty/(self.pitch_roll_pos_coeff)}  \n",
        #                  f"joint_position_limit_penalty: {joint_position_limit_penalty/self.joint_position_limit_coeff}  \n",
        #                  f"q_acceleration_penalty: {q_acceleration_penalty/self.joint_acceleration_coeff}  \n"  ,
        #                   f"q_velocity_penalty: {q_velocity_penalty}  \n",
        #                   f" torque_penalty: {torque_penalty/self.joint_torque_coeff} \n" ,
        #                 f" action_rate_penalty: {action_rate_penalty / self.action_rate_coeff} \n",
        #       f" collision_penalty: {collision_penalty / self.collision_coeff} \n",
        #       f" base_height_penalty: {base_height_penalty / self.base_height_coeff} \n",
        #       f" joint_vel_limit_penalty: {joint_vel_limit_penalty / self.joint_position_limit_coeff} \n",
        #       f" joint_torques_limits: {joint_torques_limits / self.joint_position_limit_coeff} \n",
        #       f" stand_still_penalty: {stand_still_penalty} \n")



        reward = tracking_reward + reward_penalty
        reward = max(reward, 0.0)

        # More logging metrics
        # power = np.sum(abs(self.env.current_torques) * abs(self.env.data.qvel[6:]))
        # mass_of_robot = np.sum(self.env.model.body_mass)
        # gravity = -self.env.model.opt.gravity[2]
        # velocity = np.linalg.norm(current_local_linear_velocity)
        # cost_of_transport = power / (mass_of_robot * gravity * velocity)
        # froude_number = velocity ** 2 / (gravity * trunk_z)

        current_global_velocities = np.array([current_local_linear_velocity[0], current_local_linear_velocity[1], current_local_angular_velocity[2]])
        desired_global_velocities = np.array([desired_local_linear_velocity_xy[0], desired_local_linear_velocity_xy[1], desired_local_yaw_velocity])

        tracking_performance_percentage = max(np.mean(1 - (np.abs(current_global_velocities - desired_global_velocities))), 0.0)

        self.sum_tracking_performance_percentage += tracking_performance_percentage
        if done:
            episode_tracking_performance_percentage = self.sum_tracking_performance_percentage / self.env.horizon

        # Info

        info[f"env_info/track_perf_perc/{self.env.SHORT_NAME}"] = tracking_performance_percentage
        if done:
            info[f"env_info/eps_track_perf_perc/{self.env.SHORT_NAME}"] = episode_tracking_performance_percentage


        return reward, info