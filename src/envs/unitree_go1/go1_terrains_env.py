from pathlib import Path
import psutil
import mujoco
import numpy as np
from collections import deque

from cv2.version import rolling
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from src.envs.reward_function import Reward
from src.envs.viewer import MujocoViewer
import pygame
import cv2
from src.utils.curriculum.curriculum import RewardThresholdCurriculum
import matplotlib.pyplot as plt


def quat_from_euler_xyz(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qw, qx, qy, qz])



def keyboard_control(env, vx, vy, wz):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            print("running off")

    keys = pygame.key.get_pressed()


    if keys[pygame.K_UP]==False and keys[pygame.K_DOWN]==False:
        vx = 0
    if keys[pygame.K_LEFT]==False and keys[pygame.K_RIGHT]==False:
        vy = 0
    if keys[pygame.K_w]==False and keys[pygame.K_e]==False:
        wz = 0

    if keys[pygame.K_UP]==True:
        vx = min(vx + 0.5,env.x_limit)

    if keys[pygame.K_DOWN]:
        vx = max(vx - 0.5,-env.x_limit)

    if keys[pygame.K_LEFT]:
        vy = min(vy + 0.5,env.y_limit)
    if keys[pygame.K_RIGHT]:
        vy = max(vy - 0.5,-env.y_limit)
    if keys[pygame.K_w]:
        wz = min(wz + 0.5,env.wz_limit)

    if keys[pygame.K_e]:
        wz = max(wz - 0.5,-env.wz_limit)

    return vx,vy,wz


class Airtime:
    def __init__(self):
        self.last_contact = np.zeros(4)
        self.feet_air_time = np.zeros(4)

    def calculate_airtime_before(self, contact):
        contact_filt = np.logical_or(contact, self.last_contact)
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += 0.02
        return first_contact, self.feet_air_time

    def calculate_airtime_after(self, contact):
        self.feet_air_time *= np.logical_not(contact)
        self.last_contact = contact

class PDController:
    def __init__(self, env, control_frequency_hz=50, p_gain=20, d_gain=0.5, scaling_factor=0.25):
        self.env = env #20, 0.5
        self.control_frequency_hz = control_frequency_hz
        self.apply_scale_shoulder = False
        self.scaling_factor = scaling_factor
        self.kp_noise_factor = 1.0
        self.kd_noise_factor = 1.0
        self.motor_strength_factor = np.ones(12)
        self.joint_position_offset = np.zeros(12)
        self.kp = p_gain
        self.kd = d_gain
        self.action_scale = scaling_factor
        self.default_kp = p_gain
        self.default_kd = d_gain
        self.default_action_scale = scaling_factor

    def get_gains_and_scale_factor(self):

        return np.array([self.kp, self.kd, self.action_scale])

    def update_factor_and_gains(self):
        self.kp = self.default_kp + np.random.uniform(low=-2, high=2)
        self.kd = self.default_kd + np.random.uniform(low=-0.1, high=0.2)
        self.action_scale = self.default_action_scale

    def action2torques(self, action):

        current_qpos = self.env.data.qpos[7:].copy()
        current_qvel = self.env.data.qvel[6:].copy()

        scaled_action = action * self.action_scale
        if self.apply_scale_shoulder:
            scaled_action[[0, 3, 6, 9]] *= 0.5
        target_joint_positions = self.env.nominal_joint_positions.copy() + scaled_action

        torques = self.kp * self.kp_noise_factor * (target_joint_positions - current_qpos + self.joint_position_offset)- self.kd * self.kd_noise_factor * current_qvel

        return torques * self.motor_strength_factor

class UnitreeGo1(gym.Env):
    LONG_NAME = "unitree_go1"
    SHORT_NAME = "go1"


    def __init__(self,
                 seed,
                 env_id=0,
                 timestep=0.005,
                 episode_length_in_seconds=20,
                 total_nr_envs=100,
                 nr_envs_per_type=100,
                 cpu_id=None,
                 joystick = False,
                 domain_randomization=False,
                 **env_args):

        if cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([cpu_id, ])

        self.env_id = env_id
        print("Created env {}".format(self.env_id))
        self._n_episode = 0
        self.episode_step = 0
        self.resampling_command_steps = 500 # 500
        self.joystick = joystick
        self.env_args = env_args
        self.tracking_velocities_array = []
        self.track_velocities = False
        if self.joystick:
            pygame.init()
            screen_width, screen_height = 200, 200
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Keyboard Joystick Simulation")
        self.x_limit =  env_args["vx_limit"]# 1.2, 3.5
        self.y_limit = env_args["vy_limit"]
        self.wz_limit =  env_args["wz_limit"] #1.5, 5.5
        self.vel_curriculum = RewardThresholdCurriculum(42,
                                                        x=(-5, 5, 51),
                                                        y=(-0.6 , 0.6 ,2),
                                                        z=(-5, 5,51))


        # low, high = np.array([-1, -1, -1]), np.array([1, 1, 1])
        low, high = np.array([-1, -1, -1]), np.array([1, 1, 1])




        self.vel_curriculum.set_to(low, high, value=1.0)
        _, old_batch = self.vel_curriculum.sample(1)
        self.old_batch = old_batch



        self.airtime = Airtime()
        self.seed = seed
        np.seed = self.seed

        # self.mode = mode
        self.mass = None
        self.rendering = False
        self.episode_count = 0
        self.domain_randomization = domain_randomization
        # self.missing_value = missing_value
        self.total_nr_envs = total_nr_envs
        self.nr_envs_per_type = nr_envs_per_type
        self.eval = False
        self.np_rng = np.random.default_rng(self.seed)
        self.nominal_height = 0.34 #0.325  0.3
        # self.nominal_joint_positions = np.array([
        #     -0.2, -0.7, 1.25,
        #     0.2, 0.7, -1.25,
        #     0.2, -1, 1.35,
        #     -0.2, 1, -1.35,
        # ])
        self.nominal_joint_positions = np.array([
            -0.2, -0.75, 1.5,
            0.2, 0.75, -1.5,
            0.2, -1.0, 1.5,
            -0.2, 1.0, -1.5,
        ])

        self.max_joint_velocities = np.array([30.1, 30.1, 20.06] * 4)
        self.initial_drop_height = 0.30 #0.325 0.34

        self.total_timesteps = 0

        self.goal_x_velocity = 0
        self.goal_y_velocity = 0
        self.goal_yaw_velocity = 0


        self.controller = PDController(self)
        self.gains_and_action_scaling_factor = self.controller.get_gains_and_scale_factor()
        self.control_frequency_hz = self.controller.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / timestep))
        self.nr_intermediate_steps = 1
        self.dt = timestep * self.nr_substeps * self.nr_intermediate_steps

        self.horizon = int(round(episode_length_in_seconds * self.control_frequency_hz))
        self.reward_function = Reward(self)
        self.vxy_tracking = 0
        self.wz_tracking = 0
        self.episode_len = 0
        self.command_steps = 0

        xml_path = "src/envs/unitree_go1/walk_go1.xml"
        # xml_path = "src/envs/unitree_go1/navigation_go1_rays_test.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)


        # for geom_id in range(self.model.ngeom):
        #     geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        #     print(f"Geom ID: {geom_id}, Geom Name: {geom_name}")


        self.default_mass = self.model.body("trunk").mass.copy()
        self.default_inertia = self.model.body("trunk").inertia.copy()


        # System Default Properties

        self.info = {
            "steps_until_next_pert": 0,
            "pert_duration_seconds": 0,
            "pert_duration": 0,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": np.zeros(3),
            "pert_mag": 0,
        }

        self.default_torque_limit = self.model.actuator_ctrlrange[:, 1].copy()

        self.default_joint_damping = self.model.dof_damping[6:].copy()
        self.joint_damping_min = 0
        self.joint_damping_max = 0.2

        self.joint_armature_min = 0.009
        self.joint_armature_max = 0.02
        self.default_joint_armature = self.model.dof_armature[6:].copy()

        self.joint_stiffness_min = 0.0
        self.joint_stiffness_max = 0.5
        self.default_joint_stiffness = self.model.jnt_stiffness[1:].copy()

        self.joint_friction_loss_min = 0.0
        self.joint_friction_loss_max = 0.65
        self.default_joint_frictionloss = self.model.dof_frictionloss[6:].copy()

        self.default_trunk_com = self.model.body_ipos[1].copy()
        self.add_center_mass_off_min = -0.05
        self.add_center_mass_off_max = 0.05
        self.default_gravity = -self.model.opt.gravity[2].copy()

        self.friction_tangential_min = 0.4
        self.friction_tangential_max = 1.5

        self.friction_torsional_ground_min = 0.0001
        self.friction_torsional_ground_max = 0.01

        self.friction_torsional_feet_min = 0.0001
        self.friction_torsional_feet_max = 0.04

        self.friction_rolling_ground_min = 0.0001
        self.friction_rolling_ground_max = 0.002

        self.friction_rolling_feet_min = 0.0001
        self.friction_rolling_feet_max = 0.02

        self.damping_min = 30
        self.damping_max = 130
        self.stiffness_min = 500
        self.stiffness_max = 1500
        self.kick = np.zeros(3)

        tangential_friction = self.model.geom_friction[0][0]
        stiffness = -self.model.geom_solref[:, 0][0]
        damping = -self.model.geom_solref[:, 1][0]

        self.friction = tangential_friction
        self.stiffness = stiffness
        self.damping = damping



        collision_groups = [("floor", ["floor"]),
                            ("feet", ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
                            ("FL_foot", ["FL_foot"]), ("FR_foot", ["FR_foot"]), ("RL_foot", ["RL_foot"]),
                            ("RR_foot", ["RR_foot"]),
                            ("FR_calf", ["FR_calf1"]),
                            ("FL_calf", ["FL_calf1"]),
                            ("RR_calf", ["RR_calf1"]),
                            ("RL_calf", ["RL_calf1"]),
                            ("FR_thigh", ["FR_thigh1","FR_thigh2","FR_thigh3"]),
                            ("FL_thigh", ["FL_thigh1","FL_thigh2","FL_thigh3"]),
                            ("RR_thigh", ["RR_thigh1","RR_thigh2","RR_thigh3"]),
                            ("RL_thigh", ["RL_thigh1","RL_thigh2","RL_thigh3"]),
                            ("trunk", ["trunk_1", "trunk_2", "trunk_3", "trunk_4", "trunk_5", "trunk_6", "trunk_7", "trunk_8"]),
                            ("trunk_1", ["trunk_1"]), ("trunk_2", ["trunk_2"]), ("trunk_3", ["trunk_3"])]
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) for
                                               geom_name in geom_names}


        self.viewer =None
        action_space_size = self.model.nu
        action_space_low = -np.ones(action_space_size) * np.inf
        action_space_high = np.ones(action_space_size) * np.inf
        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float64)

        self.foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        self.feet_ids = [self.model.geom(foot).id for foot in self.foot_names]

        #
        qvel = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            *([0.0] * self.model.nu)
        ]
        qpos = [
            0.0, 0.0, self.initial_drop_height,
            1.0, 0.0, 0.0, 0.0,
            *self.nominal_joint_positions
        ]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

        mujoco.mj_forward(self.model, self.data)

        self.update_orientation_attributes()

        self.observation_space, self.observation_name_to_id = self.get_observation_space()

        self.current_action = np.zeros(self.model.nu)
        self.last_action = np.zeros(self.model.nu)

        self.initial_observation = self.get_initial_observation()



        self.joint_positions_idx = [self.observation_name_to_id[joint_name + "_position"] for joint_name in
                                               self.joint_names]

        self.joint_velocities_idx = [self.observation_name_to_id[joint_name + "_velocity"] for joint_name in
                                                self.joint_names]

        self.joint_previous_actions_idx = [self.observation_name_to_id[joint_name + "_previous_action"] for
                                                      joint_name in self.joint_names]

        self.foot_ground_contact_idx = [self.observation_name_to_id[foot_name + "_ground_contact"] for
                                                   foot_name in self.foot_names]

        self.foot_airtime_idx = [
            self.observation_name_to_id[foot_name + "_airtime"] for foot_name in
            self.foot_names]

        self.trunk_linear_vel_idx = [self.observation_name_to_id["trunk_" + observation_name] for
                                                observation_name in ["x_velocity", "y_velocity", "z_velocity"]]

        self.trunk_angular_vel_idx = [self.observation_name_to_id["trunk_" + observation_name] for
                                                 observation_name in
                                                 ["roll_velocity", "pitch_velocity", "yaw_velocity"]]

        self.goal_velocity_idx = [self.observation_name_to_id["goal_" + observation_name] for
                                             observation_name in ["x_velocity", "y_velocity", "yaw_velocity"]]

        self.projected_gravity_idx = [self.observation_name_to_id["projected_gravity_" + observation_name]
                                                 for observation_name in ["x", "y", "z"]]
        self.height_idx = [self.observation_name_to_id["height_0"]]

        self.mass_idx = [self.observation_name_to_id["mass"]]
        # self.kp_idx = [self.observation_name_to_id["kp_gain"]]
        # self.kd_idx = [self.observation_name_to_id["kd_gain"]]
        # self.scale_idx = [self.observation_name_to_id["action_scaling_factor"]]
        #
        self.friction_idx = [self.observation_name_to_id["friction"]]
        self.stiffness_idx = [self.observation_name_to_id["stiffness"]]
        self.damping_idx = [self.observation_name_to_id["damping"]]
        # self.gravity_idx = [self.observation_name_to_id["gravity"]]
        # self.tangential_friction_idx = [self.observation_name_to_id["tangential_friction"]]
        # self.torsional_friction_ground_idx = [self.observation_name_to_id["torsional_friction_ground"]]
        # self.rolling_friction_ground_idx = [self.observation_name_to_id["rolling_friction_ground"]]
        # self.rolling_friction_feet_idx = [self.observation_name_to_id["rolling_friction_feet"]]
        # self.torsional_friction_feet_idx = [self.observation_name_to_id["torsional_friction_feet"]]

        self.reward_function.init()




    def initial_state(self):
        nominal_joint_pos = self.nominal_joint_positions.copy()
        initial_joint_pos = nominal_joint_pos.copy()

        qpos = np.array([
            0.0, 0.0, self.initial_drop_height,
            *[1,0,0,0],
            *initial_joint_pos
        ])
        # init_qvel = np.random.uniform(low=-0.8, high=0.8, size=12)
        init_linear = np.random.uniform(low=-0.5, high=0.5, size=3)*0
        init_angular = np.random.uniform(low=-0.5, high=0.5, size=3)*0
        # init_linear = [0,0,0]
        # init_angular = [0,0,0]
        init_qvel = np.zeros(12)
        qvel = np.array([
            *init_linear,
            *init_angular,
            *init_qvel
        ])

        return qpos, qvel


    def reset(self, seed=None, options=None):

        self.episode_step = 0
        self.vxy_tracking = 0
        self.wz_tracking = 0
        # self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity = self.get_next_command()
        self.current_action = np.zeros(self.model.nu)
        self.last_action = np.zeros(self.model.nu)
        self.current_torques = np.zeros(self.model.nu)
        #if not self.eval:
            #if self._n_episode % 1000 == 0:
                # np.save(f"data/grid_{self.env_id}_episode_{self._n_episode}.npy", self.vel_curriculum.grid)
                #np.save(f"data/weights_{self.env_id}_episode_{self._n_episode}.npy", self.vel_curriculum.weights)

        self._n_episode +=1
        self.reward_function.setup()

        qpos, qvel = self.initial_state()

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.update_orientation_attributes()
        self.airtime = Airtime()
        # if self.viewer:
        #     self.viewer.render(self.data)

        return self.get_observation(), {}

    def get_next_command(self):

        timesteps = self.resampling_command_steps
        vx_tracking = self.vxy_tracking / timesteps
        vx_tracking = self.vxy_tracking / timesteps
        wz_tracking = self.wz_tracking / timesteps

        if self.eval:
            cvx = np.random.uniform(-self.x_limit,self.x_limit)
            cvy = np.random.uniform(-self.y_limit,self.y_limit)
            cwz = np.random.uniform(-self.wz_limit,self.wz_limit)
            # self.nominal_height = np.random.uniform(0.19,0.28)
            v_command = np.array([cvx, cvy, cwz])

        else:


            self.vel_curriculum.update(self.old_batch, np.array([vx_tracking, ]), np.array([wz_tracking, ]), .8, 0.5, 0.5)

            commands, old_batch = self.vel_curriculum.sample(1)
            v_command = commands[0].copy()
            self.old_batch = old_batch

        v_command[:2] *= (np.linalg.norm(v_command[:2]) > 0.2)
        v_command[0] *= (np.abs(v_command[0])> 0.2)
        v_command[1] *= (np.abs(v_command[1])> 0.2)
        v_command[2] *= (np.abs(v_command[2]) > 0.2)

        return v_command

        self.vxy_tracking = 0
        self.wz_tracking = 0



        return v_command

    def should_terminate(self, obs):

        # trunk_collision = False
        trunk_collision = True if self.check_any_collision(["trunk"]) else False
        thigh_collision = True if self.check_any_collision(["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]) else False

        fall = self.data.qpos[2] < 0.1

        return trunk_collision or thigh_collision or fall



    def step(self, action):

        if self.track_velocities:
            vx,vy,vz = self.orientation_quat_inv.apply(self.data.qvel[:3]).copy()
            wx,wy,wz=self.data.qvel[3:6].copy()
            self.tracking_velocities_array.append([vx,vy,wz])
        if self.domain_randomization:

            should_update_mass = np.random.uniform() < 0.002
            should_update_controller = np.random.uniform() < 0.002
            should_update_properties = np.random.uniform() < 0.002
            should_kick_robot = np.random.uniform() < 0.002

            if should_kick_robot:
                kick_x_min = -1
                kick_x_max = 1
                kick_y_min = -1
                kick_y_max = 1
                kick_z_min = -1
                kick_z_max = 1
                kick_x = np.random.uniform(kick_x_min, kick_x_max)
                kick_y = np.random.uniform(kick_y_min, kick_y_max)
                kick_z = np.random.uniform(kick_z_min, kick_z_max)
                self.kick = np.array([kick_x, kick_y, kick_z])
                if np.random.uniform() < 0.001:
                    self.data.qvel[:3] = self.kick
                else:
                    self.data.qvel[:3] += self.kick
                # self.info["pert_steps"] += 1



            if should_update_mass:
                delta_mass = np.random.uniform(-1, 1)
                new_mass = self.default_mass + delta_mass
                self.model.body("trunk").mass = new_mass
                self.model.body("trunk").inertia = (new_mass / self.default_mass) * self.default_inertia

            if should_update_controller:
                self.controller.update_factor_and_gains()
                self.controller.joint_position_offset = np.random.uniform(-0.05, 0.05, size=(12,))
                torque_limit = self.default_torque_limit *  np.random.uniform(0.9,1.1, size=self.default_torque_limit.shape)
                self.model.actuator_ctrlrange[:, 1] = torque_limit
                self.model.actuator_ctrlrange[:, 0] = -torque_limit
                self.controller.motor_strength_factor =  np.random.uniform(0.9,1.1, size=self.model.nu)

            if should_update_properties:
                factor_friction = np.random.uniform(0,1)
                factor_stiffness = np.random.uniform(0,1)
                factor_damping = np.random.uniform(0,1)

                tangential_friction =  self.friction_tangential_min + (self.friction_tangential_max - self.friction_tangential_min)*factor_friction
                damping = self.damping_min + (self.damping_max - self.damping_min)*factor_stiffness
                stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min)*factor_damping
                gravity = self.default_gravity +  np.random.uniform(-1, 1)
                # torsional_friction_ground =  self.friction_torsional_ground_min + (self.friction_torsional_ground_max - self.friction_torsional_ground_min)*factor
                # rolling_friction_ground =  self.friction_rolling_ground_min + (self.friction_rolling_ground_max - self.friction_rolling_ground_min)*factor
                #
                # #
                # rolling_friction_feet = self.friction_rolling_feet_min + (self.friction_rolling_feet_max - self.friction_rolling_feet_min)*factor
                # torsional_friction_feet = self.friction_torsional_feet_min + (self.friction_torsional_feet_max - self.friction_torsional_feet_min)*factor

                # self.model.geom_friction[0] = [tangential_friction,
                #                                torsional_friction_ground,
                #                                rolling_friction_ground]
                #
                # self.model.geom_friction[self.feet_ids] = [tangential_friction,
                #                                               rolling_friction_feet,
                #                                               torsional_friction_feet]

                # self.model.geom_friction[self.feet_ids] =tangential_friction
                self.model.geom_friction[0] = tangential_friction
                self.model.geom_solref[:, 0] = -stiffness
                self.model.geom_solref[:, 1] = -damping
                self.model.opt.gravity[2] = -gravity

                self.friction = tangential_friction
                self.stiffness = stiffness
                self.damping = damping



                joint_damping_noise_factor =  np.random.uniform(0.9,1.1)
                joint_armature_noise_factor =  np.random.uniform(0.9,1.1)
                joint_stiffness_noise_factor = np.random.uniform(0.9, 1.1)
                joint_friction_loss_noise_factor = np.random.uniform(0.9, 1.1)
                # center_mass_off_noise_factor =  np.random.uniform(0.9,1.1)

                joint_damping = self.default_joint_damping.copy()
                joint_armature = self.default_joint_armature.copy()
                joint_stiffness = self.default_joint_stiffness.copy()
                joint_frictionloss = self.default_joint_frictionloss.copy()

                self.model.dof_damping[6:] = joint_damping * joint_damping_noise_factor
                self.model.dof_armature[6:] = joint_armature * joint_armature_noise_factor
                self.model.jnt_stiffness[1:] = joint_stiffness * joint_stiffness_noise_factor
                self.model.dof_frictionloss[6:] = joint_frictionloss * joint_friction_loss_noise_factor
                self.model.body_ipos[1] = (self.default_trunk_com + np.random.uniform(self.add_center_mass_off_min,
                                                                                      self.add_center_mass_off_max))

        if self.joystick:

            vx, vy, wz = keyboard_control(self, self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity)
            self.goal_x_velocity = vx
            self.goal_y_velocity = vy
            self.goal_yaw_velocity = wz
        else:

            should_sample_commands = self.episode_step % self.resampling_command_steps == 0

            if should_sample_commands:

                self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity = self.get_next_command()

        torque_limits = self.model.actuator_ctrlrange[:,1].copy()
        torques = np.clip(self.controller.action2torques(action),-torque_limits,torque_limits)

        for _ in range(self.nr_intermediate_steps):
            self.data.ctrl = torques.copy()
            mujoco.mj_step(self.model, self.data, self.nr_substeps)
            self.data.qvel[6:] = np.clip(self.data.qvel[6:], -self.max_joint_velocities, self.max_joint_velocities)


        self.update_orientation_attributes()

        self.current_action = action.copy()
        self.current_torques = torques

        contacts = np.array(
            [self.check_collision("floor", foot_name) for foot_name in self.foot_names])

        next_observation = self.get_observation()

        terminated = self.should_terminate(next_observation)

        truncated = self.episode_step >= self.horizon-1

        done = terminated | truncated
        if done:
            self.episode_count +=1

        first_contact, air_time = self.airtime.calculate_airtime_before(contacts)
        reward, r_info = self.get_reward(done, first_contact, air_time)
        self.airtime.calculate_airtime_after(contacts)
        info = {**r_info}

        self.reward_function.step(action)
        self.last_action = action.copy()


        self.vxy_tracking +=self.reward_function.tracking_xy_vel_reward
        self.wz_tracking +=self.reward_function.tracking_yaw_vel_reward


        self.episode_len +=1
        self.episode_step += 1
        self.total_timesteps += 1
        return next_observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer= MujocoViewer(self.model, self.dt)
        self.viewer.render(self.data)

    def update_orientation_attributes(self):
        self.orientation_quat = R.from_quat([self.data.qpos[4],
                                             self.data.qpos[5],
                                             self.data.qpos[6],
                                             self.data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()

    def get_observation_space(self):
        observation_names = []
        space_low, space_high = [], []

        # Joint
        for joint_name in self.joint_names:
            observation_names.extend([
                joint_name + "_position"])
        space_low.extend([-np.inf] * (12))
        space_high.extend([np.inf] * (12))

        for joint_name in self.joint_names:
            observation_names.extend([
                joint_name + "_velocity"])
        space_low.extend([-np.inf] * (12))
        space_high.extend([np.inf] * (12))

        for joint_name in self.joint_names:
            observation_names.extend([
                joint_name + "_previous_action"])
        space_low.extend([-np.inf] * (12))
        space_high.extend([np.inf] * (12))

        for foot_name in self.foot_names:
            observation_names.extend([
                foot_name + "_ground_contact",
            ])
        space_low.extend([-np.inf] * (4))
        space_high.extend([np.inf] * (4))

        for foot_name in self.foot_names:
            observation_names.extend([
                foot_name + "_airtime",
            ])
        space_low.extend([-np.inf] * (4))
        space_high.extend([np.inf] * (4))

        # General observations
        observation_names.extend([
            "trunk_x_velocity", "trunk_y_velocity", "trunk_z_velocity",
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
        ])
        space_low.extend([-np.inf] * 6)
        space_high.extend([np.inf] * 6)

        observation_names.extend(["goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity"])
        space_low.extend([-np.inf] * 3)
        space_high.extend([np.inf] * 3)

        observation_names.extend(["projected_gravity_x", "projected_gravity_y", "projected_gravity_z"])
        space_low.extend([-np.inf] * 3)
        space_high.extend([np.inf] * 3)

        observation_names.append("height_0")
        space_low.append(-np.inf)
        space_high.append(np.inf)


        observation_names.append("mass")
        space_low.append(-np.inf)
        space_high.append(np.inf)

        # # General robot context
        # observation_names.extend(["kp_gain", "kd_gain", "action_scaling_factor"])
        # space_low.extend([-np.inf] * 3)
        # space_high.extend([np.inf] * 3)


        observation_names.extend(["friction",
                                "stiffness",
                                "damping"])

        space_low.extend([-np.inf] * 3)
        space_high.extend([np.inf] * 3)


        space_low = np.array(space_low, dtype=np.float64)
        space_high = np.array(space_high, dtype=np.float64)
        name_to_idx = {name: idx for idx, name in enumerate(observation_names)}

        return gym.spaces.Box(low=space_low, high=space_high, shape=space_low.shape, dtype=np.float64), name_to_idx

    def get_initial_observation(self):

        # joint_observations = []
        # for i, joint_name in enumerate(self.joint_names):
        #     joint_observations.append(
        #         self.data.qpos[i + 7] - self.nominal_joint_positions[i])
        #     joint_observations.append(self.data.qvel[i + 6])
        #     joint_observations.append(self.last_action[i])
        qpos =self.data.qpos[7:] - self.nominal_joint_positions
        qvel = self.data.qvel[6:]
        last_actions = self.last_action.copy()
        joint_observations = np.concatenate((qpos, qvel, last_actions), dtype=np.float64)

        foot_contacts = np.array(
            [self.check_collision("floor", foot_name) for foot_name in self.foot_names])

        foot_airtime = self.airtime.feet_air_time.copy()
        foot_observations = np.concatenate((foot_contacts, foot_airtime), dtype=np.float64)

        # General observations
        trunk_linear_velocity = self.orientation_quat_inv.apply(self.data.qvel[:3])
        trunk_angular_velocity = self.data.qvel[3:6]
        goal_velocity = np.array([self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity])
        projected_gravity_vector = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        height = np.array([(self.data.qpos[2]) / self.nominal_height])


        mass = (self.default_mass / (170.0 / 2)) - 1.0

        friction = np.array([self.friction])
        stiffness = np.array([self.stiffness])
        damping = np.array([self.damping])

        observation = np.concatenate([
            joint_observations,
            foot_observations,
            trunk_linear_velocity,
            trunk_angular_velocity,
            goal_velocity,
            projected_gravity_vector,
            height,
            mass,
            friction,
            stiffness,
            damping

        ])

        return observation

    def get_robot_observation(self):

        obs = self.get_observation()
        idxs = np.concat((self.joint_positions_idx,
                          self.joint_velocities_idx,
                          self.projected_gravity_idx,
                          self.joint_previous_actions_idx,
                          self.trunk_linear_vel_idx,
                          self.trunk_angular_vel_idx,
                          self.goal_velocity_idx))

        observation = obs[idxs]
        return observation



        # noise_joint_angles = np.random.uniform(low=-0.03, high=0.03, size=12)
        # noise_joint_vel =  np.random.uniform(low=-1.5, high=1.5, size=12)
        # vel_noise =  np.random.uniform(low=-0.1, high=0.1, size=3)
        # ang_vel_noise =  np.random.uniform(low=-0.2, high=0.2, size=3)
        # gravity_noise =  np.random.uniform(low=-0.05, high=0.05, size=3)
        #
        # qpos = (self.data.qpos[7:].copy() + noise_joint_angles) - self.nominal_joint_positions #(12)
        # qvel = self.data.qvel[6:] + noise_joint_vel  #(12)
        # previous_action = self.current_action   #(12)
        # robot_linear_vel = self.orientation_quat_inv.apply(self.data.qvel[:3]).copy()  + vel_noise#(3)
        # robot_angular_vel = self.data.qvel[3:6].copy() + ang_vel_noise #(3)
        # commands = np.array([self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity]) #(3)
        # projected_gravity = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0])).copy() + gravity_noise #(3)
        #
        #
        # # Normalize and clip
        # qpos /= 4.6
        # qvel /= 35.0
        # previous_action /= 10.0
        #
        # robot_linear_vel = np.clip(robot_linear_vel / 10.0, -1.0, 1.0)
        # robot_angular_vel= np.clip(robot_angular_vel / 50.0, -1.0, 1.0)
        #
        #
        #
        #
        #
        # return observation
        #

    def get_observation(self):

        observation = self.initial_observation.copy()

        observation[self.joint_positions_idx] = self.data.qpos[
                                                           7:] - self.nominal_joint_positions.copy()
        observation[self.joint_velocities_idx] = self.data.qvel[6:]

        observation[self.joint_previous_actions_idx] = self.last_action

        observation[self.foot_ground_contact_idx] = np.array(
            [self.check_collision("floor", foot_name) for foot_name in self.foot_names])

        observation[self.foot_airtime_idx] = self.airtime.feet_air_time

        observation[self.trunk_linear_vel_idx] = self.orientation_quat_inv.apply(self.data.qvel[:3])

        observation[self.trunk_angular_vel_idx] = self.data.qvel[3:6]

        observation[self.goal_velocity_idx] = np.array(
            [self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity])
        # observation[self.goal_velocity_idx] = np.array([1.,0,2.])

        # print(observation[:48])
        # print(self.goal_velocity_idx)
        observation[self.projected_gravity_idx] = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))

        observation[self.height_idx] = np.array([self.data.qpos[2]])

        mass = self.model.body("trunk").mass
        min_mass =self.default_mass - 1
        max_mass =self.default_mass + 1
        observation[self.mass_idx] = 2*(mass - min_mass)/(max_mass - min_mass) -1

        # observation[self.mass_idx] =  (mass / (170.0 / 2)) - 1.0
        # gains_and_action_scaling_factor = (self.controller.get_gains_and_scale_factor() / [100.0 / 2, 2.0 / 2, 0.8 / 2]) - 1.0
        # observation[self.kp_idx] = gains_and_action_scaling_factor[0]
        # observation[self.kd_idx] = gains_and_action_scaling_factor[1]
        # observation[self.scale_idx] = gains_and_action_scaling_factor[2]

        #
        # exit()
        # Normalize and clip
        observation[self.joint_positions_idx] /= 4.6
        observation[self.joint_velocities_idx] /= 35.0
        observation[self.joint_previous_actions_idx] /= 10.0
        observation[self.foot_ground_contact_idx] = (observation[self.foot_ground_contact_idx] / 0.5) - 1.0
        observation[self.foot_airtime_idx] = np.clip((observation[self.foot_airtime_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.trunk_linear_vel_idx] = np.clip(
            observation[self.trunk_linear_vel_idx] / 10.0, -1.0, 1.0)
        observation[self.trunk_angular_vel_idx] = np.clip(
            observation[self.trunk_angular_vel_idx] / 50.0, -1.0, 1.0)
        observation[self.height_idx] = np.clip(
            (observation[self.height_idx] / (2 * self.nominal_height / 2)) - 1.0, -1.0, 1.0)

        # tangential_friction = self.friction_tangential_min + (
        #             self.friction_tangential_max - self.friction_tangential_min)
        # damping = self.damping_min + (self.damping_max - self.damping_min)
        # stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min)

        observation[self.friction_idx] = 2*(self.friction - self.friction_tangential_min)/ (self.friction_tangential_max -self.friction_tangential_min) -1
        observation[self.stiffness_idx] =2*(self.stiffness -self.stiffness_min)/(self.stiffness_max - self.stiffness_min) -1
        observation[self.damping_idx] = 2*(self.damping -self.damping_min)/(self.damping_max - self.damping_min) -1


        return observation

    def get_reward(self, done, first_contact, air_time):
        info = {"t": self.episode_step}
        return self.reward_function.reward_and_info(info, done, air_time, first_contact)

    def close(self):
        if self.viewer:
            self.viewer.stop()
            self.viewer=None

    def check_collision(self, groups1, groups2):
        if isinstance(groups1, list):
            ids1 = [self.collision_groups[group] for group in groups1]
            ids1 = set().union(*ids1)
        else:
            ids1 = self.collision_groups[groups1]

        if isinstance(groups2, list):
            ids2 = [self.collision_groups[group] for group in groups2]
            ids2 = set().union(*ids2)
        else:
            ids2 = self.collision_groups[groups2]

        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True

        return False

    def check_any_collision(self, groups):
        if isinstance(groups, list):
            ids = [self.collision_groups[group] for group in groups]
            ids = set().union(*ids)
        else:
            ids = self.collision_groups[groups]

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                return True

        return False

    def check_any_collision_for_all(self, groups):
        ids = [self.collision_groups[group] for group in groups]
        ids = set().union(*ids)

        any_collision = {idx: False for idx in ids}

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids:
                any_collision[con.geom1] = True
                ids.remove(con.geom1)
            if con.geom2 in ids:
                any_collision[con.geom2] = True
                ids.remove(con.geom2)

        return any_collision