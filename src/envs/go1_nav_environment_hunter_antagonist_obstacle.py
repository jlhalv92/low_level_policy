from numpy.ma.core import arctan2

from src.envs.mujoco_env import MuJoCo, ObservationType, mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.utils.height_map import *
import json  # Function to read the JSON file
import pygame
from collections import deque

class PDController:
    def __init__(self, env, control_frequency_hz=50, p_gain=20, d_gain=0.5, scaling_factor=0.25):
        self.env = env
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

    def action2torques(self, action):
        current_qpos = self.env._data.qpos[7:19].copy()
        current_qvel = self.env._data.qvel[6:18].copy()
        scaled_action = action * self.action_scale
        if self.apply_scale_shoulder:
            scaled_action[[0,3,6,9]] *= 0.5

        target_joint_positions = self.env.default_dof_pos.copy() + scaled_action
        torques = self.kp * self.kp_noise_factor * (
                    target_joint_positions - current_qpos + self.joint_position_offset) - self.kd * self.kd_noise_factor * current_qvel

        return torques * self.motor_strength_factor

class HistoryStore:
    def __init__(self, max_size=5):
        # Initialize with a maximum size and a deque to store the history
        self.max_size = max_size
        self.history = deque(maxlen=max_size)

    def reset_history(self, n_rays=10, max_length_ray=20):
        self.history = deque(maxlen=self.max_size)
        array = np.log2(max_length_ray) * np.ones(n_rays)

        for _ in range(self.max_size):
            self.add_sample(array)

    def add_sample(self, sample):
        # Add a new sample to the history
        self.history.append(sample)

    def get_history(self):
        # Return the current history as a list
        return np.array(list(self.history)).flatten()


def get_bounding_box_vertices(height, width, length, center_x, center_y, center_z, theta=0):
    """
    Calculate the vertices of a cube given its dimensions, center position, and yaw rotation.

    Parameters:
        height (float): Height of the cube.
        width (float): Width of the cube.
        length (float): Length of the cube.
        center_x (float): X-coordinate of the cube's center.
        center_y (float): Y-coordinate of the cube's center.
        theta (float): Yaw angle (rotation around z-axis) in radians.

    Returns:
        np.ndarray: A (8, 3) array containing the coordinates of the vertices.
    """
    # Half dimensions
    half_h = height / 2
    half_w = width / 2
    half_l = length / 2

    # Define relative positions of vertices from the cube's center
    relative_vertices = np.array([
        [-half_l, -half_w, -half_h],
        [-half_l, half_w, -half_h],
        [half_l, half_w, -half_h],
        [half_l, -half_w, -half_h],
        [-half_l, -half_w, half_h],
        [-half_l, half_w, half_h],
        [half_l, half_w, half_h],
        [half_l, -half_w, half_h],
    ])

    # Define the rotation matrix for yaw (rotation around z-axis)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Rotate vertices
    rotated_vertices = np.dot(relative_vertices, rotation_matrix.T)

    # Center the cube in space
    center = np.array([center_x, center_y, center_z])
    vertices = rotated_vertices + center

    return vertices


class StepProbabilitySampling:
    def __init__(self, env, probability=0.005):
        self.env = env
        self.probability = probability

    def setup(self):
        return False

    def step(self):
        return self.env.np_rng.uniform() < self.probability


class DefaultDomainPerturbation:
    def __init__(self, env,
                 push_velocity_x_min=-1.0, push_velocity_x_max=1.0,
                 push_velocity_y_min=-1.0, push_velocity_y_max=1.0,
                 push_velocity_z_min=-1.0, push_velocity_z_max=1.0,
                 set_chance=0.5, additive_multiplier=1):
        self.env = env
        self.push_velocity_x_min = push_velocity_x_min
        self.push_velocity_x_max = push_velocity_x_max
        self.push_velocity_y_min = push_velocity_y_min
        self.push_velocity_y_max = push_velocity_y_max
        self.push_velocity_z_min = push_velocity_z_min
        self.push_velocity_z_max = push_velocity_z_max
        self.set_chance = set_chance
        self.additive_multiplier = additive_multiplier
        self.sampled_push_velocity_x = 0.0
        self.sampled_push_velocity_y = 0.0
        self.sampled_push_velocity_z = 0.0

    def sample(self):
        self.sampled_push_velocity_x = self.env.np_rng.uniform(self.push_velocity_x_min, self.push_velocity_x_max)
        self.sampled_push_velocity_y = self.env.np_rng.uniform(self.push_velocity_y_min, self.push_velocity_y_max)
        self.sampled_push_velocity_z = self.env.np_rng.uniform(self.push_velocity_z_min, self.push_velocity_z_max)

        if self.env.np_rng.uniform() < self.set_chance:

            self.env._data.qvel[0:3] = np.array(
                [self.sampled_push_velocity_x, self.sampled_push_velocity_y, self.sampled_push_velocity_z])

        else:

            self.env._data.qvel[0:3] += np.array([self.sampled_push_velocity_x,
                                                  self.sampled_push_velocity_y,
                                                  self.sampled_push_velocity_z]) * self.additive_multiplier


def read_control_values(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def keyboard_control(vx, vy, wz):
    pygame.event.pump()
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         running = False
    #         print("running off")

    keys = pygame.key.get_pressed()


    if keys[pygame.K_UP]==False and keys[pygame.K_DOWN]==False:
        vx = 0
    if keys[pygame.K_a]==False and keys[pygame.K_d]==False:
        vy = 0
    if keys[pygame.K_LEFT]==False and keys[pygame.K_RIGHT]==False:

        wz = 0

    if keys[pygame.K_UP]==True:
        vx = min(vx + 0.5, 1.5)

    if keys[pygame.K_DOWN]:
        vx = max(vx - 0.5,-1.5)

    if keys[pygame.K_a]:
        vy = min(vy + 0.5, 1.5)
    if keys[pygame.K_d]:
        vy = max(vy - 0.5,-1.5)
    if keys[pygame.K_LEFT]:
        wz = min(wz + 0.5,1.5)

    if keys[pygame.K_RIGHT]:
        wz = max(wz - 0.5,-1.5)

    return vx,vy,wz

task_map = dict(nominal=0,
                navigation=1,
                obstacles_navigation=2,
                multi_obstacles_navigation=3,
                maze_evaluation=4)

TARGET_IDS = ["target{}".format(i) for i in range(1, 5)]
TARGET_IDS = ["target"]


def create_evenly_distributed_grid():
    # Provided points
    measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
    measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
    points = np.zeros((132, 3))
    # Generate z-coordinates
    X, Y = np.meshgrid(measured_points_x, measured_points_y)
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()

    return points


class Go1(MuJoCo):
    """
    Mujoco GO1

    """

    def __init__(self, horizon, gamma, target_pose=None, task_name="nominal", timestep=0.002, version=0, cfg=None, **kwargs):
        """
        Constructor.

        """

        if "max_antagonist_v" in cfg.keys():
            self.max_antagonist_v = cfg["max_antagonist_v"]
        else:
            self.max_antagonist_v = 1

        self.activate_arrow = cfg["activate_arrow"]
        self.controller_pd = PDController(self)
        self.narrow_passage = False
        self.no_walls = False
        self._obstacle_l = 0
        self._obstacle_w = 0
        self.pass_finish_line = False
        self.obstacle_robot_collision = False
        self.behaviour_cloning = False
        self.number_obstacles = 10
        self.rays_history = HistoryStore()
        self.robot_collisions = None
        self.not_hit_obstacle = None
        self._all_obstacles_move = False
        self.static_obstacles = cfg["static_obstacles"]
        self.obstacle_id = 0
        self._room_width = 10
        self.joystick = cfg["joystick"]
        if self.joystick:
            pygame.init()
            screen_width, screen_height = 300, 300
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Keyboard Joystick Simulation")

        self.qpos_offset = np.array([0.1, 0.5, 0.5] * 4)
        self.level = 0
        self.antagonist = cfg["antagonist"]
        self.obstacle_velocity = np.zeros(2)

        # self.occupancy_grid = OccupancyGrid(15,15,0.01)

        self.default_dof_pos = np.array([
            -0.1, -0.8, 1.5,
            0.1, 0.8, -1.5,
            0.1, -1.0, 1.5,
            -0.1, 1.0, -1.5,
        ])

        self.total_timesteps = 0

        self.target_vx = 0
        self.target_vy = 0
        self.target_yaw = 0
        self.cfg = cfg

        task = task_map[task_name]
        self.task = task
        self.horizon = horizon

        self.train = False
        self.success = False
        self.terminates = False

        kp = 20
        kd = 0.5
        timestep = timestep



        if task == 2:

                self.xml_path = "src/envs/unitree_go1/obstacle_h_navigation_go1_rays.xml"



        self.joint_limits = np.array([[-0.802851, 0.802851], [-1.0472, 4.18879], [-2.69653, -0.916298]] * 4)

        action_spec = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh",
                       "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]

        self.torques_limits = np.array(4 * [23.7, 23.7, 35.55])
        self.target_pose = target_pose
        self._reset_state = None
        self._ts = 0
        self.air_time_max = 0.5

        propioception = [
            ("FR_hip_pos", "FR_hip_joint", ObservationType.JOINT_POS),
            ("FR_thigh_pos", "FR_thigh_joint", ObservationType.JOINT_POS),
            ("FR_calf_pos", "FR_calf_joint", ObservationType.JOINT_POS),
            ("FL_hip_pos", "FL_hip_joint", ObservationType.JOINT_POS),
            ("FL_thigh_pos", "FL_thigh_joint", ObservationType.JOINT_POS),
            ("FL_calf_pos", "FL_calf_joint", ObservationType.JOINT_POS),
            ("RR_hip_pos", "RR_hip_joint", ObservationType.JOINT_POS),
            ("RR_thigh_pos", "RR_thigh_joint", ObservationType.JOINT_POS),
            ("RR_calf_pos", "RR_calf_joint", ObservationType.JOINT_POS),
            ("RL_hip_pos", "RL_hip_joint", ObservationType.JOINT_POS),
            ("RL_thigh_pos", "RL_thigh_joint", ObservationType.JOINT_POS),
            ("RL_calf_pos", "RL_calf_joint", ObservationType.JOINT_POS),

            ("FR_hip_vel", "FR_hip_joint", ObservationType.JOINT_VEL),
            ("FR_thigh_vel", "FR_thigh_joint", ObservationType.JOINT_VEL),
            ("FR_calf_vel", "FR_calf_joint", ObservationType.JOINT_VEL),
            ("FL_hip_vel", "FL_hip_joint", ObservationType.JOINT_VEL),
            ("FL_thigh_vel", "FL_thigh_joint", ObservationType.JOINT_VEL),
            ("FL_calf_vel", "FL_calf_joint", ObservationType.JOINT_VEL),
            ("RR_hip_vel", "RR_hip_joint", ObservationType.JOINT_VEL),
            ("RR_thigh_vel", "RR_thigh_joint", ObservationType.JOINT_VEL),
            ("RR_calf_vel", "RR_calf_joint", ObservationType.JOINT_VEL),
            ("RL_hip_vel", "RL_hip_joint", ObservationType.JOINT_VEL),
            ("RL_thigh_vel", "RL_thigh_joint", ObservationType.JOINT_VEL),
            ("RL_calf_vel", "RL_calf_joint", ObservationType.JOINT_VEL)]

        last_actions = [("last_actions", "torques", ObservationType.ACTIONS)]
        gravity_projection = [("gravity_projection", "orientation", ObservationType.IMU)]
        angular_velocity = [("angular_velocity", "a_velocity", ObservationType.A_VELOCITY)]
        linear_velocity = [("linear_velocity", "l_velocity", ObservationType.L_VELOCITY)]
        contact = [("feet_contact", "feet_sensor", ObservationType.FEET_SENSOR)]
        airtime = [("feet_airtime", "feet_sensor", ObservationType.FEET_SENSOR)]
        commands = [("commands", "command", ObservationType.COMMANDS)]
        height = [("height", "height_sensor", ObservationType.HEIGHT)]

        if self.task == 0:
            observation_spec = (propioception +
                                last_actions +
                                gravity_projection +
                                angular_velocity +
                                linear_velocity +
                                contact +
                                airtime +
                                height +
                                commands)
        elif self.task == 1:
            target_pos = [("target_pose", "target", ObservationType.POS)]
            robot_pos = [("robot_pose", "trunk", ObservationType.POS)]
            robot_orientation = [("orientation", "orientation_sensor", ObservationType.ORIENTATION)]
            current_velocity = [("current_velocity", "a_velocity", ObservationType.L_VELOCITY)]
            angular_velocity = [("angular_velocity", "a_velocity", ObservationType.L_VELOCITY)]


            observation_spec = (current_velocity +  # (3,)
                                angular_velocity +  # (3,)
                                robot_orientation +  # (3,)
                                robot_pos +  # (2,)
                                target_pos)  # (50,)

        elif self.task == 2:
            target_pos = [("target_pose", "target", ObservationType.POS)]
            rays = [("distance_rays", "distance_rays", ObservationType.RAYS)]
            robot_pos = [("robot_pose", "trunk", ObservationType.POS)]
            robot_orientation = [("orientation", "orientation_sensor", ObservationType.ORIENTATION)]
            current_velocity = [("current_velocity", "a_velocity", ObservationType.L_VELOCITY)]
            angular_velocity = [("angular_velocity", "a_velocity", ObservationType.L_VELOCITY)]
            room_width = [("room_width", "room_width", ObservationType.ROOM_WIDTH)]
            # robot_bounding_box = [("robot_bounding_box", "robot_bounding_box", ObservationType.ROBOT_BOUNDING_BOX)]
            # obstacle_bounding_box = [("obstacle_bounding_box", "obstacle_bounding_box", ObservationType.OBSTACLE_BOUNDING_BOX)]
            position_obstacle = [("obstacle_pose", "obstacle_1", ObservationType.BODY_POS)]
            velocity_obstacle = [("obstacle_velocity", "obstacle_1", ObservationType.VELOCITY)]

            observation_spec = (current_velocity +  # (3,)
                                angular_velocity + #(3,)
                                robot_orientation +  # (3,)
                                robot_pos +  # (2,)
                               #  position_obstacle +  # (2,)
                               # velocity_obstacle +  # (2,)
                                rays)  # (50,)

        elif self.task == 3:
            rays = [("distance_rays_small", "distance_rays_small", ObservationType.RAYS_SMALL)]
            robot_orientation = [("orientation", "orientation_sensor", ObservationType.HEADING)]
            current_velocity = [("current_velocity", "a_velocity", ObservationType.VELOCITY)]

            observation_spec = (propioception +
                                last_actions +
                                gravity_projection +
                                angular_velocity +
                                linear_velocity +
                                contact +
                                airtime +
                                height +
                                commands +
                                current_velocity +  # (3,)
                                robot_orientation +  # (1,)
                                rays)  # (2,)

        collision_groups = [("floor", ["floor"]),
                            ("feet", ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
                            ("FL_foot", ["FL_foot"]), ("FR_foot", ["FR_foot"]),
                            ("RL_foot", ["RL_foot"]), ("RR_foot", ["RR_foot"]),
                            ("FR_calf1", ["FR_calf1"]),
                            ("FL_calf1", ["FL_calf1"]),
                            ("RR_calf1", ["RR_calf1"]),
                            ("RL_calf1", ["RL_calf1"]),
                            ("legs", ["FR_thigh1", "FL_thigh1", "RR_thigh1", "RL_thigh1",
                                      "FR_calf1", "FL_calf1", "RR_calf1", "RL_calf1",
                                      "FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
                            ("trunk", ["trunk_1", "trunk_2", "trunk_3",
                                       "trunk_4", "trunk_5", "trunk_6",
                                       "trunk_7", "trunk_8"]),
                            ("trunk_t", ["trunk_1", "trunk_2", "trunk_3"]),
                            ("trunk_1", ["trunk_1"]),
                            ("trunk_2", ["trunk_2"]),
                            ("trunk_3", ["trunk_3"]),
                            ("trunk_4", ["trunk_4"]),
                            ("trunk_5", ["trunk_5"]),
                            ("trunk_6", ["trunk_6"]),
                            ("trunk_7", ["trunk_7"]),
                            ("trunk_8", ["trunk_8"]),
                            ("robot", ["FR_thigh1", "FL_thigh1", "RR_thigh1", "RL_thigh1",
                                       "FR_calf1", "FL_calf1", "RR_calf1", "RL_calf1",
                                       "FR_foot", "FL_foot", "RR_foot", "RL_foot",
                                       "trunk_1", "trunk_2", "trunk_3",
                                       "trunk_4", "trunk_5", "trunk_6",
                                       "trunk_7", "trunk_8"])]
        if self.task == 2:
            # collision_walls = ("walls",["wall_1", "wall_2","wall_3", "wall_4"])
            collision_walls = ("walls", ["wall_3", "wall_4"])

            collision_obstacle = ("obstacles", ["obstacle_1"])
            collision_groups.extend([collision_walls, collision_obstacle])
        if self.task == 3:
            collision_walls = ("walls", ["wall_3", "wall_4"])
            collision_walls = ("walls", ["wall_3", "wall_4"])

            collision_groups.extend([collision_walls])

        self.collision_groups_list = collision_groups

        if self.task == 0:
            max_a = self.qpos_offset
            min_a = - self.qpos_offset

        else:
            max_a = 1 * np.array([1.5, 1, 1.5])
            min_a = 1 * np.array([-1.5, -1, -1.5])

        super().__init__(self.xml_path, action_spec,
                         observation_spec,
                         gamma, horizon,
                         collision_groups=collision_groups,
                         n_intermediate_steps=1,
                         n_substeps=4,
                         timestep=timestep,
                         max_a=max_a, min_a=min_a)

        # self.renderer = mujoco.Renderer(self._model)
        self.last_action = np.array(12 * [0])
        self.torques = list()
        x, y = self._data.qpos[:2]
        self.p_gains = np.array(12 * [kp])  # 60
        self.d_gains = np.array(12 * [kd])  # 6
        self.foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self._max_torques = np.array([23.7, 23.7, 35.55])
        self._min_torques = np.array([-23.7, -23.7, -35.55])

        self.hip_indices = [0, 3, 6, 9]

        self.sitting_dof_pos = np.array([0.0 / 180 * np.pi, 70.0 / 180 * np.pi, -150.0 / 180 * np.pi] * 4)
        # self.target_yaw =0

        self.yaw = None


        self.default_trunk_mass = self._model.body_mass[1]
        self.default_trunk_inertia = self._model.body_inertia[1]



        self.use_mujoco_renderer = True
        self.mujoco_options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.mujoco_options)
        self.mujoco_options.sitegroup[0] = False
        self.mujoco_options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.renderer_main_camera = None
        self.renderer_depth = None

    def get_info(self):
        if self.task == 2:
            return dict(succes=self.success)
        else:
            return dict()


    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array).

        """

        grf = [self._get_collision_force("floor", "foot_FL")[:3],
               self._get_collision_force("floor", "foot_FR")[:3],
               self._get_collision_force("floor", "foot_RL")[:3],
               self._get_collision_force("floor", "foot_RR")[:3]]

        return grf[0], grf[1], grf[2], grf[3]

    def update_orientation_attributes(self):
        self.orientation_quat = R.from_quat(
            [self._data.qpos[4], self._data.qpos[5], self._data.qpos[6], self._data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()

    def _modify_observation(self, obs):


        linear_velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])
        vx = linear_velocity[0]
        vy = linear_velocity[1]
        wz = self._data.qvel[5]
        current_velocity = np.array([vx, vy, wz])
        robot_orientation = self.orientation_euler.copy() # (1,)
        obs = self.obs_helper.modify_obs(obs, "current_velocity", current_velocity)
        obs = self.obs_helper.modify_obs(obs, "orientation", robot_orientation)
        angular_velocity = self._data.qvel[3:6]
        obs = self.obs_helper.modify_obs(obs, "angular_velocity", angular_velocity)
        # pos_diff = self._data.body("obstacle_1").xpos[:3] - self._data.body("trunk").xpos[:3]
        # relative_obstacle_pos = self.orientation_quat_inv.apply(pos_diff)
        # vel_diff = self._data.qvel[:3] - self._data.joint("obstacle").qvel[:3]
        # relative_obstacle_vel = self.orientation_quat_inv.apply(vel_diff)
        #
        # obs = self.obs_helper.modify_obs(obs, "obstacle_pose", relative_obstacle_pos[:2])
        # obs = self.obs_helper.modify_obs(obs, "obstacle_velocity", relative_obstacle_vel[:2])




        return obs

    def _compute_action(self, cur_obs, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            action (np.ndarray): numpy array with the actions
                provided at every step.

        Returns:
            The action to be set in the actual mujoco simulation.

        """

        torques = self.controller_pd.action2torques(action)

        return torques



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

        for coni in range(0, self._data.ncon):
            con = self._data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True

        return False

    def update_velocity_commands(self,v):
        self.target_vx = v[0]
        self.target_vy = v[1]
        self.target_wz = v[2]

    def _step_init(self, obs, action):

        if self.activate_arrow:
            module = np.sqrt(self.target_vx**2 + self.target_vy**2)
            angle = arctan2(self.target_vy, self.target_vx)
            angle_rate = self.target_yaw*self.dt
            scaled_module = module*0.1
            yaw = angle + angle_rate
            euler_angles = [0, np.pi/2, yaw]
            quaternion = R.from_euler('xyz', euler_angles).as_quat()
            quat = quaternion[[3,0,1,2]].copy()
            module_arrow = scaled_module + 0.01
            self._model.site("velocity_arrow").pos = [module_arrow*np.cos(yaw), module_arrow*np.sin(yaw), 0.1]
            self._model.site("velocity_arrow").size[1] = scaled_module + 0.01
            self._model.site("velocity_arrow").quat = quat

        # self._data.site("velocity_arrow").euler[2] = angle + angle_rate

        if self.antagonist:

            self._data.actuator("thrust1").ctrl =np.clip(self.obstacle_velocity[0], -self.max_antagonist_v,0)
            self._data.actuator("thrust2").ctrl =np.clip(self.obstacle_velocity[1],-self.max_antagonist_v,self.max_antagonist_v)
            self._data.actuator("rotation_z").ctrl = 0

        else:
            pos_robot_x = self._data.qpos[0]
            pos_obstacle_x = self._data.body("obstacle_1").xpos[0]
            pos_robot_y = self._data.qpos[1]
            pos_obstacle_y = self._data.body("obstacle_1").xpos[1]
            module_x = np.clip(2*(pos_robot_x - pos_obstacle_x), -self.max_antagonist_v,0)
            module_y = np.clip(2*(pos_robot_y - pos_obstacle_y),-self.max_antagonist_v,self.max_antagonist_v)

            self._data.actuator("thrust1").ctrl =module_x
            self._data.actuator("thrust2").ctrl =module_y
            self._data.actuator("rotation_z").ctrl = 0

    def check_any_collision(self, groups):
        if isinstance(groups, list):
            ids = [self.collision_groups[group] for group in groups]
            ids = set().union(*ids)
        else:
            ids = self.collision_groups[groups]

        for con_i in range(0, self._data.ncon):
            con = self._data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                return True

        return False

    def check_any_collision_for_all(self, groups):
        ids = [self.collision_groups[group] for group in groups]
        ids = set().union(*ids)

        any_collision = {idx: False for idx in ids}

        for con_i in range(0, self._data.ncon):
            con = self._data.contact[con_i]
            if con.geom1 in ids:
                any_collision[con.geom1] = True
                ids.remove(con.geom1)
            if con.geom2 in ids:
                any_collision[con.geom2] = True
                ids.remove(con.geom2)

        return any_collision


    def calculate_velocity_distance_reward(self, current_velocity, current_pos, target_pos):
        vect = target_pos - current_pos
        norm = np.linalg.norm(vect)
        vect_norm = vect / (norm + 1e-5)
        projected_vel = np.dot(vect_norm, current_velocity)

        r = np.min([projected_vel, 1]) / 1
        return r

    def reward_task_1(self, moving_coef=1, angle_coef=0.5):
        current_global_linear_velocity = self._data.qvel[:3]
        current_local_linear_velocity = self.orientation_quat_inv.apply(current_global_linear_velocity)
        current_velocity = current_local_linear_velocity[:2]
        current_pos = self._data.body("trunk").xpos[:2]
        target_pos = self._data.geom("target").xpos[:2]

        vect_direction = target_pos - current_pos
        norm = np.linalg.norm(vect_direction)
        target_vec_norm = vect_direction / (norm + 1e-5)
        target_yaw = np.arctan2(target_vec_norm[1], target_vec_norm[0])

        yaw = self.orientation_euler[2]

        reward_moving = self.calculate_velocity_distance_reward(current_velocity, current_pos, target_pos)

        reward = moving_coef * reward_moving + angle_coef * np.exp(-np.abs(target_yaw - yaw))
        return reward

    def proximity_penalty(self):

        distance_to_obstacle = np.linalg.norm(
            np.array(self._model.site("obstacle_1").pos[:2]) - self._data.body("trunk").xpos[:2])
        obstacle_collision_radius = 1
        obstacle_save_radius = 0.8
        robot_collision_bubble_radius = 0.4

        if distance_to_obstacle > obstacle_collision_radius + obstacle_save_radius + robot_collision_bubble_radius:
            penalty = 0
        elif obstacle_collision_radius + robot_collision_bubble_radius + obstacle_save_radius > distance_to_obstacle > obstacle_collision_radius + robot_collision_bubble_radius:

            penalty = -np.exp(2 * (obstacle_collision_radius + robot_collision_bubble_radius - distance_to_obstacle))

        else:
            penalty = -1
        return penalty

    def collision_penalty(self):
        legs_collision_walls = self._check_collision("legs", "walls")
        feet_collision_walls = self._check_collision("feet", "walls")
        trunk_collision_walls = self._check_collision("trunk", "walls")

        legs_collision_obstacles = self._check_collision("legs", "obstacles")
        trunk_collision_obstacles = self._check_collision("trunk", "obstacles")
        feet_collision_obstacles = self._check_collision("feet", "obstacles")

        collision_obstacle = (legs_collision_obstacles or trunk_collision_obstacles or feet_collision_obstacles)
        collision_wall = (legs_collision_walls or trunk_collision_walls or feet_collision_walls)
        penalty = 20 * (collision_obstacle) + 10 * (collision_wall)
        return penalty

    def reward_velocity(self, current_velocity, target_pos, current_pos):
        vect = target_pos - current_pos
        norm = np.linalg.norm(vect)
        vect_norm = vect / (norm + 1e-5)
        projected_vel = max(np.dot(vect_norm, current_velocity), 0)
        r = np.min([projected_vel, 1])
        return r

    def obs_antagonist(self):
        # pos_antagonist = self._data.body("obstacle_1").xpos[:2]
        pos_robot = self._data.body("obstacle_1").xpos[:2] -self._data.body("trunk").xpos[:2]
        velocity_robot = self._data.qvel[:2]
        velocity_obstacle = np.array([self._data.joint("obstacle").qvel[1]])
        distance = self._data.body("obstacle_1").xpos[0] -self._data.body("trunk").xpos[0]
        pass_finish_line = np.array([distance < -0.35])
                                # 2, 2, 1, 1
        return np.concatenate((pos_robot, velocity_robot, velocity_obstacle, pass_finish_line))

    def get_reward(self, action):


        if self.task == 1:
            return self.reward_task_1()

        if self.task == 2 or self.task == 3 or self.task == 4:
            yaw = self.orientation_euler[2]
            global_vel = self._data.qvel[:2].copy()
            y_pos = self._data.body("trunk").xpos[1]
            robot_pos = self._data.qpos[:2].copy()
            desired_yaw = 0  # Assuming 0 is the forward direction
            yaw_deviation_penalty = np.abs(yaw - desired_yaw)
            distance = self._data.site("target").xpos[0] - self._data.qpos[0]

            distance_to_obstacle = np.linalg.norm(self._data.site("obstacle_1").xpos[:2] - self._data.qpos[:2])

            soft_hit_condition = self._model.site("obstacle_1").size[0] + self._model.site("robot_contact_limit").size[
                0]

            proximity_penalty = (distance_to_obstacle < soft_hit_condition) * np.exp(1 / distance_to_obstacle)

            penalties = 1.*yaw_deviation_penalty + (np.abs(y_pos)) * (distance < -0.35) + proximity_penalty * (
                        not self.narrow_passage and not self.no_walls) + self.collision_penalty()
            y_bounders = 5*(np.abs(y_pos) >= self._room_width/2)

            target_vel = 1
            reward = 2.5 * max(min(target_vel, global_vel[0]), 0)

            return reward - penalties - y_bounders

    def reward(self, state, action, next_state, absorbing):

        reward = self.get_reward(action)


        return reward

    def update_variables(self, action, cur_obs):
        self.last_action = action
        self.previous_vel = self.orientation_quat_inv.apply(self._data.qvel[:3].copy())
        self.previous_angular_vel = self._data.qvel[5].copy()
    def get_student_rays(self):
        rf_readings = np.array([self._data.sensor("rf_l_{}".format(i + 1)).data.copy()[0] for i in range(10)])
        rf_readings = np.where(rf_readings == -1, 20, rf_readings)
        rf_readings = np.where(rf_readings == 0, rf_readings + 0.01, rf_readings)
        rf_readings_2 = np.where(rf_readings >= 20, 20, rf_readings)
        rf_readings_f = np.log2(rf_readings_2)
        obs = rf_readings_f
        return obs

    # def render(self, record=False, depth=False):
    #
    #     if self.use_mujoco_renderer:
    #
    #         self.renderer_main_camera.update_scene(self._data, "backview")
    #         img = self.renderer_main_camera.render()
    #
    #         if depth:
    #             self.renderer_depth.update_scene(self._data, "head", self.mujoco_options)
    #             depth_image = self.renderer_depth.render()
    #
    #             depth_image = np.clip(depth_image, 0, 5)
    #             min_val = depth_image.min()
    #             max_val = depth_image.max()
    #             norm_depth_image = 255 * ((depth_image - min_val) / (max_val - min_val))
    #             # # Shift nearest values to the origin.
    #             # depth_image -= depth_image.min()
    #             # # Scale by 2 mean distances of near rays.
    #             # depth_image /= 2 * depth_image[depth_image <= 1].mean()
    #             # # Scale to [0, 255]
    #             # depth_image = 255 * np.clip(depth_image, 0, 1)
    #
    #             return img, norm_depth_image
    #         else:
    #             return img

    def get_robot_observation(self):

        noise_joint_angles = np.random.uniform(low=-0.03, high=0.03, size=12)
        noise_joint_vel = np.random.uniform(low=-1.5, high=1.5, size=12)
        vel_noise = np.random.uniform(low=-0.1, high=0.1, size=3)
        ang_vel_noise = np.random.uniform(low=-0.2, high=0.2, size=3)
        gravity_noise = np.random.uniform(low=-0.05, high=0.05, size=3)

        qpos = (self._data.qpos[7:19].copy() + noise_joint_angles) - self.default_dof_pos  # (12)
        qvel = self._data.qvel[6:18] + noise_joint_vel  # (12)
        previous_action = self.last_action # (12)
        robot_linear_vel = self.orientation_quat_inv.apply(self._data.qvel[:3]).copy() + vel_noise  # (3)
        robot_angular_vel = self._data.qvel[3:6].copy() + ang_vel_noise  # (3)
        commands = np.array([self.target_vx, self.target_vy, self.target_yaw])  # (3)
        projected_gravity = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0])).copy() + gravity_noise  # (3)

        # Normalize and clip
        qpos /= 4.6
        qvel /= 35.0
        previous_action /= 10.0

        robot_linear_vel = np.clip(robot_linear_vel / 10.0, -1.0, 1.0)
        robot_angular_vel = np.clip(robot_angular_vel / 50.0, -1.0, 1.0)

        observation = np.concatenate([qpos,
                                      qvel,
                                      projected_gravity,
                                      previous_action,
                                      robot_linear_vel,
                                      robot_angular_vel,
                                      commands])

        return observation


    def _step_finalize(self):
        # if self.use_mujoco_renderer:
        #     self.renderer_main_camera = mujoco.Renderer(self._model, width=1024, height=720)
        #     self.renderer_depth = mujoco.Renderer(self._model, width=1024, height=720)
        #     self.renderer_main_camera.update_scene(self._data, "head")
        #     self.renderer_depth.update_scene(self._data, "backview", self.mujoco_options)

        # print(self._data.sensor("collision_distance").data)
        self._ts += 1
        self.update_orientation_attributes()
        y_pos = self._data.body("trunk").xpos[1]
        out_of_limit = 5 * (np.abs(y_pos) >= self._room_width / 2)

        # if self.task == 2:
        #     if not self.static_obstacles:
        #         if not self.antagonist:
        #             obstacle_pose = self._data.body("obstacle_1").xpos[1]
        #             robot_pose = self._data.body("trunk").xpos[1]
        #             self._data.ctrl[-3] = self.obstacle_pose_x
        #             self._data.ctrl[-2] = 0
        #             self._data.ctrl[-1] = np.clip((robot_pose - obstacle_pose), -1, 1)

        if self.task == 3 or self.task == 2:
            robot_pose = self._data.qpos[:2]
            target_pose = self._data.site("target").xpos[:2]
            norm_distance = target_pose[0] - robot_pose[0]
            self.pass_finish_line = norm_distance <= 0.1

            if self._ts >= self.horizon:
                self.success = self.robot_collisions == 0 and self.pass_finish_line and not out_of_limit

        if self.joystick:
            vx, vy, wz = keyboard_control(self.target_vx, self.target_vy, self.target_yaw)

            self.target_vx = vx
            self.target_vy = vy
            self.target_yaw = wz

        if self.train:
            self.total_timesteps += 1

        # gravity_z_proj = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))[2]
        height = self._data.qpos[2]
        fall = self.check_collision("floor", "trunk_t") or height <= 0.1

        # if self.task > 1:
        legs_collision_obstacles = self._check_collision("legs", "obstacles")
        trunk_collision_obstacles = self._check_collision("trunk", "obstacles")

        self.obstacle_robot_collision = legs_collision_obstacles + trunk_collision_obstacles
        # collision_robot_wall = self._check_collision("robot", "walls")
        collision_robot = legs_collision_obstacles or trunk_collision_obstacles


        if fall | collision_robot |out_of_limit :
            self.terminates = True

    def reward_antagonist(self):

        # Reward Antagonist
        # distance = self._data.site("target").xpos[0] - self._data.qpos[0]
        # pass_finish_line = distance < -0.35
        distance_x = self._data.body("obstacle_1").xpos[0] - self._data.qpos[0]
        distance_y = np.abs(self._data.body("obstacle_1").xpos[1] - self._data.qpos[1])
        pass_finish_line = distance_x < -0.2

        # pos_antagonist = self._data.body("obstacle_1").xpos[:2]
        #
        # pos_robot = self._data.body("trunk").xpos[:2]
        # u_v = (pos_robot - pos_antagonist) / (np.linalg.norm(pos_robot - pos_antagonist) + 1e-5)
        # current_vel = self._data.joint("obstacle").qvel[:2]

        reward_vel = -distance_y*(not pass_finish_line) -10*(pass_finish_line)

        # return reward_vel
        reward_collision = 10 * self.obstacle_robot_collision
        # reward_distance = -np.linalg.norm(pos_robot - pos_antagonist)
        # reward_velocity = 2*np.abs(self.mdp._data.joint("obstacle").qvel[1])*np.abs(pos_robot[1] - pos_antagonist[1])
        reward_antagonist = (reward_collision + reward_vel)

        return reward_antagonist

    # def reward_antagonist(self):
    #
    #     # Reward Antagonist
    #     distance = self._data.site("target").xpos[0] - self._data.qpos[0]
    #     pass_finish_line = distance < -0.35
    #
    #     pos_antagonist = self._data.body("obstacle_1").xpos[:2]
    #
    #     pos_robot = self._data.body("trunk").xpos[:2]
    #
    #     reward_collision = 10 * self.obstacle_robot_collision
    #     reward_distance = -np.abs(pos_robot[1] - pos_antagonist[1])
    #     # reward_velocity = 2*np.abs(self.mdp._data.joint("obstacle").qvel[1])*np.abs(pos_robot[1] - pos_antagonist[1])
    #     reward_antagonist = (reward_collision + reward_distance) * ( not pass_finish_line) +  -1*(pass_finish_line)
    #     return reward_antagonist

    def is_absorbing(self, state):

        if self.terminates == True:

            return True

        else:
            return False



    def get_info(self):
        return {"success": self.success, "collisions": self.robot_collisions}

    def setup(self, obs, cond):

        self.robot_collisions = 0

        self.torques = list()

        self.update_orientation_attributes()
        self.success = False
        self.terminates = False

        self._ts = 0
        self.episode_steps = 0
        self._data.qpos[7:19] = self.default_dof_pos.copy()
        self._data.qpos[0:3] = [0, 0, 0.36]
        self.reset_states(self.task)



        mujoco.mj_step(self._model, self._data)

        self.last_action = np.zeros(12)
        self.previous_vel = np.zeros(3)
        self.previous_angular_vel = 0
        self.last_velocity_command = np.array([self.target_vx, self.target_vy, self.target_yaw])



    def reset_states(self, task):
        self._model.site("target").pos[0] = 10


        if task == 2:

            spec = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/obstacle_h_navigation_go1_rays.xml")

            geoms = spec.worldbody.geoms
            walls = ["wall_3", "wall_4"]
            self._room_width = 3.0
            self._obstacle_l = 0.5
            self._obstacle_w  = 0.5
            # if self._room_width<=2:
            #     self._obstacle_l =np.random.uniform(low=0.05,  high= (self._room_width/2)*0.4)
            #     self._obstacle_w = np.random.uniform(low=0.05, high= (self._room_width/2)*0.4)
            # else:
            #     self._obstacle_l = np.random.uniform(low=0.2, high= (self._room_width/2)*0.4)
            #     self._obstacle_w = np.random.uniform(low=0.2, high= (self._room_width/2)*0.4)


            self.wall_pos = self._room_width / 2
            worldbody = spec.worldbody
            # z_wall = -2
            z_wall = 0.8
            worldbody.add_site(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[5, self._room_width/2, 0.05],
                pos=[3,0,0],
                rgba=[0,1,0,0.1]
            )
            worldbody.add_site(
                name="finish_line",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[1, self._room_width/2, 0.05],
                pos=[7,0,0],
                rgba=[1,0,0,0.1]
            )

            wall_pos = 0.5 + self._room_width/2

            for geom in geoms:
                if geom.name in walls:
                    if geom.pos[1] < 0:
                        geom.pos[1] = -(wall_pos) - geom.size[1] / 2
                    else:
                        geom.pos[1] = (wall_pos) + geom.size[1] / 2
                    geom.pos[2] = z_wall


            obstacle = spec.find_body("obstacle_1")
            obstacle_height = np.random.uniform(low=0.25, high=0.8)
            obstacle.first_geom().size = [self._obstacle_l, self._obstacle_w, obstacle_height]
            if self.activate_arrow:
                trunk = spec.find_body("trunk")
                trunk.add_site(name="velocity_arrow",
                type = mujoco.mjtGeom.mjGEOM_CAPSULE,
                size =  [0.01, 0.1, 0],
                euler =[0, np.pi/2, 0],
                pos = [0.1, 0, 0.1],
                rgba = [1, 0, 0, 1])
                trunk.add_site(name="velocity_base",
                type = mujoco.mjtGeom.mjGEOM_SPHERE,
                size =  [0.02, 0, 0],
                euler =[0, 0, 0],
                pos = [0., 0, 0.1],
                rgba = [1, 0, 0, 1])

            model = spec.compile()
            self._model = model

            self._model.site("obstacle_1").size[0] = 1.1 * np.sqrt((self._obstacle_l) ** 2 + (self._obstacle_w) ** 2)
            y_robot = np.random.uniform(-self.wall_pos + 0.3, self.wall_pos - 0.3)
            y_obstacle = y_robot + np.random.uniform(-0.3, 0.3)

            self._model.opt.timestep = 0.005
            self._data = mujoco.MjData(model)
            self._data.qpos[7:19] = self.default_dof_pos.copy()
            # x = np.random.uniform(-1.5, 1.5)
            x = 0
            self._data.qpos[0:3] = [x, y_robot, 0.36]
            self.obstacle_pose_x = x + np.random.uniform(2.5, 3.5)

            self._data.joint("obstacle").qpos[:3] = [self.obstacle_pose_x, y_obstacle, obstacle_height]
            self._model.site("target").pos[0] = self.obstacle_pose_x + 1
            self._model.site("target").size[1] = self.wall_pos
            self._model.site("target").pos[-1] = 0.1
            if self._viewer is not None:
                self._viewer.load_new_model(self._model)

            # if self.use_mujoco_renderer:
            #     self.renderer_main_camera = None
            #     self.renderer_depth = None
            #     self.renderer_main_camera = mujoco.Renderer(self._model, width=1024, height=720)
            #     self.renderer_depth = mujoco.Renderer(self._model, width=106, height=60)
            #     self.renderer_depth.enable_depth_rendering()

            # if self.task == 2:
            #     if not self.antagonist:
            #         if not self.static_obstacles:
            #             self._data.ctrl[-1] = np.random.uniform(-1, 1)
            #         else:
            #             self._data.ctrl[-1] = 0
            #
            #
            #     else:
            #         self._data.ctrl[-1] = self.obstacle_velocity
            #
            #     self._data.ctrl[-3] = self.obstacle_pose_x
            #     self._data.ctrl[-2] = 0


