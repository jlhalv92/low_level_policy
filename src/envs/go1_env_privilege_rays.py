from sympy.physics.units import length
from torch.utils.hipify.hipify_python import file_add_header

from src.envs.mujoco_env import MuJoCo, ObservationType, mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.envs.reward.reward import RudinOwnVarReward as reward_f
from src.utils.height_map  import *
import json# Function to read the JSON file
import pygame
from collections import deque

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
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1]
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

            self.env._data.qvel[0:3] = np.array([self.sampled_push_velocity_x, self.sampled_push_velocity_y, self.sampled_push_velocity_z])

        else:

            self.env._data.qvel[0:3] += np.array([self.sampled_push_velocity_x,
                                                  self.sampled_push_velocity_y,
                                                  self.sampled_push_velocity_z]) * self.additive_multiplier

def read_control_values(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)



def keyboard_control(vx, vy, wz):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("running off")
            return 0, 0, 0  # Stop motion on quit

    keys = pygame.key.get_pressed()

    vx = min(vx + 0.1, 1.5) if keys[pygame.K_UP] else max(vx - 0.1, -1.5) if keys[pygame.K_DOWN] else 0
    vy = min(vy + 0.1, 0.6) if keys[pygame.K_LEFT] else max(vy - 0.1, -0.6) if keys[pygame.K_RIGHT] else 0
    wz = min(wz + 0.1, 1.5) if keys[pygame.K_w] else max(wz - 0.1, -1.5) if keys[pygame.K_e] else 0

    return vx, vy, wz


task_map = dict(nominal=0,
                navigation=1,
                obstacles_navigation=2,
                multi_obstacles_navigation=3,
                maze_evaluation=4)

TARGET_IDS= ["target{}".format(i) for i in range(1 ,5)]
TARGET_IDS= ["target"]



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

    def __init__(self, horizon, gamma, target_pose=None, task_name="nominal",timestep=0.002, version=0,  cfg=None):
        """
        Constructor.

        """
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
        self.domain_randomization = cfg["domain_randomization"]
        self.static_obstacles = cfg["static_obstacles"]
        self.obstacle_id = 0
        self._room_width = 10
        self.curriculum = cfg["curriculum"]
        self.resample =  cfg["resample"]
        self.joystick =  cfg["joystick"]
        if self.joystick:
            pygame.init()
            screen_width, screen_height = 300, 300
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Keyboard Joystick Simulation")


        self.qpos_offset = np.array([0.1, 0.5, 0.5] * 4)
        self.level = 0
        self.antagonist = cfg["antagonist"]
        self.max_level = cfg["max_level"]
        self.obstacle_velocity = 0


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

        kp =cfg['kp']
        kd =cfg['kd']
        timestep = timestep

        if task==0:
            xml_path = "src/envs/unitree_go1/walk_go1.xml"
            self.xml_path = "src/envs/unitree_go1/navigation_go1_rays.xml"

        if task==1:
            self.xml_path = "src/envs/unitree_go1/navigation_walk_go1.xml"

        if task==2:
            if self.behaviour_cloning:
                self.xml_path = "src/envs/unitree_go1/navigation_go1_rays_behaviour_cloning.xml"


            else:
                self.xml_path = "src/envs/unitree_go1/navigation_go1_rays.xml"



        if task==3:
            self.xml_path = "src/envs/unitree_go1/navigation_go1_rays.xml"

        if task==4:
            self.xml_path = "src/envs/unitree_go1/obstacles_navigation_complex_go1_evaluation_v{}.xml".format(version)

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


        if self.task==0:
            observation_spec = (propioception +
                                last_actions +
                                gravity_projection +
                                angular_velocity +
                                linear_velocity +
                                contact +
                                airtime +
                                height +
                                commands)
        elif self.task==1:
            target_pos = [("target_pose", "target", ObservationType.SITE_POS)]
            robot_pos = [("robot_pose", "trunk", ObservationType.BODY_POS)]
            robot_orientation = [("orientation", "orientation_sensor", ObservationType.ORIENTATION)]
            distance_to_target = [("distance_target", "target", ObservationType.DISTANCE)]
            
            observation_spec = (propioception +
                                last_actions +
                                gravity_projection +
                                angular_velocity +
                                linear_velocity +
                                contact +
                                airtime +
                                height +
                                commands +
                                target_pos +
                                robot_pos +
                                robot_orientation +
                                distance_to_target)
        
        elif self.task==2:
            target_pos = [("target_pose", "target", ObservationType.POS)]
            rays = [("distance_rays", "distance_rays", ObservationType.RAYS)]
            robot_pos = [("robot_pose", "trunk", ObservationType.POS)]
            robot_orientation = [("orientation", "orientation_sensor", ObservationType.HEADING)]
            current_velocity = [("current_velocity", "a_velocity", ObservationType.L_VELOCITY)]
            room_width = [("room_width", "room_width", ObservationType.ROOM_WIDTH)]
            # robot_bounding_box = [("robot_bounding_box", "robot_bounding_box", ObservationType.ROBOT_BOUNDING_BOX)]
            # obstacle_bounding_box = [("obstacle_bounding_box", "obstacle_bounding_box", ObservationType.OBSTACLE_BOUNDING_BOX)]
            position_obstacle = [("obstacle_pose", "obstacle_1", ObservationType.BODY_POS)]
            velocity_obstacle = [("obstacle_velocity", "obstacle_1", ObservationType.VELOCITY)]


            observation_spec = (propioception +
                                last_actions +
                                gravity_projection +
                                angular_velocity +
                                linear_velocity +
                                contact +
                                airtime +
                                height +
                                commands +
                                current_velocity + # (3,)
                                robot_pos + #(2,)
                                robot_orientation + # (1,)
                                position_obstacle + #(2,)
                                velocity_obstacle + #(2,)
                                rays ) # (50,)

        elif self.task==3:
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
                                current_velocity + # (3,)
                                robot_orientation + # (1,)
                                rays ) # (2,)


        collision_groups = [("floor", ["floor"]),
                            ("feet", ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
                            ("FL_foot", ["FL_foot"]), ("FR_foot", ["FR_foot"]),
                            ("RL_foot", ["RL_foot"]), ("RR_foot", ["RR_foot"]),
                            ("FR_calf1", ["FR_calf1"]),
                            ("FL_calf1", ["FL_calf1"]),
                            ("RR_calf1", ["RR_calf1"]),
                            ("RL_calf1", ["RL_calf1"]),
                            ("legs", ["FR_thigh1","FL_thigh1","RR_thigh1","RL_thigh1",
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
                            ("robot", ["FR_thigh1","FL_thigh1","RR_thigh1","RL_thigh1",
                                      "FR_calf1", "FL_calf1", "RR_calf1", "RL_calf1",
                                      "FR_foot", "FL_foot", "RR_foot", "RL_foot",
                                       "trunk_1", "trunk_2", "trunk_3",
                                       "trunk_4", "trunk_5", "trunk_6",
                                       "trunk_7", "trunk_8"])]
        if self.task==2:
            # collision_walls = ("walls",["wall_1", "wall_2","wall_3", "wall_4"])
            collision_walls = ("walls",["wall_3", "wall_4"])

            collision_obstacle =  ("obstacles",["obstacle_1"])
            collision_groups.extend([collision_walls, collision_obstacle])
        if self.task==3:
            collision_walls = ("walls",["wall_3", "wall_4"])
            collision_walls = ("walls",["wall_3", "wall_4"])

            collision_groups.extend([collision_walls])

        self.collision_groups_list = collision_groups

        if self.task == 0:
            max_a = self.qpos_offset
            min_a = - self.qpos_offset

        else:
            max_a = 1*np.array([1.5,0.6,1.5])
            min_a = 1*np.array([-1.5,-0.6,-1.5])


        super().__init__(self.xml_path, action_spec,
                         observation_spec,
                         gamma, horizon,
                         collision_groups=collision_groups,
                          n_intermediate_steps=4,
                         timestep=timestep,
                         max_a=max_a, min_a=min_a)

        # self.renderer = mujoco.Renderer(self._model)
        self.last_action =  np.array(12 * [0])
        self.torques = list()
        x,y = self._data.qpos[:2]
        self.p_gains = np.array(12 * [kp]) #60
        self.d_gains = np.array(12 * [kd]) #6
        self.foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self._max_torques = np.array([23.7, 23.7, 35.55])
        self._min_torques = np.array([-23.7, -23.7, -35.55])

        self.hip_indices = [0, 3, 6, 9]

        self.sitting_dof_pos = np.array([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)
        # self.target_yaw =0

        self.yaw = None
        self.last_contacts = np.zeros(4)
        self.feet_air_time =  np.zeros(4)
        self.np_rng = np.random.default_rng(seed=0)
        self.sampling = StepProbabilitySampling(self)
        self.sampling_command = StepProbabilitySampling(self)
        self.sampling_friction = StepProbabilitySampling(self)
        self.sampling_mass = StepProbabilitySampling(self)
        self.sampling_gravity = StepProbabilitySampling(self)

        self.default_trunk_mass = self._model.body_mass[1]
        self.default_trunk_inertia = self._model.body_inertia[1]


        if self.task==0:
            curriculum_steps= cfg["rewards"]["curriculum_steps"]
        else:
            curriculum_steps = 1
        self.reward_function = reward_f(self, curriculum_steps=curriculum_steps)
        self.perturbation = DefaultDomainPerturbation(self)
        self.reward_function.init()

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

    def handle_domain_randomization(self, function="setup"):
        if function == "setup":
            self.should_randomize_domain_perturbation = self.sampling.setup()
        elif function == "step":
            self.should_randomize_domain_perturbation = self.sampling.step()

        if self.should_randomize_domain_perturbation:

            self.perturbation.sample()

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
        self.orientation_quat = R.from_quat([self._data.qpos[4], self._data.qpos[5], self._data.qpos[6], self._data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()


    def _modify_observation(self, obs):
        # target yaw
        #
        diff_pose = self._data.qpos[7:19].copy() - self.default_dof_pos.copy()
        obs[:12] = diff_pose
        linear_velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])
        angular_velocity = self.orientation_quat_inv.apply(self._data.qvel[3:6])
        projected_gravity = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        contacts = np.array([self.check_collision("floor", foot_name) for foot_name in self.foot_names])
        airtime = np.array([self.reward_function.time_since_last_touchdown_fr,
                               self.reward_function.time_since_last_touchdown_fl,
                               self.reward_function.time_since_last_touchdown_rr,
                               self.reward_function.time_since_last_touchdown_rl])
        obs[24:36]= self.last_action #last action
        obs = self.obs_helper.modify_obs(obs, "gravity_projection", projected_gravity)
        obs = self.obs_helper.modify_obs(obs, "angular_velocity", angular_velocity)
        obs = self.obs_helper.modify_obs(obs, "linear_velocity", linear_velocity)
        obs = self.obs_helper.modify_obs(obs, "feet_contact", contacts)
        obs = self.obs_helper.modify_obs(obs, "feet_airtime", airtime)
        obs = self.obs_helper.modify_obs(obs, "height", self._data.qpos[2])


        if self.task == 0:
            commands =  np.array([self.target_vx, self.target_vy, self.target_yaw])
            obs = self.obs_helper.modify_obs(obs, "commands", commands)

        if self.task == 1:
           self.target_vx, self.target_vy, self.target_yaw = obs[54:57]

        if self.task == 2:
            commands =  np.array([self.target_vx, self.target_vy, self.target_yaw])
            obs = self.obs_helper.modify_obs(obs, "commands", commands)

            linear_velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])
            vx = linear_velocity[0]
            vy = linear_velocity[1]
            wz = self._data.qvel[5]
            current_velocity = np.array([vx, vy, wz])*np.array([2,2,0.25])  # (3,)
            robot_orientation  = self.orientation_euler[2] # (1,)

            obs = self.obs_helper.modify_obs(obs, "current_velocity", current_velocity)
            obs = self.obs_helper.modify_obs(obs, "orientation", robot_orientation)
            pos_diff = self._data.body("obstacle_1").xpos[:3] - self._data.body("trunk").xpos[:3]
            relative_obstacle_pos = self.orientation_quat_inv.apply(pos_diff)
            vel_diff = self._data.qvel[:3] - self._data.joint("obstacle").qvel[:3]
            relative_obstacle_vel = self.orientation_quat_inv.apply(vel_diff)
            # distance = self._data.site("target").xpos[0] - self._data.qpos[0]
            # if distance > 0:
            obs = self.obs_helper.modify_obs(obs, "obstacle_pose", relative_obstacle_pos[:2])
            obs = self.obs_helper.modify_obs(obs, "obstacle_velocity", relative_obstacle_vel[:2])
            # else:
            #     obs = self.obs_helper.modify_obs(obs, "obstacle_pose", [-1,-1])
            #     obs = self.obs_helper.modify_obs(obs, "obstacle_velocity", [-1,-1])



            # room_width = self._room_width
            # obs = self.obs_helper.modify_obs(obs, "room_width",room_width)
            # obs = self.obs_helper.modify_obs(obs, "obstacle_bounding_box",obstacle_bounding_box_robot)
            # obs = self.obs_helper.modify_obs(obs, "robot_bounding_box",obstacle_bounding_box_robot)
        if self.task == 3:
            commands =  np.array([self.target_vx, self.target_vy, self.target_yaw])
            obs = self.obs_helper.modify_obs(obs, "commands", commands)

            linear_velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])
            vx = linear_velocity[0]
            vy = linear_velocity[1]
            wz = self._data.qvel[5]
            current_velocity = np.array([vx, vy, wz])*np.array([2,2,0.25])  # (3,)
            robot_orientation  = self.orientation_euler[2] # (1,)

            obs = self.obs_helper.modify_obs(obs, "current_velocity", current_velocity)
            obs = self.obs_helper.modify_obs(obs, "orientation", robot_orientation)

        # Norm and clip
        obs[:12]  *= 1 #joint pos
        obs[12:24] *= 0.05 #joint vel
        obs[24:36] *= 1 # last action
        obs[36:39] = projected_gravity  # gravity
        obs[39:42] *= 0.25
        obs[42:45] *= 2
        obs[45:49] = (obs[45:49] / 0.5) - 1.0
        obs[49:53] = np.clip((obs[49:53] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        obs[53] = np.clip((self._data.qpos[2] -0.5), -1.0, 1.0)*5 #height
        obs[54:57] *= np.array([2,2,0.25])


        return obs



    def _compute_action(self,cur_obs, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            action (np.ndarray): numpy array with the actions
                provided at every step.

        Returns:
            The action to be set in the actual mujoco simulation.

        """

        torques = self.controller(action)
        torques_f = np.clip(torques,-self.torques_limits, self.torques_limits)
        self.torques.append(torques_f.copy())
        return torques

    def _preprocess_action(self, action):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.

        Returns:
            The action to be used for the current step
        """

        action = np.clip(action, -100,100)
        action *=0.25
        action[[0,3,6,9]] *= 0.5


        action_full =self.default_dof_pos.copy() + action




        return action_full

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



    def _step_init(self, obs, action):


        if self.antagonist:
            self._data.ctrl[-1] = self.obstacle_velocity

            self._data.ctrl[-2] = 0
            self._data.ctrl[-3] = self.obstacle_pose_x





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


    def resample_targets(self):

        # commands, bins = self.curriculum.sample(1)
        # self.bin_id = bins

        self.target_vx = np.random.uniform(-1.5,1.5)
        self.target_vy = np.random.uniform(-1,1)
        self.target_yaw = np.random.uniform(-1.5,1.5)

        self.target_vx = np.random.normal(self.target_vx, 0.1,1)[0]
        self.target_vy = np.random.normal(self.target_vy, 0.1,1)[0]
        self.target_yaw = np.random.normal(self.target_yaw, 0.1,1)[0]

        if np.abs(self.target_vx) <= 0.2:
            self.target_vx = 0
        if np.abs(self.target_vy) <= 0.2:
            self.target_vy = 0
        if np.abs(self.target_yaw) <=0.2:
            self.target_yaw = 0


    def calculate_velocity_distance_reward(self, current_velocity, current_pos, target_pos):
        vect = target_pos - current_pos
        norm = np.linalg.norm(vect)
        vect_norm = vect / (norm + 1e-5)
        projected_vel = np.dot(vect_norm,current_velocity)

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

        reward = moving_coef*reward_moving  + angle_coef* np.exp(-np.abs(target_yaw - yaw))
        return reward

    def proximity_penalty(self):

        distance_to_obstacle = np.linalg.norm(np.array(self._model.site("obstacle_1").pos[:2]) - self._data.body("trunk").xpos[:2])
        obstacle_collision_radius = 1
        obstacle_save_radius = 0.8
        robot_collision_bubble_radius = 0.4

        if distance_to_obstacle > obstacle_collision_radius + obstacle_save_radius + robot_collision_bubble_radius:
            penalty = 0
        elif obstacle_collision_radius + robot_collision_bubble_radius + obstacle_save_radius > distance_to_obstacle > obstacle_collision_radius + robot_collision_bubble_radius:

            penalty = -np.exp(2*(obstacle_collision_radius+robot_collision_bubble_radius - distance_to_obstacle))

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

        collision_obstacle =  legs_collision_obstacles + trunk_collision_obstacles + feet_collision_obstacles
        collision_wall  =  legs_collision_walls + trunk_collision_walls + feet_collision_walls
        penalty =  -20*(collision_obstacle)  -10*(collision_wall)
        return penalty

    def reward_velocity(self, current_velocity,target_pos, current_pos):
        vect = target_pos - current_pos
        norm = np.linalg.norm(vect)
        vect_norm = vect / (norm + 1e-5)
        projected_vel = max(np.dot(vect_norm, current_velocity),0)
        r = np.min([projected_vel, 1])
        return r

    def get_reward(self, action):

        if self.task==0:
            return self.reward_function.reward(action)

        if self.task==1:
            return self.reward_task_1()

        if self.task==2 or self.task==3 or self.task==4:
            yaw = self.orientation_euler[2]
            global_vel = self._data.qvel[:2]
            y_pos = self._data.body("trunk").xpos[1]
            desired_yaw = 0  # Assuming 0 is the forward direction
            yaw_deviation_penalty = np.abs(yaw - desired_yaw)
            distance =  self._data.site("target").xpos[0] - self._data.qpos[0]
            penalty = self.collision_penalty()
            distance_to_obstacle = np.linalg.norm(self._data.site("obstacle_1").xpos[:2] - self._data.qpos[:2])

            soft_hit_condition = self._model.site("obstacle_1").size[0] + self._model.site("robot_contact_limit").size[0]
            if self.narrow_passage:
                proximity_penalty = 0
            else:
                proximity_penalty = (distance_to_obstacle < soft_hit_condition) * np.exp(1 / distance_to_obstacle)


            penalties = (np.abs(y_pos) + yaw_deviation_penalty)*(distance < -0.35) + proximity_penalty*(not self.narrow_passage and not self.no_walls)
            reward = 1.5*np.clip(min(1, global_vel[0]), 0,1) - penalties
            return reward + penalty





    def reward(self, state, action, next_state, absorbing):

        reward = self.get_reward(action)

        self.reward_function.step(action)



        return reward

    def update_variables(self, action, cur_obs):
        self.last_action = action


    def get_student_rays(self):
        rf_readings = np.array([self._data.sensor("rf_l_{}".format(i + 1)).data.copy()[0] for i in range(10)])
        rf_readings = np.where(rf_readings == -1, 20, rf_readings)
        rf_readings = np.where(rf_readings == 0, rf_readings + 0.01, rf_readings)
        rf_readings_2 = np.where(rf_readings >= 20, 20, rf_readings)
        rf_readings_f = np.log2(rf_readings_2)
        obs = rf_readings_f
        return obs

    def render(self,record=False, depth=False):

        if self.use_mujoco_renderer:

            self.renderer_main_camera.update_scene(self._data, "backview")
            img = self.renderer_main_camera.render()

            if depth:
                self.renderer_depth.update_scene(self._data, "head", self.mujoco_options)
                depth_image = self.renderer_depth.render()

                depth_image = np.clip(depth_image, 0, 5)
                min_val = depth_image.min()
                max_val = depth_image.max()
                norm_depth_image = 255 * ((depth_image - min_val) / (max_val - min_val))
                # # Shift nearest values to the origin.
                # depth_image -= depth_image.min()
                # # Scale by 2 mean distances of near rays.
                # depth_image /= 2 * depth_image[depth_image <= 1].mean()
                # # Scale to [0, 255]
                # depth_image = 255 * np.clip(depth_image, 0, 1)

                return img, norm_depth_image
            else:
                return img



    def _step_finalize(self):
        # if self.use_mujoco_renderer:
        #     self.renderer_main_camera = mujoco.Renderer(self._model, width=1024, height=720)
        #     self.renderer_depth = mujoco.Renderer(self._model, width=1024, height=720)
        #     self.renderer_main_camera.update_scene(self._data, "head")
        #     self.renderer_depth.update_scene(self._data, "backview", self.mujoco_options)

        # print(self._data.sensor("collision_distance").data)
        self._ts += 1
        self.update_orientation_attributes()
        if self.task==0:

            if self.domain_randomization :
                self.handle_domain_randomization(function="step")

                if self.sampling_command.step() and self.resample:
                    self.resample_targets()


                if self.sampling_friction.step():
                    factor = np.random.uniform(0,1)
                    self.friction_tangential_min = 0.01
                    self.friction_tangential_max = 2.0
                    self.friction_torsional_feet_min =  0.00001
                    self.friction_torsional_feet_max = 0.04
                    self.friction_rolling_feet_min = 0.00001
                    self.friction_rolling_feet_max = 0.02
                    sampled_friction_tangential =  self.friction_tangential_min + (self.friction_tangential_max - self.friction_tangential_min) *factor
                    sampled_friction_torsional_feet = self.friction_torsional_feet_min + (self.friction_torsional_feet_max - self.friction_torsional_feet_min) * factor
                    sampled_friction_rolling_feet = self.friction_rolling_feet_min + (self.friction_rolling_feet_max - self.friction_rolling_feet_min) * factor
                    self._model.geom_friction[[20, 31, 43, 55]] = [sampled_friction_tangential, sampled_friction_torsional_feet, sampled_friction_rolling_feet]

                if self.sampling_mass.step():
                    add_trunk_mass_min = -0.8
                    add_trunk_mass_max = 0.8,
                    trunk_mass = self.default_trunk_mass + np.random.uniform(add_trunk_mass_min,
                                                                                        add_trunk_mass_max)

                    self._model.body_mass[1] = trunk_mass
                    self._model.body_inertia[1] = self.default_trunk_inertia * (
                                self._model.body_mass[1] / self.default_trunk_mass)


                if self.sampling_gravity.step():
                    gravity_min = 8.81
                    gravity_max = 10.81
                    sampled_gravity = np.random.uniform(gravity_min, gravity_max)
                    self._model.opt.gravity[2] = -sampled_gravity

        if self.task==3:

            if not self.static_obstacles:
                for object_id in range(self.number_obstacles):
                    obstacle_pose = self._data.body(f"n-obstacle{object_id}").xpos[1]

                    if obstacle_pose >= self.wall_pos -  2*self.obstacle_sizes[object_id]:


                        action = - np.random.uniform(0.1, 1)
                        self.set_control_obstacle(object_id, action)

                    if obstacle_pose <= -self.wall_pos +  2*self.obstacle_sizes[object_id]:
                        action = np.random.uniform(0.1, 1)

                        self.set_control_obstacle(object_id, action)

        if self.task==2 :
            if not self.static_obstacles:
                if not self.antagonist:
                    obstacle_pose = self._data.body("obstacle_1").xpos[1]
                    robot_pose = self._data.body("trunk").xpos[1]
                    self._data.ctrl[-3] = self.obstacle_pose_x
                    self._data.ctrl[-2] = 0
                    self._data.ctrl[-1] = np.clip((robot_pose - obstacle_pose),-1,1)

        if self.task==3 or self.task==2:
            robot_pose = self._data.qpos[:2]
            target_pose = self._data.site("target").xpos[:2]
            norm_distance = target_pose[0] - robot_pose[0]
            self.pass_finish_line = norm_distance <= 0.1

            if self._ts>=self.horizon:
                self.success = self.robot_collisions == 0 and self.pass_finish_line



        if self.joystick and (self._ts%5 ==0):


            vx,vy,wz = keyboard_control(self.target_vx, self.target_vy, self.target_yaw)

            self.target_vx = vx
            self.target_vy = vy
            self.target_yaw = wz

        if self.train:

            self.total_timesteps += 1

        # gravity_z_proj = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))[2]
        height = self._data.qpos[2]
        fall = self.check_collision("floor", "trunk_t") or height <= 0.1


        if self.task>1:
            legs_collision_obstacles = self._check_collision("legs", "obstacles")
            trunk_collision_obstacles = self._check_collision("trunk", "obstacles")
            self.robot_collisions += (self._check_collision("robot", "obstacles") + self._check_collision("robot", "walls"))
            # collision = self._check_collision("robot", "obstacles") or self._check_collision("robot", "walls")
            self.obstacle_robot_collision = legs_collision_obstacles + trunk_collision_obstacles
            collision_robot_wall = self._check_collision("robot", "walls")
            collision_robot = legs_collision_obstacles or trunk_collision_obstacles or collision_robot_wall


        if fall | collision_robot:
            self.terminates = True



    def is_absorbing(self, state):

        if self.terminates == True:

            return True

        else:
            return False



    def controller(self, action):

        cur_pos = self._data.qpos[7:19].copy()
        cur_vel = self._data.qvel[6:18].copy()

        target_q = action

        torques = self.p_gains * (target_q - cur_pos) - self.d_gains * cur_vel


        return torques


    def get_info(self):
        return {"success": self.success, "collisions": self.robot_collisions}





    def setup(self, obs, cond):

        self.robot_collisions = 0
        self.reward_function.setup()

        self.torques = list()

        self.update_orientation_attributes()
        self.success = False
        self.terminates = False

        self._ts = 0
        self.episode_steps = 0
        self._data.qpos[7:19] = self.default_dof_pos.copy()
        self._data.qpos[0:3] = [-3, 0, 0.36]

        if self._reset_state is None:
            if self.task==2 or self.task==3 or self.task==4:
                self.reset_states(self.task)

            for i in range(0, 10):

                a = self.default_dof_pos.copy() + np.zeros(12)
                torques = self.controller(a)
                for i in range(4):
                    self._data.ctrl[:12] = torques
                    mujoco.mj_step(self._model, self._data)

            self._reset_state = dict(q_pos=self._data.qpos.copy(), q_vel=self._data.qvel.copy())


        else:
            if self.task == 2 or self.task == 3:
                self.reset_states(self.task)
            else:
                self._data.qpos = self._reset_state["q_pos"]
                self._data.qvel = self._reset_state["q_vel"]


            mujoco.mj_step(self._model, self._data)

        self.last_action =np.zeros(12)
        self.last_velocity_command = np.array([self.target_vx, self.target_vy, self.target_yaw])

        if self.task == 0:

            self.handle_domain_randomization(function="setup")


            if cond is not None:
                self.target_vx = cond[0]
                self.target_vy = cond[1]
                self.target_yaw = cond[2]

            else:
                if not self.joystick:
                    self.resample_targets()


    def reset_states(self, task):
        self._model.site("target").pos[0] = 10

        # x = -4 + np.random.choice([-1, 1, 0.5, -0.5, 0, -1.5, 1.5])

        if task==2:
            if self.behaviour_cloning:
                spec = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/navigation_go1_rays_behaviour_cloning.xml")
            else:
                spec = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/navigation_go1_rays.xml")


            geoms = spec.worldbody.geoms
            walls = ["wall_3", "wall_4"]
            self._room_width = np.random.uniform(low=1.2, high=5)
            if self._room_width > 2.0:
                self.narrow_passage = False
                self._obstacle_l = np.random.uniform(0.1, 0.6)
                self._obstacle_w = np.random.uniform(0.1, 0.6)
            else:
                self.narrow_passage = True
                self._obstacle_l = np.random.uniform(0.05, 0.25)
                self._obstacle_w = np.random.uniform(0.05, 0.25)


            self.wall_pos = self._room_width/2
            y_obstacle = np.random.uniform(-self.wall_pos + self._obstacle_w , self.wall_pos - self._obstacle_w)
            y_robot = np.random.uniform(-self.wall_pos + 0.25, self.wall_pos - 0.25)
            self.no_walls = np.random.uniform() < 0.5
            self.no_walls = False
            if self.no_walls:
                z_wall = -2
            else:
                z_wall = 0.8
            for geom in geoms:
                if geom.name in walls:
                    if  geom.pos[1] <0:
                        geom.pos[1] = -self.wall_pos - geom.size[1]/2
                    else:
                        geom.pos[1] = self.wall_pos + geom.size[1]/2
                    geom.pos[2] = z_wall

            obstacle = spec.find_body("obstacle_1")
            obstacle_height = np.random.uniform(low=0.4, high=0.8)
            obstacle.first_geom().size = [ self._obstacle_l , self._obstacle_w ,obstacle_height]
            target_heading = np.random.uniform(-np.pi/4, np.pi/4)
            goal_position_x =  np.random.uniform(6, 20)
            if  self.no_walls:
                goal_position_y = np.random.uniform(-3, 3)
            else:
                goal_position_y = np.random.uniform(-self.wall_pos + 0.5 , self.wall_pos - 0.5)
            goal_site =spec.worldbody.add_site(name="goal",
                          type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.4,0.2,0.28],
                          rgba=[1,1,1,0.2],
                          pos=[goal_position_x, goal_position_y, 0.28], euler=[0,0, target_heading])

            # self._model.site("obstacle_1").size = max(self._obstacle_l , self._obstacle_w )
            model = spec.compile()
            self._model = model

            self._model.site("obstacle_1").size[0] = 1.2*np.sqrt((self._obstacle_l)**2 + (self._obstacle_w)**2)



            self._model.opt.timestep = 0.005
            self._data = mujoco.MjData(model)
            self._data.qpos[7:19] = self.default_dof_pos.copy()
            x = np.random.uniform(-3, 1.5)
            self._data.qpos[0:3] = [x, y_robot, 0.36]
            self.obstacle_pose_x = np.random.uniform(2, 3.5)

            self._data.joint("obstacle").qpos[:3] = [self.obstacle_pose_x, y_obstacle,obstacle_height]
            self._model.site("target").pos[0] = self.obstacle_pose_x + 1
            self._model.site("target").size[1] = self.wall_pos
            self._model.site("target").pos[-1] = 0.1
            if self._viewer is not None:
                self._viewer.load_new_model(self._model)

            if self.use_mujoco_renderer:
                self.renderer_main_camera = None
                self.renderer_depth = None
                self.renderer_main_camera = mujoco.Renderer(self._model, width=1024, height=720)
                self.renderer_depth = mujoco.Renderer(self._model, width=106, height=60)
                self.renderer_depth.enable_depth_rendering()

            if self.task == 2:
                if not self.antagonist:
                    if  not self.static_obstacles:
                        self._data.ctrl[-1] = np.random.uniform(-1, 1)
                    else:
                        self._data.ctrl[-1] = 0


                else:
                    self._data.ctrl[-1] = self.obstacle_velocity

                self._data.ctrl[-3] = self.obstacle_pose_x
                self._data.ctrl[-2] = 0


        if task==3:
            spec = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/navigation_go1_rays_corridor.xml")
            obstacle = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/obstacle.xml")

            geoms = spec.worldbody.geoms
            walls = ["wall_3", "wall_4"]
            self._room_width = np.random.uniform(low=2, high=5)

            self.wall_pos = self._room_width / 2

            for geom in geoms:
                if geom.name in walls:
                    if geom.pos[1] < 0:
                        geom.pos[1] = -self.wall_pos - geom.size[1] / 2
                    else:
                        geom.pos[1] = self.wall_pos + geom.size[1] / 2

            while obstacle.keys:
                obstacle.keys[-1].delete()
            n_obstacles = []


            self.wall_pos = self._room_width / 2
            self.obstacle_pos_x_list = []
            obstacle_pos_y_list = []
            self.obstacle_sizes = []


            for i in range(self.number_obstacles):
                obstacle_l = np.random.uniform(0.1, 0.6)
                obstacle_w = np.random.uniform(0.1, 0.6)
                self.obstacle_sizes.append(obstacle_w)
                y_obstacle = np.random.uniform(-self.wall_pos + obstacle_w, self.wall_pos - obstacle_w)

                obstacle.find_body('obstacle').first_geom().size = [obstacle_l, obstacle_w, 0.8]

                frame = spec.worldbody.add_frame(pos=[i*3, y_obstacle, 0.8])
                frame.attach_body(obstacle.find_body('obstacle'),'n-', str(i))

                spec.actuators[-2].refsite = "reference"
                spec.actuators[-3].refsite = "reference"

                n_obstacles.append(spec.find_body('n-obstacle{}'.format(i)).first_geom().name)
                self.obstacle_pose_x = i*3
                self.obstacle_pos_x_list.append(self.obstacle_pose_x)
                obstacle_pos_y_list.append(y_obstacle)

            model = spec.compile()


            self._model = model
            self._model.opt.timestep = 0.005
            self._data = mujoco.MjData(model)
            self._model.site("target").pos[0] = self.obstacle_pose_x + 1
            self._model.site("target").size[1] = self.wall_pos


            if self._viewer is not None:
                self._viewer.load_new_model(self._model)

            collision_obstacle =  ("obstacles",n_obstacles)
            self.collision_groups_list.extend([collision_obstacle])
            self.collision_groups = self.create_collision_groups(self.collision_groups_list)


            self.set_control_obstacle(static=self.static_obstacles)

            #
            # self._model.site("target").pos[0]=40

    def set_control_obstacle(self, object_id=None, control_value=None, static=False):

        def initialize_controller(static):
            for i in range(self.number_obstacles):
                j = i * 3
                slide_0 = 12 + j
                slide_1 = 15 + j
                if static:
                    action = 0
                else:
                    velocity = np.random.uniform(0.1, 1)
                    action =np.random.uniform(-velocity, velocity)

                self._data.ctrl[slide_0:slide_1] = [self.obstacle_pos_x_list[i],
                                                    0,
                                                    action]

        def set_control_2_obstacle(object_id, control_value):
            j = object_id * 3
            idx = 14 + j
            self._data.ctrl[idx] =control_value

        if object_id is None:
            initialize_controller(static)
        else:
            set_control_2_obstacle(object_id, control_value)














