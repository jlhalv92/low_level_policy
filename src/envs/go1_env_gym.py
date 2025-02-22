
from src.envs.mujoco_env import MuJoCo, ObservationType, mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.envs.reward.reward import RudinOwnVarReward as reward_f
from src.utils.height_map  import *
import json# Function to read the JSON file
import pygame

def calculate_square_vertices(x, y, s, theta):
    # Half diagonal distance from the center to a vertex
    half_diagonal = (s * np.sqrt(2)) / 2

    # List to store the coordinates of the four vertices
    vertices = [[x,y]]

    # Iterate over the 4 angles (45° + k * 90°)
    for k in range(4):
        angle =  theta + np.pi/4 + k * (np.pi / 2)
        vertex_x = x + half_diagonal * np.cos(angle)
        vertex_y = y + half_diagonal * np.sin(angle)
        vertices.append((round(vertex_x, 3), round(vertex_y, 3)))

    return np.array(vertices)

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
            running = False
            print("running off")

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        vx = min(vx + 0.1,1.5)
    if keys[pygame.K_UP]==False and keys[pygame.K_DOWN]==False:
        vx = 0
    if keys[pygame.K_LEFT]==False and keys[pygame.K_RIGHT]==False:
        vy = 0
    if keys[pygame.K_w]==False and keys[pygame.K_e]==False:
        wz = 0

    if keys[pygame.K_DOWN]:
        vx = max(vx - 0.1,-1.5)

    if keys[pygame.K_LEFT]:
        vy = min(vy + 0.1,1.5)
    if keys[pygame.K_RIGHT]:
        vy = max(vy - 0.1,-1.5)
    if keys[pygame.K_w]:
        wz = min(wz + 0.1,1.5)

    if keys[pygame.K_e]:
        wz = max(wz - 0.1,-1.5)

    return vx,vy,wz





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
        self._all_obstacles_move = False
        self.domain_randomization = cfg["domain_randomization"]
        self.indices = [0, 1, 2]
        self.obstacle_id = 0

        self.curriculum = cfg["curriculum"]
        self.resample =  cfg["resample"]
        self.joystick =  cfg["joystick"]
        if self.joystick:
            pygame.init()
            screen_width, screen_height = 300, 300
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Keyboard Joystick Simulation")


        self.level = 0
        self.antagonist = cfg["antagonist"]
        self.max_level = cfg["max_level"]
        self.obstacle_velocity = 0


        # self.occupancy_grid = OccupancyGrid(15,15,0.01)


        self.default_dof_pos = np.array([
            -0.2, -0.7, 1.25,
            0.2, 0.7, -1.25,
            0.2, -1, 1.35,
            -0.2, 1, -1.35,
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

        xml_path = "src/envs/unitree_go1/walk_go1.xml"

        action_spec = ["FL_hip", "FL_thigh", "FL_calf",
                       "FR_hip", "FR_thigh", "FR_calf",
                       "RL_hip", "RL_thigh", "RL_calf",
                       "RR_hip", "RR_thigh", "RR_calf"]

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
        commands = [("commands", "command", ObservationType.COMMANDS)]


        if self.task==0:

            observation_spec = (propioception +
                                gravity_projection +
                                last_actions +
                                linear_velocity +
                                angular_velocity +
                                commands)


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
                            ("trunk_8", ["trunk_8"])]

        max_a = np.inf
        min_a = - np.inf




        super().__init__(xml_path, action_spec, observation_spec, gamma, horizon, collision_groups=collision_groups,
                          n_intermediate_steps=1,n_substeps=4, timestep=timestep,
                         max_a=max_a, min_a=min_a)


        self.last_action =  np.array(12 * [0])
        self.torques = list()
        self.p_gains = np.array(12 * [kp]) #60
        self.d_gains = np.array(12 * [kd]) #6
        self.foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


        self.yaw = None
        self.last_contacts = np.zeros(4)
        self.feet_air_time =  np.zeros(4)



        self.reward_function = reward_f(self, curriculum_steps=1)
        self.perturbation = DefaultDomainPerturbation(self)
        self.reward_function.init()


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
        self.orientation_quat = R.from_quat([self._data.qpos[4], self._data.qpos[5], self._data.qpos[6], self._data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()


    def get_robot_observation(self):

        noise_joint_angles = np.random.uniform(low=-0.03, high=0.03, size=12)
        noise_joint_vel =  np.random.uniform(low=-1.5, high=1.5, size=12)
        vel_noise =  np.random.uniform(low=-0.1, high=0.1, size=3)
        ang_vel_noise =  np.random.uniform(low=-0.2, high=0.2, size=3)
        gravity_noise =  np.random.uniform(low=-0.05, high=0.05, size=3)

        qpos = (self._data.qpos[7:].copy() + noise_joint_angles) - self.default_dof_pos.copy() #(12)

        qvel = self._data.qvel[6:].copy() + noise_joint_vel  #(12)
        previous_action = self.last_action   #(12)
        robot_linear_vel = self.orientation_quat_inv.apply(self._data.qvel[:3]).copy()  + vel_noise#(3)
        robot_angular_vel = self._data.qvel[3:6].copy() + ang_vel_noise #(3)
        commands = np.array([self.target_vx, self.target_vy, self.target_yaw]) #(3)
        projected_gravity = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0])).copy() + gravity_noise #(3)


        # Normalize and clip
        qpos /= 4.6
        qvel /= 35.0
        previous_action /= 10.0

        robot_linear_vel = np.clip(robot_linear_vel / 10.0, -1.0, 1.0)
        robot_angular_vel= np.clip(robot_angular_vel / 50.0, -1.0, 1.0)

        observation = np.concatenate([qpos,
                                qvel,
                                projected_gravity,
                                previous_action,
                                robot_linear_vel,
                                robot_angular_vel,
                                commands])

        return observation

    def _modify_observation(self, obs):

        return self.get_robot_observation()




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
        self.target_vy = np.random.uniform(-0.8,0.8)
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



    def reward(self, state, action, next_state, absorbing):


        reward = 0

        self.reward_function.step(action)

        return reward

    def update_variables(self, action, cur_obs):
        self.last_action = action

    def _step_init(self, obs, action):
        if self.joystick:


            vx,vy,wz = keyboard_control(self.target_vx, self.target_vy, self.target_yaw)
            self.target_vx = vx
            self.target_vy = vy
            self.target_yaw = wz

    def _step_finalize(self):
        self._ts += 1
        self.update_orientation_attributes()

        height = self._data.qpos[2]
        fall = self.check_collision("floor", "trunk_t") or height <= 0.1


        if fall :
            self.terminates = True



    def is_absorbing(self, state):

        if self.terminates == True:

            return True

        else:
            return False



    def controller(self, action):
        action *=0.25
        target_q =self.default_dof_pos.copy() + action

        cur_pos = self._data.qpos[7:19].copy()
        cur_vel = self._data.qvel[6:18].copy()

        torques = self.p_gains * (target_q - cur_pos) - self.d_gains * cur_vel


        return torques


    def get_info(self):
        return {"success": self.success}

    def setup(self, obs, cond):

        self._data.qpos[7:19] = self.default_dof_pos.copy()
        self._data.qvel[6:18] = np.zeros(12)
        self._data.qpos[0:3] = [0, 0, 0.325]
        self.last_action =np.zeros(12)
        mujoco.mj_forward(self._model, self._data)
        self.update_orientation_attributes()
        self.reward_function.setup()
        self.torques = list()

        self.success = False
        self.terminates = False


















