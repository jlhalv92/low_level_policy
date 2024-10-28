
from src.envs.mujoco import MuJoCo, ObservationType, mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.envs.reward.reward import RudinOwnVarReward as reward_f
from src.utils.height_map  import *
import json# Function to read the JSON file


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

        if task==1:
            xml_path = "src/envs/unitree_go1/navigation_walk_go1.xml"

        if task==2:
            xml_path = "src/envs/unitree_go1/obstacles_navigation_go1.xml"

        if task==3:
            xml_path = "src/envs/unitree_go1/obstacles_navigation_complex_go1.xml"
            self.obstacles = ["obstacle_1", "obstacle_2", "obstacle_3"]

        if task==4:
            xml_path = "src/envs/unitree_go1/obstacles_navigation_complex_go1_evaluation_v{}.xml".format(version)
            self.obstacles = ["obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4"]

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
        angular_velocity = [("angular_velocity", "a_velocity", ObservationType.VELOCITY)]
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
        
        elif self.task==2 or self.task==3 or self.task==4:
            target_pos = [("target_pose", "target", ObservationType.POS)]
            obstacle_pos = [("obstacle_pose", "obstacle_1", ObservationType.POS_OBSTACLE)]
            robot_pos = [("robot_pose", "trunk", ObservationType.POS)]
            robot_orientation = [("orientation", "orientation_sensor", ObservationType.ORIENTATION)]
            # distance_to_target = [("distance_target", "target", ObservationType.DISTANCE)]
            # occupancy_grid = [("occupancy_grid", "occupancy_grid", ObservationType.OCCUPANCY)]
            current_velocity = [("current_velocity", "a_velocity", ObservationType.HEADING)]
            observation_spec = (propioception +
                                last_actions +
                                gravity_projection +
                                angular_velocity +
                                linear_velocity +
                                contact +
                                airtime +
                                height +
                                commands +
                                robot_pos +  # (2,)
                                current_velocity + # (3,)
                                robot_orientation + # (1,)
                                target_pos + # (2,)
                                obstacle_pos ) # (10,)

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
        if self.task==2:
            collision_walls = ("walls",["left_wall", "right_wall","front_wall", "back_wall"])
            collision_obstacle =  ("obstacles",["obstacle_1"])
            collision_groups.extend([collision_walls, collision_obstacle])
        if self.task==3 or self.task==4:
            collision_walls = ("walls",["left_wall", "right_wall","front_wall", "back_wall"])
            collision_obstacle =  ("obstacles",self.obstacles)
            collision_groups.extend([collision_walls, collision_obstacle])
        if self.task == 0:
            max_a = self.qpos_offset
            min_a = - self.qpos_offset

        else:
            max_a = np.array([1.,1.,1.])
            min_a = np.array([-1.,-1,-1])


        super().__init__(xml_path, action_spec, observation_spec, gamma, horizon, collision_groups=collision_groups,
                          n_intermediate_steps=4, timestep=timestep,
                         max_a=max_a, min_a=min_a)


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
        diff_pose = self._data.qpos[7:19].copy() - self.default_dof_pos.copy()
        obs[:12] = diff_pose
        linear_velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])
        angular_velocity = self.orientation_quat_inv.apply(self._data.qvel[3:6])
        projected_gravity = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))

        obs[24:36]= self.last_action #last action
        obs[36:39] = projected_gravity # gravity
        obs[39:42] = angular_velocity
        obs[42:45] = linear_velocity
        obs[45:49] = np.array([self.check_collision("floor", foot_name) for foot_name in self.foot_names])
        obs[49:53] = np.array([self.reward_function.time_since_last_touchdown_fr,
                               self.reward_function.time_since_last_touchdown_fl,
                               self.reward_function.time_since_last_touchdown_rr,
                               self.reward_function.time_since_last_touchdown_rl])
        obs[53] = self._data.qpos[2] #height

        if self.task == 0:
            obs[54:57] =  np.array([self.target_vx, self.target_vy, self.target_yaw])

        if self.task == 1:

           self.target_vx, self.target_vy, self.target_yaw = obs[54:57]

        if self.task == 2 or  self.task == 3 or self.task == 4:
            obs[54:57] =  np.array([self.target_vx, self.target_vy, self.target_yaw])

            linear_velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])
            vx = linear_velocity[0]
            vy = linear_velocity[1]
            wz = self._data.qvel[5]
            # obs[-3:] = np.array([vx,vy,wz])*np.array([2,2,0.25])
            robot_pos = self._data.qpos[:2]
            # (2,)
            current_velocity = np.array([vx, vy, wz])*np.array([2,2,0.25])  # (3,)
            robot_orientation  = self.orientation_euler[2] # (1,)
            target_pos = self._data.geom("target").xpos[:2]  # (2,)

            # i = self.indices[self.obstacle_id]
            i = self.obstacle_id

            x,y = self._data.geom(self.obstacles[i]).xpos[:2]
            s= 2*self._model.geom(self.obstacles[i]).size[0]
            theta = 0
            obstacle_pos =calculate_square_vertices(x, y, s, theta)
            obs[57:59] = robot_pos
            obs[59:62] = current_velocity
            obs[62] = robot_orientation
            obs[63:65] = target_pos - robot_pos
            obs[65:] = (obstacle_pos -robot_pos).flatten()



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

    def closest_obstacle_in_front(self, robot_pos, n_obstacles=3, last_id=0):
        """
        Given a robot position (x, y) and a set of obstacle positions [(id, x, y), ...],
        returns the id of the closest obstacle directly in front of the robot in the x-direction.

        Parameters:
        - robot_pos: tuple (x, y) representing the robot's position.
        - obstacles: list of tuples [(id, x, y), ...] representing obstacles with their (id, x, y) coordinates.

        Returns:
        - closest_id: ID of the closest obstacle in front of the robot. None if no obstacle is found in front.
        """
        robot_x, robot_y = robot_pos
        min_distance = float('inf')
        closest_id = last_id
        obstacles = [(i,self._data.geom(self.obstacles[i]).xpos[0],self._data.geom(self.obstacles[i]).xpos[1]) for i in range(n_obstacles)]

        for obs_id, obs_x, obs_y in obstacles:
            # Check if the obstacle is in front of the robot (same y level and x > robot_x)
            if obs_x > robot_x:
                obstacle_length = self._model.geom(self.obstacles[obs_id]).size[0]
                distance = obs_x + obstacle_length - robot_x  # Calculate horizontal distance
                if distance < min_distance:
                    min_distance = distance
                    closest_id = obs_id

        return closest_id

    def _step_init(self, obs, action):

        if self.task == 3 or self.task == 4:
            # robot_pos_x = self._data.body("trunk").xpos[0]
            # obstacle_1_pos_x = self._data.geom(self.obstacles[self.indices[0]]).xpos[0]
            # obstacle_2_pos_x = self._data.geom(self.obstacles[self.indices[1]]).xpos[0]
            #
            # obstacle_1_size_x = self._model.geom(self.obstacles[self.indices[0]]).size[0]
            # obstacle_2_size_x = self._model.geom(self.obstacles[self.indices[1]]).size[0]
            # if robot_pos_x <= obstacle_1_pos_x + obstacle_1_size_x:
            #     self.obstacle_id = 0
            #
            # if obstacle_2_pos_x + obstacle_2_size_x > robot_pos_x > obstacle_1_pos_x + obstacle_1_size_x:
            #     self.obstacle_id = 1
            #
            # if obstacle_2_pos_x + obstacle_2_size_x < robot_pos_x:
            #     self.obstacle_id = 2

            self.obstacle_id = self.closest_obstacle_in_front(self._data.body("trunk").xpos[:2], last_id=self.obstacle_id, n_obstacles=self.task)

            if self.antagonist:
                self._data.ctrl[12] = self.obstacle_velocity
            else:
                self._data.ctrl[12] = 0.0

        if self.task == 2:

            if self.curriculum or self.max_level:

                pos_obstacle = self._data.joint(self.obstacles[0] + "_y").qpos
                robot_y = self._data.qpos[1]
                velocity = self.obstacle_velocity * (robot_y - pos_obstacle)
                self._data.ctrl[12] = velocity

            else:

                self._data.ctrl[12] = self.obstacle_velocity


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
        legs_collision_obstacles = self._check_collision("legs", "obstacles")
        trunk_collision_obstacles = self._check_collision("trunk", "obstacles")
        trunk_collision_walls = self._check_collision("trunk", "walls")


        penalty =  -5 *(legs_collision_walls + trunk_collision_walls)  -10 *(legs_collision_obstacles + trunk_collision_obstacles)

        return penalty

    def reward_velocity(self, current_velocity,target_pos, current_pos):

        vect = target_pos - current_pos
        norm = np.linalg.norm(vect)
        vect_norm = vect / (norm + 1e-5)
        projected_vel = np.clip(np.dot(vect_norm, current_velocity),0,1)

        r = np.min([projected_vel, 1])
        return r

    def get_reward(self, action):

        if self.task==0:
            return self.reward_function.reward(action)

        if self.task==1:
            return self.reward_task_1()

        if self.task==2 or self.task==3 or self.task==4:
            current_pos = self._data.body("trunk").xpos[:2]
            target_pos = self._data.geom("target").xpos[:2]
            # norm_distance = np.linalg.norm(target_pos - current_pos)
            vect_direction = target_pos - current_pos
            norm_distance = np.linalg.norm(vect_direction)
            target_vec_norm = vect_direction / (norm_distance + 1e-5)

            target_yaw = np.arctan2(target_vec_norm[1], target_vec_norm[0])
            yaw = self.orientation_euler[2]
            reward_angle = np.exp(-np.abs(target_yaw - yaw))

            local_vel = self.orientation_quat_inv.apply(self._data.qvel[:3])[:2]
            velocity_norm = np.linalg.norm(local_vel)
            reward_velocity = self.reward_velocity(local_vel, target_pos, current_pos)

            # global_vel = self._data.qvel[:1]/(np.linalg.norm(self._data.qvel[:1]) + 1e-5)

            # velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])


            stall_penalty = -1 if norm_distance > 0.5 and  velocity_norm < 0.1 else 0
            # condition = 1 if global_vel > 0 and norm_distance >= 0.5 else 0
            # condition2 = 2 if (norm_distance < 0.5) else 0
            # reward = (3*(condition + condition2))/(1+norm_distance)
            penalty = self.collision_penalty() + stall_penalty
            reward_target = 5*(1/(1+norm_distance)) if norm_distance < 0.5 else 0
            reward = 0.5*reward_angle + reward_velocity + reward_target

            return reward + penalty





    def reward(self, state, action, next_state, absorbing):


        reward = self.get_reward(action)

        self.reward_function.step(action)



        return reward

    def update_variables(self, action, cur_obs):
        self.last_action = action



    def _step_finalize(self):
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


        if self.task==2:
            robot_pose = self._data.qpos[:2]
            target_pose = self._data.geom("target").xpos[:2]
            norm_distance = np.linalg.norm(target_pose - robot_pose)
            self.success = False

            if norm_distance <= 0.2 and self._ts>=self.horizon:
                self.success = True
                self.level +=1
                self.level = np.clip(self.level, 0,6)



            if norm_distance >= 4 and self._ts>=self.horizon:
                self.level -=1
                self.level = np.clip(self.level, 0,6)


        if self.joystick:

            joystick=read_control_values("src/control.json")
            self.target_vx = joystick["control"]["vx"]
            self.target_vy = joystick["control"]["vy"]
            self.target_yaw = joystick["control"]["wz"]


        if self.train:

            self.total_timesteps += 1

        # gravity_z_proj = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))[2]
        height = self._data.qpos[2]
        fall = self.check_collision("floor", "trunk_t") or height <= 0.1


        hit_obstacle = False

        if self.task>1:
            legs_collision_obstacles = self._check_collision("legs", "obstacles")
            trunk_collision_obstacles = self._check_collision("trunk", "obstacles")
            hit_obstacle = legs_collision_obstacles or trunk_collision_obstacles


        if fall or hit_obstacle:
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
        return {"success": self.success}

    def setup(self, obs, cond):

        self.reward_function.setup()

        self.torques = list()

        self.update_orientation_attributes()
        self.success = False
        self.terminates = False

        self._ts = 0

        self._data.qpos[7:19] = self.default_dof_pos.copy()

        self._data.qpos[0:3] = [-3, 0, 0.36]



        if self._reset_state is None:
            if self.task==2 or self.task==3 or self.task==4:
                self.obstacle_id = 0
                self.reset_states(self.task)

            for i in range(0, 10):

                a = self.default_dof_pos.copy() + np.zeros(12)
                torques = self.controller(a)
                for i in range(4):
                    self._data.ctrl[:12] = torques
                    mujoco.mj_step(self._model, self._data)

            self._reset_state = dict(q_pos=self._data.qpos.copy(), q_vel=self._data.qvel.copy())


        else:
            self._data.qpos = self._reset_state["q_pos"]
            self._data.qvel = self._reset_state["q_vel"]
            if self.task == 2 or self.task == 3:
                self.reset_states(self.task)

            mujoco.mj_step(self._model, self._data)



        self.last_dof_vel = self._reset_state["q_vel"][6:]
        self.last_dof_pos = self._reset_state["q_pos"][6:]

        self.last_action =np.zeros(12)


        if self.task == 0:

            self.handle_domain_randomization(function="setup")


            if cond is not None:
                self.target_vx = cond[0]
                self.target_vy = cond[1]
                self.target_yaw = cond[2]

            else:

                self.resample_targets()


    def reset_states(self, task):


        if self.curriculum:
            self.obstacle_curriculum()
        if self.max_level:
            self.obstacle_velocity = 1

        y = np.random.choice([-1, 1, 0.5, -0.5, 0, -2, 2, -1.5, 1.5])

        if task==2:
            self._data.qpos[0:3] = [-4, y, 0.36]

            x = np.random.uniform(-2.5,2.5)


            self._data.joint(self.obstacles[0] + '_x').qpos = x
            self._data.joint(self.obstacles[0] + '_y').qpos = y


        if task==3:

            self.indices = np.random.permutation(len(self.obstacles))
            x = [-2.5, 0, 2.5]
            self._data.qpos[0:3] = [-4, y, 0.36]

            for i,n in enumerate(self.indices):
                x_p = x[i]
                if n == 1:
                    x_p = x[i] -2
                if n == 2:
                    x_p = x[i] -3
                if i>0:
                    y = np.random.uniform(-2,2)

                self._data.joint(self.obstacles[n] + '_x').qpos = x_p
                self._data.joint(self.obstacles[n] + '_y').qpos = y


    def obstacle_curriculum(self):
        # print("level", self.level)
        self.obstacle_velocity_levels = 2*np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])

        if self.level<= 5:
            self.obstacle_velocity = self.obstacle_velocity_levels[self.level]
        else:
            self.obstacle_velocity = np.random.choice(self.obstacle_velocity_levels[1:])













