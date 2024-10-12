import time

from sympy.strategies.branch import condition

from src.envs.mujoco import MuJoCo, ObservationType, mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.envs.reward.reward import RudinOwnVarReward as reward_f
from src.utils.height_map  import *


from src.envs.curriculum.curriculum import RewardThresholdCurriculum


from collections import deque


class NormalizingBuffer:
    def __init__(self, max_size=100):
        # Initialize the buffer with a fixed max size
        self.buffer = deque(maxlen=max_size)

    def add_sample(self, sample):
        # Add new sample to the buffer
        self.buffer.append(sample)

    def data(self):

        return np.array(self.buffer)

    def normalize(self):
        # Normalize the buffer contents (return mean 0, std 1)
        if len(self.buffer) == 0:
            return None  # If the buffer is empty, return None

        data = np.array(self.buffer)  # Convert buffer to numpy array
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return data  # Avoid division by zero in case of constant data

        normalized_data = (data - mean) / std
        return normalized_data

    def get_buffer(self):
        # Return the current buffer as a list
        return list(self.buffer)







task_map = dict(nominal=0,
                nominal_navigation=6,
                nominal_parkour=1,
                gaps=2,
                hurdles=3,
                steps=4,
                ramps=5)
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

    def __init__(self, horizon, gamma, n_substep=1, target_pose=None, task_name="nominal_parkour",timestep=0.002,  cfg=None):
        """
        Constructor.

        """
        self.resample = True
        self.qpos_offset = np.array([0.1, 0.5, 0.5] * 4)
        max_a = self.qpos_offset
        min_a = - self.qpos_offset

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


        xml_path = "src/envs/unitree_go1/navigation_walk_go1.xml"





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
        target_position = [("target_pose_command", "target_pose_command", ObservationType.TARGET_COMMAND)]
        distance_2_target = [("distance2target", "d2t", ObservationType.DISTANCE)]
        yaw_tracking = [("yaw_tracking", "yaw", ObservationType.HEADING)]


        observation_spec = (propioception +
                            last_actions +
                            gravity_projection +
                            angular_velocity +
                            linear_velocity +
                            contact +
                            airtime +
                            height +
                            commands)

        collision_groups = [("floor", ["floor"]),
                            ("feet", ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
                            ("FL_foot", ["FL_foot"]), ("FR_foot", ["FR_foot"]),
                            ("RL_foot", ["RL_foot"]), ("RR_foot", ["RR_foot"]),
                            ("FR_calf1", ["FR_calf1"]),
                            ("FL_calf1", ["FL_calf1"]),
                            ("RR_calf1", ["RR_calf1"]),
                            ("RL_calf1", ["RL_calf1"]),
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

        super().__init__(xml_path, action_spec, observation_spec, gamma, horizon, collision_groups=collision_groups,
                          n_intermediate_steps=4, timestep=timestep,
                         max_a=max_a, min_a=min_a)


        self.last_action =  np.array(12 * [0])
        self.torques = list()

        self.p_gains = np.array(12 * [kp]) #60
        self.d_gains = np.array(12 * [kd]) #6
        self.foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self._max_torques = np.array([23.7, 23.7, 35.55])
        self._min_torques = np.array([-23.7, -23.7, -35.55])

        self.hip_indices = [0, 3, 6, 9]

        self.sitting_dof_pos = np.array([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)
        # self.target_yaw =0
        self.target_yaw = None
        self.target_vx = None
        self.target_vy = None
        self.yaw = None
        self.last_contacts = np.zeros(4)
        self.feet_air_time =  np.zeros(4)

        self.reward_function = reward_f(self)
        self.reward_function.init()





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
        diff_pose = self._data.qpos[7:].copy() - self.default_dof_pos.copy()
        obs[:12] = diff_pose
        linear_velocity = self.orientation_quat_inv.apply(self._data.qvel[:3])
        # linear_velocity = self._data.qvel[:3]
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
        obs[53] = self._data.qpos[2]
        obs[54:57] =  np.array([self.target_vx, self.target_vy, self.target_yaw])
        # Norm and clip
        # obs[:12]  /= 4.6 #joint pos
        # obs[12:24]/= 35.0 #joint vel
        # obs[24:36] /= 10.0 # last action
        # obs[36:39] = projected_gravity  # gravity
        # obs[39:42] = np.clip(angular_velocity / 50.0, -1.0, 1.0)
        # obs[42:45] = np.clip(linear_velocity / 10.0, -1.0, 1.0)
        # obs[45:49] = (obs[45:49] / 0.5) - 1.0
        # obs[49:53] = np.clip((obs[49:53] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        # obs[53] = np.clip((self._data.qpos[2]/0.325), -1.0, 1.0)
        #
        # obs[54:57] = np.array([self.target_vx, self.target_vy, self.target_yaw])


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

        self.target_vx = np.random.uniform(-2,2)
        self.target_vy = np.random.uniform(-0.6,0.6)
        self.target_yaw = np.random.uniform(-2,2)

        self.target_vx = np.random.normal(self.target_vx, 0.1,1)[0]
        self.target_vy = np.random.normal(self.target_vy, 0.1,1)[0]
        self.target_yaw = np.random.normal(self.target_yaw, 0.1,1)[0]

        if not self.train:
            print("commands: ", self.target_vx, " ", self.target_vy, " ", self.target_yaw)

    def get_reward(self, action):


        return self.reward_function.reward(action)


    def reward(self, state, action, next_state, absorbing):

        # print(self._data.qvel[:3])
        #state[:12] /= 4.6 #joint pos
        # state[12:24] /= 35. #joint vel
        # print("last action: ",state[24:36])
        # print(" action: ", action)
        # print(state[36:39]) # gravity
        # state[39:42]
        # state[42:45]
        # state[45:48]
        #

        reward = self.get_reward(action)

        self.reward_function.step(action)



        return reward

    def update_variables(self, action, cur_obs):
        self.last_action = action



    def _step_finalize(self):
        self._ts += 1
        self.update_orientation_attributes()
        if self._ts % 200 == 0 and self.resample:

            self.resample_targets()
        if self.train:

            self.total_timesteps += 1

        fall = self.check_collision("floor", "trunk_t")

        if fall:
            self.terminates = True





    def is_absorbing(self, state):

        if self.terminates == True:

            return True

        else:
            return False



    def controller(self, action):

        cur_pos = self._data.qpos[7:].copy()
        cur_vel = self._data.qvel[6:].copy()

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
        self._data.qpos[0:3] = [0,0,0.35]
        self._data.qpos[7:] = self.default_dof_pos.copy()
        if self._reset_state is None:
            for i in range(0, 100):

                a = self.default_dof_pos.copy() + np.zeros(12)
                torques = self.controller(a)
                for i in range(4):
                    self._data.ctrl = torques
                    mujoco.mj_step(self._model, self._data)

            self._reset_state = dict(q_pos=self._data.qpos.copy(), q_vel=self._data.qvel.copy())

        else:
            self._data.qpos = self._reset_state["q_pos"]
            self._data.qvel = self._reset_state["q_vel"]

        self.last_dof_vel = self._reset_state["q_vel"][6:]
        self.last_dof_pos = self._reset_state["q_pos"][6:]

        self.last_action =np.zeros(12)

        if cond is not None:
            self.target_vx = cond[0]
            self.target_vy = cond[1]
            self.target_yaw = cond[2]
        else:
            self.resample_targets()













