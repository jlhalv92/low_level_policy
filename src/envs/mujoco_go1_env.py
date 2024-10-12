from src.envs.mujoco import MuJoCo, ObservationType, mujoco
import numpy as np
from pathlib import Path
from copy import deepcopy
from src.utils.height_map  import *
import dm_control.utils.transformations as tr
from scipy import ndimage
_TERRAIN_SMOOTHNESS = 0.85  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE =201/20 # Spatial scale of terrain bumps (in meters).
from mushroom_rl.utils.spaces import Box
from collections import deque
import cv2

def rotate_quaternion_z(quaternion, angle):
    """
    Rotates a quaternion around the Z-axis by a given angle.

    Parameters:
    - quaternion: A list or array with four elements [w, x, y, z] representing the original quaternion.
    - angle: The angle in degrees by which to rotate the quaternion around the Z-axis.

    Returns:
    - A new quaternion [w, x, y, z] after the rotation.
    """
    # Convert the angle from degrees to radians
    angle_rad = np.radians(angle)

    # Create the rotation quaternion for Z-axis rotation
    cos_half_angle = np.cos(angle_rad / 2)
    sin_half_angle = np.sin(angle_rad / 2)
    rotation_quaternion = np.array([cos_half_angle, 0, 0, sin_half_angle])

    # Multiply the original quaternion by the rotation quaternion
    w1, x1, y1, z1 = quaternion
    w2, x2, y2, z2 = rotation_quaternion

    # Quaternion multiplication (Hamilton product)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


class CircularBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size

    def reset(self):
        self.buffer = deque(maxlen=self.size)


    def append(self, item):
        self.buffer.append(item)

    def get(self):
        return np.array(self.buffer)

    def get_window(self,size_w):
        return np.array(self.buffer)[-size_w:].mean(axis=0)


def get_height_at_coordinate(height_field, resolution, center_x, center_y, x_m, y_m):
    # Convert coordinates from meters to centimeters
    x_cm = int(x_m * resolution)
    y_cm = int(y_m * resolution)
    x_adj = center_x + x_cm
    y_adj = center_y + y_cm


    return height_field[
        y_adj, x_adj]
def rotate_points(points, yaw_angle):
    """
    Rotate a set of 3D points around the z-axis by a given yaw angle.

    Parameters:
    points (ndarray): Array of shape (N, 3) representing N points in 3D space.
    yaw_angle (float): The yaw angle in radians.

    Returns:
    ndarray: Rotated points of shape (N, 3).
    """
    # Define the rotation matrix
    cos_theta = np.cos(yaw_angle)
    sin_theta = np.sin(yaw_angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # Apply the rotation to each point
    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points

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

    def __init__(self, horizon, gamma, n_substep=4,  target_pose=None, task_name="nominal_parkour",timestep=0.002,  cfg=None):
        """
        Constructor.

        """
        self.level = 0
        self.scandots = 132
        self.cfg = cfg
        self.action_buffer = CircularBuffer(10)
        self.scandots_buffer = CircularBuffer(2)
        self.size_map_px = 401 #401
        self.size_map = 20 #40
        self.cfg_rewards =cfg["rewards"]
        task = task_map[task_name]
        self.task = task
        self.horizon = horizon
        self._edge_mask = None
        self._cycles_since_last_touchdown_fr = None
        self._cycles_since_last_touchdown_fl = None
        self._cycles_since_last_touchdown_rr = None
        self._cycles_since_last_touchdown_rl = None

        self.reached_target = 0
        self.success = False
        kp =cfg['kp']
        kd =cfg['kd']
        timestep = timestep

        if task==0 or task==6:
            xml_path = "src/envs/unitree_go1/parkour_go1.xml"

        else:
            xml_path = "src/envs/unitree_go1/parkour_go1.xml"

            self.map = HeightMapParkour(self.size_map_px/self.size_map, self.size_map_px, self.size_map_px)



        action_spec = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh",
                       "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
        self.torques_limits = np.array(4 * [23.7, 23.7, 35.55])
        self.target_pose = target_pose
        self._reset_state = None
        self._ts = 0
        self.air_time_max = 0.5

        propioception = [("FR_hip_pos", "FR_hip_joint", ObservationType.JOINT_POS),
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

        # pos_body =[("trunk_pos", "trunk", ObservationType.BODY_POS)]
        feet = ["FR_foot_touch",
                "FL_foot_touch",
                "RR_foot_touch",
                "RL_foot_touch"]
        foot_contacs = [("foot_contacts", feet, ObservationType.CONTACT_SENSOR)]
        last_actions = [("last_actions", "torques", ObservationType.TORQUES)]
        vel_body = [("local_velocity", "velocimeter", ObservationType.SENSOR_VELOCITY)]
        root_orientation = [("root_orientation", "Body_Quat", ObservationType.SENSOR_ORIENTATION)]
        if not self.cfg['use_all_obs'] or self.task==0:
            observation_spec = root_orientation + vel_body + propioception + foot_contacs + last_actions
        else:
            # robot_pose = [("robot_pose", "trunk", ObservationType.BODY_POS)]
            privilege_obs = [("privilege_obs", "robot", ObservationType.PRIVILEGE_OBS)]
            observation_spec = root_orientation + vel_body + propioception + foot_contacs  + last_actions + privilege_obs

        robot_torso = ["torso_{}".format(i) for i in range(1,9)]
        # robot_FR_Leg = ["FR_hip{}".format(i) for i in range(1,3)] + ["FR_thigh{}".format(i) for i in range(1,4)] +  ["FR_calf{}".format(i) for i in range(1,3)]
        # robot_FL_Leg = ["FL_hip{}".format(i) for i in range(1,3)] + ["FL_thigh{}".format(i) for i in range(1,4)] +  ["FL_calf{}".format(i) for i in range(1,3)]
        # robot_RR_Leg =["RR_hip{}".format(i) for i in range(1,3)] + ["RR_thigh{}".format(i) for i in range(1,4)] +  ["RR_calf{}".format(i) for i in range(1,3)]
        # robot_RL_Leg = ["RL_hip{}".format(i) for i in range(1,3)] + ["RL_thigh{}".format(i) for i in range(1,4)] +  ["RL_calf{}".format(i) for i in range(1,3)]

        robot_FR_hip = ["FR_hip{}".format(i) for i in range(1,3)] 
        robot_FR_thigh = ["FR_thigh{}".format(i) for i in range(1,4)]
        robot_FR_calf = ["FR_calf{}".format(i) for i in range(1,2)]

        collision_robot_FR_hip = ("robot_FR_hip", robot_FR_hip)
        collision_robot_FR_thigh = ("robot_FR_thigh", robot_FR_thigh)
        collision_robot_FR_calf = ("robot_FR_calf", robot_FR_calf)

        robot_FL_hip = ["FL_hip{}".format(i) for i in range(1,3)]
        robot_FL_thigh = ["FL_thigh{}".format(i) for i in range(1,4)]
        robot_FL_calf = ["FL_calf{}".format(i) for i in range(1,2)]

        collision_robot_FL_hip = ("robot_FL_hip", robot_FL_hip)
        collision_robot_FL_thigh = ("robot_FL_thigh", robot_FL_thigh)
        collision_robot_FL_calf = ("robot_FL_calf", robot_FL_calf)

        robot_RR_hip = ["RR_hip{}".format(i) for i in range(1,3)]
        robot_RR_thigh = ["RR_thigh{}".format(i) for i in range(1,4)]
        robot_RR_calf = ["RR_calf{}".format(i) for i in range(1,2)]

        collision_robot_RR_hip = ("robot_RR_hip", robot_RR_hip)
        collision_robot_RR_thigh = ("robot_RR_thigh", robot_RR_thigh)
        collision_robot_RR_calf = ("robot_RR_calf", robot_RR_calf)

        robot_RL_hip = ["RL_hip{}".format(i) for i in range(1,3)]
        robot_RL_thigh = ["RL_thigh{}".format(i) for i in range(1,4)]
        robot_RL_calf = ["RL_calf{}".format(i) for i in range(1,2)]

        collision_robot_RL_hip = ("robot_RL_hip", robot_RL_hip)
        collision_robot_RL_thigh = ("robot_RL_thigh", robot_RL_thigh)
        collision_robot_RL_calf = ("robot_RL_calf", robot_RL_calf)
        

        
        # robot_FL_Leg = ["FL_hip{}".format(i) for i in range(1,3)] + ["FL_thigh{}".format(i) for i in range(1,4)]
        # robot_RR_Leg =["RR_hip{}".format(i) for i in range(1,3)] + ["RR_thigh{}".format(i) for i in range(1,4)]
        # robot_RL_Leg = ["RL_hip{}".format(i) for i in range(1,3)] + ["RL_thigh{}".format(i) for i in range(1,4)]


        # robot_body_parts = robot_torso + robot_RL_Leg+ robot_FR_Leg +robot_RR_Leg + robot_FL_Leg
        # collision_robot =("robot", robot_body_parts)
        collision_robot_torso = ("torso", robot_torso)

        collision_groups = [("floor", ["floor"]),
                            ("foot_FR", ["FR"]),
                            ("foot_FL", ["FL"]),
                            ("foot_RR", ["RR"]),
                            ("foot_RL", ["RL"]),
                            collision_robot_torso,
                            collision_robot_FR_hip,
                            collision_robot_FR_thigh,
                            collision_robot_FR_calf,
                            collision_robot_FL_hip,
                            collision_robot_FL_thigh,
                            collision_robot_FL_calf,
                            collision_robot_RR_hip,
                            collision_robot_RR_thigh,
                            collision_robot_RR_calf,
                            collision_robot_RL_hip,
                            collision_robot_RL_thigh,
                            collision_robot_RL_calf]


        self.qpos_offset = np.array([0.2, 0.7, 0.7] * 4)
        zero_action = np.array(4* [0, 0.9, -1.8])

        max_a = self.qpos_offset
        min_a = - self.qpos_offset
        super().__init__(xml_path, action_spec, observation_spec, gamma, horizon, collision_groups=collision_groups,
                         n_substeps=n_substep, timestep=timestep,
                         max_a=max_a, min_a=min_a)


        # self.action_low = min_a
        # self.action_max = max_a
        if self.task==0 or self.task==1:
            if self.task==1:
                self.bumpy_terrain()

        self.last_dof_vel = np.array(12 * [0])
        self.last_dof_pos = np.array(12 * [0])
        self.prev_action = np.array(12 * [0])
        self.last_actions = list()
        self.torques = list()
        self.init_robot_pos = np.array([0,
                                        0,
                                        0.42,
                                        1,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0.9,
                                        -1.8,
                                        0,
                                        0.9,
                                        -1.8,
                                        0,
                                        0.9,
                                        -1.8,
                                        0,
                                        0.9,
                                        -1.8])
        self.init_joints_robot_pos = np.array(4 * [0, 0.9, -1.8])
        # self.p_gains = np.array(12 * [40])
        # self.d_gains = np.array(12 * [1])
        self.air_time_reward = np.zeros(4)
        self.target_idx = 1
        self.target_id = "target"
        self._model.geom(self.target_id).pos.copy()

        self.p_gains = np.array(12 * [kp]) #60
        self.d_gains = np.array(12 * [kd]) #6

        self._action_scale = 1 #0.25
        self._max_torques = np.array([23.7, 23.7, 35.55])
        self._min_torques = np.array([-23.7, -23.7, -35.55])
        self._hip_scale_reduction = 1
        self.hip_indices = [0, 3, 6, 9]
        self.default_dof_pos = np.array(4 * [0,
                                             0.9,
                                             -1.8])
        self.sitting_dof_pos = np.array([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)
        self.target_yaw =0
        self.yaw = 0
        self.last_contacts = np.zeros(4)
        self.feet_air_time =  np.zeros(4)
        self.init_robot_pos = np.array([0,
                                        0,
                                        0.42,
                                        1,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0.9,
                                        -1.8,
                                        0,
                                        0.9,
                                        -1.8,
                                        0,
                                        0.9,
                                        -1.8,
                                        0,
                                        0.9,
                                        -1.8])




        # self._model.hfield_data = np.random.rand(640 * 640) * 0.01



    def _step_init(self, obs, action):
        """
        Allows information to be initialized at the start of a step.

        """


        body_parts = ["torso",
                      "robot_FR_hip", "robot_FR_thigh", "robot_FR_calf",
                      "robot_FL_hip", "robot_FL_thigh", "robot_FL_calf",
                      "robot_RR_hip", "robot_RR_thigh", "robot_RR_calf",
                      "robot_RL_hip", "robot_RL_thigh", "robot_RL_calf"]

        for body_part in body_parts:
            for id in self.collision_groups[body_part]:
                self._model.geom(id).rgba = [1, 1, 1, 1]

        target_pos = self._model.geom(self.target_id).pos.copy()[:2]
        robot_pos = self._data.body("trunk").xpos.copy()[:2]
        vect_direction = target_pos - robot_pos
        norm = np.linalg.norm(vect_direction)
        target_vec_norm = vect_direction / (norm + 1e-5)
        target_yaw = np.arctan2(target_vec_norm[1], target_vec_norm[0])
        _, _, yaw = self.get_roll_pitch_yaw_radians()

        self.yaw = yaw
        self.target_yaw = target_yaw


    def get_collisions(self):

        body_parts = ["torso",
                      "robot_FR_hip", "robot_FR_thigh", "robot_FR_calf",
                      "robot_FL_hip", "robot_FL_thigh", "robot_FL_calf",
                      "robot_RR_hip", "robot_RR_thigh", "robot_RR_calf",
                      "robot_RL_hip", "robot_RL_thigh", "robot_RL_calf"]

        # body_parts = ["torso",
        #               "robot_FR_hip", "robot_FR_thigh",
        #               "robot_FL_hip", "robot_FL_thigh",
        #               "robot_RR_hip", "robot_RR_thigh",
        #               "robot_RL_hip", "robot_RL_thigh"]


        collision_array = list()
        for body_part in body_parts:
            collision = self._check_collision("floor", body_part)
            if collision:
                # print("-------------- {} ----------------".format(self._ts))
                for id in self.collision_groups[body_part]:
                    self._model.geom(id).rgba = [0.3, 0, 0, 1]
            collision_array.append(collision)
 
        return collision_array






        # print(self._cycles_since_last_touchdown_fr, self._cycles_since_last_touchdown_fl, self._cycles_since_last_touchdown_rr, self._cycles_since_last_touchdown_rl  )
    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array).

        """

        grf = [self._get_collision_force("floor", "foot_FL")[:3],
               self._get_collision_force("floor", "foot_FR")[:3],
               self._get_collision_force("floor", "foot_RL")[:3],
               self._get_collision_force("floor", "foot_RR")[:3]]

        return grf[0], grf[1], grf[2], grf[3]

    def sample_terrain(self):
        grid_points = create_evenly_distributed_grid()
        rot_grid_points = rotate_points(grid_points, self.yaw)
        robot_z_pos = self._data.qpos[2].copy()
        robot_pos_xy = self._data.qpos[:2].copy()


        center_x = self.size_map_px//2
        center_y = self.size_map_px//2
        resolution = self.size_map_px//self.size_map
        height_field = self._model.hfield_data.copy().reshape((self.size_map_px,self.size_map_px))
        points_pose = rot_grid_points[:, :2] + robot_pos_xy
        for i in range(self.scandots):

            rot_grid_points[i,2] = get_height_at_coordinate(height_field, resolution, center_x, center_y, points_pose[i, 0], points_pose[i, 1])*5

        # if self._ts%10==0:
        #     x =points_pose[:,0]
        #     y = points_pose[:,1]
        #     z = rot_grid_points[:,2]
        #
        #     # Create a 3D scatter plot
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
        #
        #     # Add color bar which maps values to colors
        #     colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        #
        #     # Set labels
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')
        #
        #     # Show plot
        #     plt.show()
        return rot_grid_points[:,2]
    def get_privilege_obs(self):

        p_obs = np.zeros(self.scandots + 5)
        # self.scandots_buffer.append(self.sample_terrain())
        scandots = self.sample_terrain()
        p_obs[:self.scandots] = scandots


        # print('------------------------------------------------')
        #
        # print((rot_grid_points[:,2]>0.2).reshape((11,12)).astype(int))
        if self.task == 2:
            #gaps
            n = 5
            l = 2
            w = 1.5
            height = 0.5
            gap = 0.2
            p_obs[-5:] = [n,l, w ,height,gap]
        if self.task == 3:
            #hurdles
            n = 5
            l = 0.25
            w = 2
            gap = 2
            height = 0.015*5
            p_obs[-5:] = [n,l,w,height,gap]
        if self.task == 4:
            #steps
            n = 5
            l = 1.5
            w = 2
            step_size = 0.01 * 5
            max_height = 0.01*25
            p_obs[-5:] = [n,l,w,step_size,max_height]
        if self.task == 5:
            #ramps
            n = 5
            l = 1.5
            w= 3
            slope = 0.05/3
            height = 0.05*5+0.15*0.5
            p_obs[-5:] = [n, l, w, slope,height]
        return p_obs

    def _modify_observation(self, obs):
        # target yaw
        obs[2] = self.target_yaw - self.yaw
        # Qpos - REF
        obs[6:18] -= self.init_robot_pos[7:].copy()

        if self.task==0 or self.task==1 or self.task==6:
            obs[-12:] = self.last_actions[-1]

        else:
            obs[-149:-137] = self.last_actions[-1]
            obs[-137:] = self.get_privilege_obs()

        return obs

    def _compute_action(self, obs, action, i):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            obs (np.ndarray): numpy array with the current state of teh simulation;
            action (np.ndarray): numpy array with the actions, provided at every step.

        Returns:
            The action to be set in the actual pybullet simulation.

        """

        torques = self.controller(action)

        torques_f = np.clip(torques,-self.torques_limits, self.torques_limits)

        self.torques.append(torques_f.copy())

        return torques_f

    def calculate_velocity_distance_reward(self, current_velocity, target_velocity, current_pos, target_pos):
        vect = np.array(target_pos) - current_pos
        norm = np.linalg.norm(vect)
        vect_norm = vect / (norm + 1e-5)
        vel_vect_sum = np.sum(vect_norm * current_velocity)
        r = np.min([vel_vect_sum, target_velocity]) / target_velocity
        return r

    def _reward_tracking_yaw(self):
        rew = np.exp(-np.abs(self.target_yaw - self.yaw))
        return rew

    def _reward_lin_vel_z(self, base_lin_vel):
        rew = np.square(base_lin_vel[2])

        return rew

    def _reward_ang_vel_xy(self, base_ang_vel):
        return np.sum(np.square(base_ang_vel[:2]))

    def _reward_orientation(self, projected_gravity):
        rew = np.sum(np.square(projected_gravity[:2]))

        return rew

    def _reward_dof_acc(self, dof_vel):
        return np.sum(np.square((self.last_dof_vel - dof_vel) / self.dt))

    def _reward_action_rate(self, actions):
        return np.linalg.norm(self.last_actions[-1] - actions)

    def _reward_delta_torques(self, torques, last_torques):
        return np.sum(np.square(torques - last_torques))

    def _reward_torques(self, torques):
        return np.sum(np.square(torques))

    def _reward_hip_pos(self, dof_pos):
        error_hip = np.sum(np.square(dof_pos[self.hip_indices] - self.default_dof_pos[self.hip_indices]))

        return error_hip

    def _reward_dof_error(self, dof_pos):
        dof_error = np.sum(np.square(dof_pos - self.default_dof_pos))
        return dof_error

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        contact_fl, contact_fr, contact_rl, contact_rr = self._get_ground_forces()

        contact_forces = np.concatenate([[contact_fl], [contact_fr], [contact_rl], [contact_rr]], axis=0)
        horizontal_force = np.linalg.norm(contact_forces[:, :2], axis=1)

        vertical_force = np.abs(contact_forces[:, 2])

        rew = np.any(horizontal_force > 4*vertical_force)
        return rew.astype(float)

    def _reward_edges(self):
        pos_feet = np.zeros((4,2))
        pos_feet_front = np.zeros((4, 2))
        feet_name = ['FR_foot_site', 'FL_foot_site', 'RR_foot_site', 'RL_foot_site']
        edges_feet = []
        for i,foot in enumerate(feet_name):
            pos_feet[i,:] = self._data.site(foot).xpos.copy()[:2]
            pos_feet_front[i, :] = self._data.site(foot).xpos.copy()[:2] + rotate_points(np.array([0.1,0,0]), self.yaw)[:2]
            # x_cm = int(pos_feet[i,0] * self.size_map_px/40)
            # y_cm = int(pos_feet[i,1] * self.size_map_px/40)
            # x_adj = self.size_map_px//2 + x_cm
            # y_adj = self.size_map_px//2 + y_cm
            # edges_feet.append(self._edge_mask[y_adj, x_adj]/255)
        # plt.scatter(pos_feet[:,0], pos_feet[:,1])
        # plt.scatter(pos_feet_front[:, 0], pos_feet_front[:, 1])
        # plt.show()
        # exit()
        return np.array(edges_feet)

    def quat_rotate_inverse(self, q, v):
        q_w = q[-1]
        q_vec = q[:3]

        # Compute the components a, b, and c
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0

        # Reshape q_vec and v for batch matrix multiplication
        q_vec_reshaped = q_vec
        v_reshaped = v

        # Batch matrix multiplication and squeezing
        dot_product = np.squeeze(np.matmul(q_vec_reshaped, v_reshaped), axis=-1)

        c = q_vec * dot_product * 2.0

        return a - b + c

    def get_dof_pos(self):

        return self._data.qpos[7:].copy()

    def get_dof_vel(self):

        return self._data.qvel[6:].copy()

    def air_time_reward_func(self):
        contacts = self.feet_contact()
        contact_filt = np.logical_or(contacts, self.last_contacts)
        self.last_contacts = contacts
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = np.sum((self.feet_air_time - 0.5) * first_contact)  # reward only on first contact with the ground
        self.feet_air_time *= ~contact_filt
        return rew_airTime

        return self.air_time_reward.sum()
    def _preprocess_action(self, action_offset):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.

        Returns:
            The action to be used for the current step
        """
        # print('hip',(action-self.default_dof_pos)[[0,3,6,9]].max(), (action-self.default_dof_pos)[[0,3,6,9]].min())
        # print('thigh',(action - self.default_dof_pos)[[1,4,7,10]].max(), (action -self.default_dof_pos)[[1,4,7,10]].min())
        # print('calf',(action -self.default_dof_pos)[[2,5,8,11]].max(), (action - self.default_dof_pos)[[2,5,8,11]].min())
        # offset = np.random.uniform(-self.qpos_offset,self.qpos_offset)
        #
        # self._data.qpos[1:3] = [0, 1.5]
        # self._data.qvel[:6] = [0.5, 0, 0,0,0,0]
        # self.qpos_offset * np.sin(5 * self.dt * self._ts)
        # q_pos = self._data.qpos[7:].copy()

        # print('hip',(offset)[[0,3,6,9]].max(), (offset)[[0,3,6,9]].min())
        # print('thigh',(offset)[[1,4,7,10]].max(), (offset)[[1,4,7,10]].min())
        # print('calf',(offset)[[2,5,8,11]].max(), (offset)[[2,5,8,11]].min())
        # print(np.random.uniform(-self.qpos_offset,self.qpos_offset))
        # action = self.default_dof_pos
        # action = np.array(4 * [0, 0.9, -1.8]) - 0.2 * np.sin((np.pi / 2) * self.t)
        # desired_positions = np.linspace(q_pos, self.action_buffer.get_window(), self._n_intermediate_steps)
        #
        # if self.interpolate_actions:
        #     return desired_positions
        #
        # else:
        action = self.default_dof_pos.copy() + action_offset
        self.action_buffer.append(action)
        return self.action_buffer.get_window(self.cfg['interpolate'])

    def reward(self, obs, action, next_obs, absorbing):



        tracking_goal_vel = self.cfg_rewards["tracking_goal_vel"]
        tracking_yaw = self.cfg_rewards["tracking_yaw"]
        lin_vel_z = self.cfg_rewards["lin_vel_z"]
        ang_vel_xy = self.cfg_rewards["ang_vel_xy"]
        orientation = self.cfg_rewards["orientation"]
        dof_acc = self.cfg_rewards["dof_acc"]
        collision = self.cfg_rewards["collision"]
        action_rate = self.cfg_rewards["action_rate"]
        delta_torques = self.cfg_rewards["delta_torques"]
        torques = self.cfg_rewards["torques"]
        hip_pos = self.cfg_rewards["hip_pos"]
        dof_error = self.cfg_rewards["dof_error"]
        feet_stumble =self.cfg_rewards["feet_stumble"]
        airtime =self.cfg_rewards["airtime"]

        g = np.array([.0, 0, -1])

        current_velocity = self._data.sensor("trunk_linvel").data.copy()
        current_angular_velocity = self._data.sensor("torso_angularvel").data.copy()
        base_quat =  self._data.sensor('Body_Quat').data.copy()
        base_lin_vel = self.quat_rotate_inverse(base_quat,current_velocity)
        base_ang_vel = self.quat_rotate_inverse(base_quat, current_angular_velocity)
        projected_gravity = self.quat_rotate_inverse(base_quat, g)

        target_velocity = 1.5

        target_pos = self._model.geom(self.target_id).pos.copy()[:2]
        robot_pos = self._data.body("trunk").xpos.copy()[:2]

        current_velocity = self._data.sensor("trunk_linvel").data.copy()[:2]
        reward_vel = self.calculate_velocity_distance_reward(current_velocity,
                                                                                 target_velocity,
                                                                                 robot_pos,
                                                                                 target_pos)

        reward_lin_vel_z =  self._reward_lin_vel_z(base_lin_vel)


        reward_angular_xy =  self._reward_ang_vel_xy(base_ang_vel)

        reward_feet_stumble = self._reward_feet_stumble()


        reward_orientation = self._reward_orientation(projected_gravity)


        reward_tracking_yaw =  self._reward_tracking_yaw()

        curr_dof_vel = self.get_dof_vel()
        reward_dof_acc = self._reward_dof_acc(curr_dof_vel)
        self.last_dof_vel = curr_dof_vel

        #Action penalties
        reward_action_rate = self._reward_action_rate(action)
        self.last_actions.append(action.copy())


        # Torque penalties
        reward_delta_torques = self._reward_delta_torques(self.torques[-1], self.torques[-2])
        reward_torques = self._reward_torques(self.torques[-1])

        curr_dof_pos = self.get_dof_pos()
        reward_hip_pos =self._reward_hip_pos(curr_dof_pos)
        reward_dof_error = self._reward_dof_error(curr_dof_pos)
        self.last_dof_pos = curr_dof_pos

        reward_collision = any(self.get_collisions())
        #
        #
        #
        #
        # rw_airtime = self.air_time_reward_func()
        #
        # self._reward_edges()
        # rw_edge = 0*np.sum(self._reward_edges())/4

        # print("current velocity: ",current_velocity,
        # "\nreward_vel : ", reward_vel
        #            ,"\n reward_tracking_yaw : ", reward_tracking_yaw
        #            ,"\n  reward_lin_vel_z : ", reward_lin_vel_z ,
        #          "\n  reward_feet_stumble : ", reward_feet_stumble ,
        #       "\n  reward_lin_vel_z : ", reward_lin_vel_z ,
        #       " \n reward_angular_xy : ", reward_angular_xy ,
        #       " \n reward_orientation : ", reward_orientation ,
        #       " \n reward_dof_acc : ", reward_dof_acc ,
        #       "\n  reward_action_rate : ", reward_action_rate ,
        #       " \n reward_delta_torques : ", reward_delta_torques ,
        #       " \n reward_torques : ", reward_torques ,
        #       " \n reward_hip_pos : ", reward_hip_pos ,
        #       " \n reward_dof_error : ", reward_dof_error ,
        #       " \n reward_collision : ", reward_collision ,
        #       " \n rw_airtime : ", rw_airtime)
        #



        # reward =  max(tracking_goal_vel * reward_vel
        #            + tracking_yaw*reward_tracking_yaw +
        #            lin_vel_z * reward_lin_vel_z +
        #           feet_stumble*reward_feet_stumble +
        #           lin_vel_z*reward_lin_vel_z +
        #           ang_vel_xy*reward_angular_xy +
        #           orientation*reward_orientation +
        #           dof_acc*reward_dof_acc +
        #           action_rate*reward_action_rate +
        #           delta_torques*reward_delta_torques +
        #           torques*reward_torques +
        #           hip_pos*reward_hip_pos +
        #           dof_error*reward_dof_error +
        #           collision*reward_collision +
        #           airtime*rw_airtime ,0.)


        # reward = max(tracking_goal_vel * reward_vel
        #            + tracking_yaw*reward_tracking_yaw +
        #            lin_vel_z * reward_lin_vel_z +
        #           feet_stumble*reward_feet_stumble +
        #           lin_vel_z*reward_lin_vel_z +
        #           ang_vel_xy*reward_angular_xy +
        #           orientation*reward_orientation +
        #           dof_acc*reward_dof_acc +
        #           action_rate*reward_action_rate +
        #           delta_torques*reward_delta_torques +
        #           torques*reward_torques +
        #           hip_pos*reward_hip_pos +
        #           dof_error*reward_dof_error +
        #           collision*reward_collision +
        #           airtime*rw_airtime, 0)

        total_reward = (tracking_goal_vel * reward_vel
                     + tracking_yaw * reward_tracking_yaw +
                     lin_vel_z * reward_lin_vel_z +
                     feet_stumble * reward_feet_stumble +
                     lin_vel_z * reward_lin_vel_z +
                     ang_vel_xy * reward_angular_xy +
                     orientation * reward_orientation +
                     dof_acc * reward_dof_acc +
                     action_rate * reward_action_rate +
                     delta_torques * reward_delta_torques +
                     torques * reward_torques +
                     hip_pos * reward_hip_pos +
                     dof_error * reward_dof_error +
                     collision * reward_collision)

        # print("current velocity: ",current_velocity,
        # "\nreward_vel : ",tracking_goal_vel* reward_vel
        #            ,"\n reward_tracking_yaw : ",tracking_yaw* reward_tracking_yaw
        #            ,"\n  reward_lin_vel_z : ",lin_vel_z* reward_lin_vel_z ,
        #          "\n  reward_feet_stumble : ", feet_stumble* reward_feet_stumble ,
        #       "\n  reward_lin_vel_z : ",reward_lin_vel_z* reward_lin_vel_z ,
        #       " \n reward_angular_xy : ",ang_vel_xy* reward_angular_xy ,
        #       " \n reward_orientation : ", orientation*reward_orientation ,
        #       " \n reward_dof_acc : ", dof_acc*reward_dof_acc ,
        #       "\n  reward_action_rate : ", action_rate*reward_action_rate ,
        #       " \n reward_delta_torques : ", delta_torques*reward_delta_torques ,
        #       " \n reward_torques : ", torques*reward_torques ,
        #       " \n reward_hip_pos : ", hip_pos*reward_hip_pos ,
        #       " \n reward_dof_error : ", dof_error*reward_dof_error ,
        #       " \n reward_collision : ",collision* reward_collision ,
        #       " \n rw_airtime : ",airtime* rw_airtime,
        #       " \n total R: ", total_reward)
        if self.cfg_rewards["clip"]:
            reward = max(total_reward, 0.)
        else:
            reward = total_reward
        return  reward

    def get_roll_pitch_yaw(self):
        quat = self._data.sensor('Body_Quat').data.copy()
        return np.rad2deg(tr.quat_to_euler(quat))

    def get_roll_pitch_yaw_radians(self):
        quat = self._data.sensor('Body_Quat').data.copy()
        return tr.quat_to_euler(quat)


    def _step_finalize(self):

        self._ts +=1

        target_pos = self._model.geom(self.target_id).pos.copy()
        robot_pos = self._data.body("trunk").xpos.copy()
        vect = np.array(target_pos) - robot_pos
        norm = np.linalg.norm(vect)
        if norm <= 0.2:

            self._model.geom(self.target_id).rgba = [0, .3, 0, 1]
            self.target_idx += 1
            self.reached_target += 1
            if self.reached_target >= 10 and self.level < 6:
               self.level += 1
               self.reached_target = 0

            self.success =  True
            self.target_id = "target"

    def is_absorbing(self, state):
        # last_target = "target"
        # target_pos = self._model.geom(last_target).pos.copy()[:2]
        # robot_pos = self._data.body("trunk").xpos.copy()[:2]
        # vect = np.array(target_pos) - robot_pos
        # norm = np.linalg.norm(vect)
        fall = self._check_collision("floor", "torso")
        roll, pitch, _ = self.get_roll_pitch_yaw()

        if self.success == True:
            return True


        termination_angle = 30 if self.task == 0 or self.task == 1  else 60
        if np.abs(pitch) >= termination_angle or np.abs(roll) >= termination_angle or fall:

            return True
        return False

    def controller(self, action):

        cur_pos = self._data.qpos[7:].copy()
        cur_vel = self._data.qvel[6:].copy()
        a_lim_min = np.array(4 * [-1.047, -0.663, -2.721])
        a_lim_max = np.array(4 * [1.047, 2.966, -0.837])
        target_q = np.clip(action, a_lim_min, a_lim_max)

        torques = self.p_gains * (target_q - cur_pos) - self.d_gains * cur_vel


        return torques

    def bumpy_terrain(self):
        terrain_size = 2*self.size_map_px
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
        bumps = np.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
        smooth_bumps = ndimage.zoom(bumps, self.size_map_px / float(bump_res))
        # Terrain is elementwise product.
        terrain = 0.2/5 * np.ones((self.size_map_px, self.size_map_px)) * smooth_bumps
        self._model.hfield_data = terrain.ravel()

    def gaps_terrain(self):
        self.map.reset_map()
        self.bumpy_terrain()
        self.map.spawn_object(-5,0,5,1.5,0.1)
        self._data.qpos[2] = 1.5
        self._data.qpos[0] = -5
        gap = 0.05
        x_0 = -5
        l_0 = 2.5
        l_1 =1
        x_1 = gap + l_0 + l_1 + x_0

        for i in range(1,6):

            self._model.geom("target{}".format(i)).pos = np.array([x_1, 0, 1])
            self.map.spawn_object(x_1,0,2,1.5,0.1)
            x_0 = x_1
            l_0 = 1
            l_1 = 1
            x_1 = gap + l_0 + l_1 + x_0

        terrain = self.map.heights

        self._model.hfield_data += terrain.ravel()

    def hurdles_terrain(self):
        self.map.reset_map()
        self.bumpy_terrain()
        self._data.qpos[0] = -5
        gap = 2
        x_0 = -5
        l_0 = 0.1
        l_1 = 0.1
        h =0.015
        x_1 = gap + l_0 + l_1 + x_0
        for i in range(1,6):
            self._model.geom("target{}".format(i)).pos = np.array([x_1+1, 0, 0.5])
            self.map.spawn_object(x_1, 0, 0.25, 2, h)
            x_0 = x_1
            l_0 = 0.1
            l_1 = 0.1
            x_1 = gap + l_0 + l_1 + x_0

        terrain = self.map.heights

        self._model.hfield_data += terrain.ravel()


    def steps_terrain(self):
        #STEP
        self.map.reset_map()
        self.bumpy_terrain()
        self._data.qpos[0] = -5
        gap = 0
        x_0 = -5
        l_0 = 2
        l_1 = 0
        h = 0.01
        x_1 = gap + l_0 + l_1 + x_0
        for i in range(1,6):
            self._model.geom("target{}".format(i)).pos = np.array([x_1, 0, 5*h*i+0.2])
            self.map.spawn_object(x_1, 0, 1.5, 2, h*i)
            x_0 = x_1
            l_0 = 0.75
            l_1 = 0.75
            x_1 = gap + l_0 + l_1 + x_0


        terrain = self.map.heights

        self._model.hfield_data += terrain.ravel()

    def ramps_terrain(self):
        self.map.reset_map()
        self.bumpy_terrain()
        self._data.qpos[2] = 2.5
        self._data.qpos[0] = -5
        self.map.spawn_object(-5, 0, 3, 3, 0.2)
        gap = 0
        x_0 = -5
        l_0 = 1.5
        l_1 = 0.75
        x_1 = gap + l_0 + l_1 + x_0
        for i in range(1,6):
            type = 0 if i%2 == 0 else 1
            sign = 1 if i % 2 == 0 else -1
            self._model.geom("target{}".format(i)).pos = np.array([x_1, sign*0.7, 5*0.25])
            self.map.add_ramp(x_1,sign*0.5,1.5,3,0.05,type, offset=0.15)
            x_0 = x_1
            l_0 = 0.75
            l_1 = 0.75
            x_1 = gap + l_0 + l_1 + x_0

        terrain = self.map.heights

        self._model.hfield_data += terrain.ravel()
    def get_info(self):
        return {"success": self.success}
    def feet_contact(self):
        FR_contact =  self._check_collision("floor", "foot_FR")
        FL_contact = self._check_collision("floor", "foot_FL")
        RR_contact =self._check_collision("floor", "foot_RR")
        RL_contact = self._check_collision("floor", "foot_RL")
        return np.array([FR_contact,FL_contact,RR_contact,RL_contact])

    def setup(self, obs):
        self.scandots_buffer.reset()
        self.air_time_reward = np.zeros(4)
        # quat = self._data.qpos[:7]
        # print(self._data.sensor('Body_Quat').data.copy(), quat)

        self.action_buffer.reset()
        self.last_actions = list()
        self.torques = list()
        self.success = False
        self._cycles_since_last_touchdown_fr = 0
        self._cycles_since_last_touchdown_fl = 0
        self._cycles_since_last_touchdown_rr = 0
        self._cycles_since_last_touchdown_rl = 0
        # self.reached_target = 0
        self._ts = 0
        self.last_contacts = np.zeros(4)
        self.feet_air_time = np.zeros(4)

        if self.task==0 or self.task==1:
            if self.task==1:
                # self.ramps_terrain()
                self.bumpy_terrain()
                # self.gaps_terrain()
                # self.hurdles_terrain()
            # self._data.qpos[0] = 0
            # quat = self._data.qpos[3:7].copy()
            # angle = np.random.uniform(0,180)

            # self._data.qpos[3:7] = rotate_quaternion_z(quat, angle)

            x_0, y_0 = (0, 0)
            angles = [np.pi/12, np.pi/10, np.pi/8, np.pi/4, np.pi/2, np.pi]
            radius = 8
            angle_limit = angles[self.level]
            angle = np.random.uniform(-angle_limit, angle_limit)
            x = x_0 + radius * np.cos(angle)
            y = y_0 + radius * np.sin(angle)
            x_px = int((self.size_map_px / self.size_map) * x + self.size_map_px // 2)
            y_px = int((self.size_map_px / self.size_map) * y + self.size_map_px // 2)
            h = self._model.hfield_data.reshape((self.size_map_px, self.size_map_px))[x_px, y_px]

            self._model.geom("target").pos = np.array([x, y, h + 0.3])

            # for i in range(1, 6):
            #     radius = np.random.uniform(2, 3.5)
            #     factor = 4
            #     angle = np.pi / factor
            #     angles = np.random.uniform(-angle, angle)
            #
            #     x = x_0 + radius * np.cos(angles)
            #     y = y_0 + radius * np.sin(angles)
            #
            #     target_pos = (x, y)
            #     x_px = int((self.size_map_px / self.size_map) * x + self.size_map_px // 2)
            #     y_px = int((self.size_map_px / self.size_map) * y + self.size_map_px // 2)
            #
            #     h = self._model.hfield_data.reshape((self.size_map_px, self.size_map_px))[x_px, y_px]
            #     self._model.geom("target{}".format(i)).pos = np.array([target_pos[0], target_pos[1], h + 0.1])
            #     x_0, y_0 = target_pos

        if self.task==2:
            self.gaps_terrain()

        if self.task==3:
            self.hurdles_terrain()

        if self.task==4:
            self.steps_terrain()

        if self.task==5:
            self.ramps_terrain()

        # hfiield = self._model.hfield_data.copy().reshape((self.size_map_px,self.size_map_px))
        # min_val = np.min(hfiield)
        # max_val = np.max(hfiield)
        # normalized_matrix = ((hfiield - min_val) / (max_val - min_val))*255
        # edges = cv2.Canny(np.uint8(normalized_matrix), threshold1=100, threshold2=200)
        # kernel = np.ones((1, 1), np.uint8)
        # self._edge_mask = cv2.erode(edges, kernel, iterations=1)


        for i in TARGET_IDS:
            self._model.geom(i).rgba = [.3, 0, 0, 1]
        self.target_idx = 1
        self.target_id = "target"


        if self._reset_state is None:
            for i in range(0, 1000):

                a = self.default_dof_pos.copy()
                torques = self.controller(a)
                self._data.ctrl = torques
                mujoco.mj_step(self._model, self._data)


            self._reset_state = dict(q_pos=self._data.qpos.copy(), q_vel=self._data.qvel.copy())
        else:
            self._data.qpos = self._reset_state["q_pos"]
            self._data.qvel = self._reset_state["q_vel"]

        self.last_dof_vel = self._reset_state["q_vel"][6:]
        self.last_dof_pos = self._reset_state["q_pos"][7:]
        torques = self.controller(self.default_dof_pos)
        self.torques.append(torques.copy())
        self.last_actions.append(self.default_dof_pos.copy())
        self.action_buffer.append(self.default_dof_pos.copy())

        target_pos = self._model.geom(self.target_id).pos.copy()[:2]
        robot_pos = self._data.body("trunk").xpos.copy()[:2]
        vect_direction = target_pos - robot_pos
        norm = np.linalg.norm(vect_direction)
        target_vec_norm = vect_direction / (norm + 1e-5)
        target_yaw = np.arctan2(target_vec_norm[1], target_vec_norm[0])
        _, _, yaw = self.get_roll_pitch_yaw_radians()

        self.yaw = yaw
        self.target_yaw = target_yaw
        for i in range(5):
            self.scandots_buffer.append(self.sample_terrain())
