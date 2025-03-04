from numpy.ma.core import arctan2

from src.envs.mujoco_env import MuJoCo, ObservationType, mujoco
from scipy.spatial.transform import Rotation as R
from src.envs.env_utils.utils import *



task_map = dict(nominal=0,
                navigation=1,
                obstacles_navigation=2,
                multi_obstacles_navigation=3,
                maze_evaluation=4)

TARGET_IDS = ["target{}".format(i) for i in range(1, 5)]
TARGET_IDS = ["target"]





class Go1(MuJoCo):
    """
    Mujoco GO1

    """

    def __init__(self, horizon,
                 gamma,
                 task_name="nominal",
                 timestep=0.002, cfg=None, **kwargs):
        """
        Constructor.

        """
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
        self.domain_randomization = cfg["domain_randomization"]
        self.static_obstacles = cfg["static_obstacles"]
        self.obstacle_id = 0
        self._room_width = 10
        self.curriculum = cfg["curriculum"]
        self.resample = cfg["resample"]
        self.joystick = cfg["joystick"]
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


        timestep = timestep



            # else:
        self.xml_path = "src/envs/unitree_go1/navigation_go1_rays.xml"

        self.joint_limits = np.array([[-0.802851, 0.802851], [-1.0472, 4.18879], [-2.69653, -0.916298]] * 4)

        action_spec = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh",
                       "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]

        self.torques_limits = np.array(4 * [23.7, 23.7, 35.55])
        self._reset_state = None
        self._ts = 0
        self.air_time_max = 0.5



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
                            position_obstacle +  # (2,)
                            velocity_obstacle +  # (2,)
                            rays)  # (50,)



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

        self.last_action = np.array(12 * [0])
        self.torques = list()

        self.foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self._max_torques = np.array([23.7, 23.7, 35.55])
        self._min_torques = np.array([-23.7, -23.7, -35.55])

        self.hip_indices = [0, 3, 6, 9]

        self.sitting_dof_pos = np.array([0.0 / 180 * np.pi, 70.0 / 180 * np.pi, -150.0 / 180 * np.pi] * 4)

        self.yaw = None
        self.last_contacts = np.zeros(4)
        self.feet_air_time = np.zeros(4)
        self.np_rng = np.random.default_rng(seed=0)

        self.default_trunk_mass = self._model.body_mass[1]
        self.default_trunk_inertia = self._model.body_inertia[1]
        #
        #
        # self.use_mujoco_renderer = True
        # self.mujoco_options = mujoco.MjvOption()
        # mujoco.mjv_defaultOption(self.mujoco_options)
        # self.mujoco_options.sitegroup[0] = False
        # self.mujoco_options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        # self.renderer_main_camera = None
        # self.renderer_depth = None

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
        pos_diff = self._data.body("obstacle_1").xpos[:3] - self._data.body("trunk").xpos[:3]
        relative_obstacle_pos = self.orientation_quat_inv.apply(pos_diff)
        vel_diff = self._data.qvel[:3] - self._data.joint("obstacle").qvel[:3]
        relative_obstacle_vel = self.orientation_quat_inv.apply(vel_diff)

        obs = self.obs_helper.modify_obs(obs, "obstacle_pose", relative_obstacle_pos[:2])
        obs = self.obs_helper.modify_obs(obs, "obstacle_velocity", relative_obstacle_vel[:2])


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

    def obs_antagonist(self):
        # pos_antagonist = self._data.body("obstacle_1").xpos[:2]
        pos_robot = self._data.body("trunk").xpos[:2] - self._data.body("obstacle_1").xpos[:2]
        velocity_robot = self._data.qvel[:2]
        velocity_obstacle = np.array([self._data.joint("obstacle").qvel[1]])
        distance = self._data.site("target").xpos[0] - self._data.qpos[0]
        pass_finish_line = np.array([distance < -0.35])
                                # 2, 2, 1, 1
        return np.concatenate((pos_robot, velocity_robot, velocity_obstacle, pass_finish_line))

    def get_reward(self, action):


        yaw = self.orientation_euler[2]
        global_vel = self._data.qvel[:2]
        y_pos = self._data.body("trunk").xpos[1]
        desired_yaw = 0  # Assuming 0 is the forward direction
        yaw_deviation_penalty = np.abs(yaw - desired_yaw)
        distance = self._data.site("target").xpos[0] - self._data.qpos[0]

        distance_to_obstacle = np.linalg.norm(self._data.site("obstacle_1").xpos[:2] - self._data.qpos[:2])

        soft_hit_condition = self._model.site("obstacle_1").size[0] + self._model.site("robot_contact_limit").size[
            0]
        if self.narrow_passage:
            proximity_penalty = 0
        else:
            proximity_penalty = (distance_to_obstacle < soft_hit_condition) * np.exp(1 / distance_to_obstacle)

        penalties = 1.*yaw_deviation_penalty + (np.abs(y_pos)) * (distance < -0.35) + proximity_penalty * (
                    not self.narrow_passage and not self.no_walls) + self.collision_penalty()


        reward = 2.5 * np.clip(min(1, global_vel[0]), 0, 1)


        # local_vel = self.orientation_quat_inv.apply(self._data.qvel[:3].copy())[:2]
        # local_accelation = (local_vel - self.previous_vel[:2])/self.dt
        # angular_velocity = self._data.qvel[5]
        # angular_aceleration = (angular_velocity- self.previous_angular_vel)/self.dt
        # local_vel_rate = 0.*np.sum(np.square(local_vel))
        # local_accelation_rate = 0.01*np.sum(np.square(local_accelation))
        # angular_aceleration_rate = 0.005*(angular_aceleration**2)
        # regularization = local_vel_rate + local_accelation_rate + angular_aceleration_rate
        #
        #     # print(local_vel_rate, local_accelation_rate, angular_aceleration_rate)

        return reward - penalties

    def reward(self, state, action, next_state, absorbing):

        reward = self.get_reward(action)

        return reward

    def get_student_rays(self):
        rf_readings = np.array([self._data.sensor("rf_l_{}".format(i + 1)).data.copy()[0] for i in range(10)])
        rf_readings = np.where(rf_readings == -1, 20, rf_readings)
        rf_readings = np.where(rf_readings == 0, rf_readings + 0.01, rf_readings)
        rf_readings_2 = np.where(rf_readings >= 20, 20, rf_readings)
        rf_readings_f = np.log2(rf_readings_2)
        obs = rf_readings_f
        return obs


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

        self._ts += 1
        self.update_orientation_attributes()

        # if self.task == 3:
        #
        #     if not self.static_obstacles:
        #         for object_id in range(self.number_obstacles):
        #             obstacle_pose = self._data.body(f"n-obstacle{object_id}").xpos[1]
        #
        #             if obstacle_pose >= self.wall_pos - 2 * self.obstacle_sizes[object_id]:
        #                 action = - np.random.uniform(0.1, 1)
        #                 self.set_control_obstacle(object_id, action)
        #
        #             if obstacle_pose <= -self.wall_pos + 2 * self.obstacle_sizes[object_id]:
        #                 action = np.random.uniform(0.1, 1)
        #
        #                 self.set_control_obstacle(object_id, action)

        if self.task == 2:
            if not self.static_obstacles:
                if not self.antagonist:
                    obstacle_pose = self._data.body("obstacle_1").xpos[1]
                    robot_pose = self._data.body("trunk").xpos[1]
                    self._data.ctrl[-3] = self.obstacle_pose_x
                    self._data.ctrl[-2] = 0
                    self._data.ctrl[-1] = np.clip((robot_pose - obstacle_pose), -1, 1)

        if self.task == 3 or self.task == 2:
            robot_pose = self._data.qpos[:2]
            target_pose = self._data.site("target").xpos[:2]
            norm_distance = target_pose[0] - robot_pose[0]
            self.pass_finish_line = norm_distance <= 0.1

            if self._ts >= self.horizon:
                self.success = self.robot_collisions == 0 and self.pass_finish_line

        if self.joystick:
            vx, vy, wz = keyboard_control(self.target_vx, self.target_vy, self.target_yaw)

            self.target_vx = vx
            self.target_vy = vy
            self.target_yaw = wz


        height = self._data.qpos[2]
        fall = self.check_collision("floor", "trunk_t") or height <= 0.1

        # if self.task > 1:
        legs_collision_obstacles = self._check_collision("legs", "obstacles")
        trunk_collision_obstacles = self._check_collision("trunk", "obstacles")

        self.obstacle_robot_collision = legs_collision_obstacles + trunk_collision_obstacles
        collision_robot_wall = self._check_collision("robot", "walls")
        collision_robot = legs_collision_obstacles or trunk_collision_obstacles or collision_robot_wall

        if fall | collision_robot:
            self.terminates = True

    def reward_antagonist(self):

        # Reward Antagonist
        distance = self._data.site("target").xpos[0] - self._data.qpos[0]
        pass_finish_line = distance < -0.35

        pos_antagonist = self._data.body("obstacle_1").xpos[:2]

        pos_robot = self._data.body("trunk").xpos[:2]

        reward_collision = 10 * self.obstacle_robot_collision
        reward_distance = -np.abs(pos_robot[1] - pos_antagonist[1])
        reward_antagonist = (reward_collision + reward_distance) * ( not pass_finish_line) +  -1*(pass_finish_line)
        return reward_antagonist

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

        if self._reset_state is None:
            if self.task == 2 or self.task == 3 or self.task == 4:
                self.reset_states(self.task)

            for i in range(0, 10):

                a = self.default_dof_pos.copy() + np.zeros(12)
                torques = self.controller_pd.action2torques(a)
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

        self.last_action = np.zeros(12)
        self.previous_vel = np.zeros(3)
        self.previous_angular_vel = 0
        self.last_velocity_command = np.array([self.target_vx, self.target_vy, self.target_yaw])


    def reset_states(self, task):
        self._model.site("target").pos[0] = 10


        if task == 2:

            spec = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/navigation_go1_rays.xml")

            geoms = spec.worldbody.geoms
            walls = ["wall_3", "wall_4"]
            # self._room_width = np.random.uniform(low=1.2, high=6)
            self._room_width = 3
            self._obstacle_l = 0.5
            self._obstacle_w = 0.5

            # if self._room_width > 2.0:
            #     self.narrow_passage = False
            #
            #     self._room_width= np.random.uniform(0.1, 0.6)
            #     self._obstacle_w = np.random.uniform(0.1, 0.6)
            # else:
            #     self.narrow_passage = True
            #     self._obstacle_l = np.random.uniform(0.05, 0.25)
            #     self._obstacle_w = np.random.uniform(0.05, 0.25)

            self.wall_pos = self._room_width / 2


            self.no_walls = np.random.uniform() < 0.5
            self.no_walls = False
            if self.no_walls:
                z_wall = -2
            else:
                z_wall = 0.8
            for geom in geoms:
                if geom.name in walls:
                    if geom.pos[1] < 0:
                        geom.pos[1] = -self.wall_pos - geom.size[1] / 2
                    else:
                        geom.pos[1] = self.wall_pos + geom.size[1] / 2
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
            # target_heading = np.random.uniform(-np.pi / 4, np.pi / 4)
            # goal_position_x = np.random.uniform(6, 20)
            # if self.no_walls:
            #     goal_position_y = np.random.uniform(-3, 3)
            # else:
            #     goal_position_y = np.random.uniform(-self.wall_pos + 0.5, self.wall_pos - 0.5)
            # goal_site = spec.worldbody.add_site(name="goal",
            #                                     type=mujoco.mjtGeom.mjGEOM_BOX,
            #                                     size=[0.4, 0.2, 0.28],
            #                                     rgba=[1, 1, 1, 0.2],
            #                                     pos=[goal_position_x, goal_position_y, 0.28],
            #                                     euler=[0, 0, target_heading])

            # self._model.site("obstacle_1").size = max(self._obstacle_l , self._obstacle_w )
            model = spec.compile()
            self._model = model

            self._model.site("obstacle_1").size[0] = 1.1 * np.sqrt((self._obstacle_l) ** 2 + (self._obstacle_w) ** 2)
            y_robot = np.random.uniform(-self.wall_pos + 0.3, self.wall_pos - 0.3)
            y_obstacle = np.random.uniform(-self.wall_pos + self._obstacle_w, self.wall_pos - self._obstacle_w)
            self._model.opt.timestep = 0.005
            self._data = mujoco.MjData(model)
            self._data.qpos[7:19] = self.default_dof_pos.copy()
            x = np.random.uniform(-1.5, 1.5)
            self._data.qpos[0:3] = [x, y_robot, 0.36]
            self.obstacle_pose_x = x + np.random.uniform(2, 3.5)




            self._data.joint("obstacle").qpos[:3] = [self.obstacle_pose_x, y_obstacle, obstacle_height]
            self._model.site("target").pos[0] = self.obstacle_pose_x + 1
            self._model.site("target").size[1] = self.wall_pos
            self._model.site("target").pos[-1] = 0.1
            if self._viewer is not None:
                self._viewer.load_new_model(self._model)


            if self.task == 2:
                if not self.antagonist:
                    if not self.static_obstacles:
                        self._data.ctrl[-1] = np.random.uniform(-1, 1)
                    else:
                        self._data.ctrl[-1] = 0

                else:
                    self._data.ctrl[-1] = self.obstacle_velocity

                self._data.ctrl[-3] = self.obstacle_pose_x
                self._data.ctrl[-2] = 0

        # if task == 3:
        #     spec = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/navigation_go1_rays_corridor.xml")
        #     obstacle = mujoco.MjSpec.from_file(filename="src/envs/unitree_go1/obstacle.xml")
        #
        #     geoms = spec.worldbody.geoms
        #     walls = ["wall_3", "wall_4"]
        #     self._room_width = np.random.uniform(low=2, high=5)
        #
        #     self.wall_pos = self._room_width / 2
        #
        #     for geom in geoms:
        #         if geom.name in walls:
        #             if geom.pos[1] < 0:
        #                 geom.pos[1] = -self.wall_pos - geom.size[1] / 2
        #             else:
        #                 geom.pos[1] = self.wall_pos + geom.size[1] / 2
        #
        #     while obstacle.keys:
        #         obstacle.keys[-1].delete()
        #     n_obstacles = []
        #
        #     self.wall_pos = self._room_width / 2
        #     self.obstacle_pos_x_list = []
        #     obstacle_pos_y_list = []
        #     self.obstacle_sizes = []
        #
        #     for i in range(self.number_obstacles):
        #         obstacle_l = np.random.uniform(0.1, 0.6)
        #         obstacle_w = np.random.uniform(0.1, 0.6)
        #         self.obstacle_sizes.append(obstacle_w)
        #         y_obstacle = np.random.uniform(-self.wall_pos + obstacle_w, self.wall_pos - obstacle_w)
        #
        #         obstacle.find_body('obstacle').first_geom().size = [obstacle_l, obstacle_w, 0.8]
        #
        #         frame = spec.worldbody.add_frame(pos=[i * 3, y_obstacle, 0.8])
        #         frame.attach_body(obstacle.find_body('obstacle'), 'n-', str(i))
        #
        #         spec.actuators[-2].refsite = "reference"
        #         spec.actuators[-3].refsite = "reference"
        #
        #         n_obstacles.append(spec.find_body('n-obstacle{}'.format(i)).first_geom().name)
        #         self.obstacle_pose_x = i * 3
        #         self.obstacle_pos_x_list.append(self.obstacle_pose_x)
        #         obstacle_pos_y_list.append(y_obstacle)
        #
        #     model = spec.compile()
        #
        #     self._model = model
        #     self._model.opt.timestep = 0.005
        #     self._data = mujoco.MjData(model)
        #     self._model.site("target").pos[0] = self.obstacle_pose_x + 1
        #     self._model.site("target").size[1] = self.wall_pos
        #
        #     if self._viewer is not None:
        #         self._viewer.load_new_model(self._model)
        #
        #     collision_obstacle = ("obstacles", n_obstacles)
        #     self.collision_groups_list.extend([collision_obstacle])
        #     self.collision_groups = self.create_collision_groups(self.collision_groups_list)
        #
        #     self.set_control_obstacle(static=self.static_obstacles)
        #
        #     #
        #     # self._model.site("target").pos[0]=40

    # def set_control_obstacle(self, object_id=None, control_value=None, static=False):
    #
    #     def initialize_controller(static):
    #         for i in range(self.number_obstacles):
    #             j = i * 3
    #             slide_0 = 12 + j
    #             slide_1 = 15 + j
    #             if static:
    #                 action = 0
    #             else:
    #                 velocity = np.random.uniform(0.1, 1)
    #                 action = np.random.uniform(-velocity, velocity)
    #
    #             self._data.ctrl[slide_0:slide_1] = [self.obstacle_pos_x_list[i],
    #                                                 0,
    #                                                 action]
    #
    #     def set_control_2_obstacle(object_id, control_value):
    #         j = object_id * 3
    #         idx = 14 + j
    #         self._data.ctrl[idx] = control_value
    #
    #     if object_id is None:
    #         initialize_controller(static)
    #     else:
    #         set_control_2_obstacle(object_id, control_value)