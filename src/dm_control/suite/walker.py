# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Planar Walker Domain."""

import collections
from dm_control.mujoco.wrapper import mjbindings

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import matplotlib.pyplot as plt
import numpy as np
from dm_control import mjcf

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8


class HeightMap:
    def __init__(self, width_m, length_m, resolution_cm):
        self.resolution_cm = resolution_cm
        self.width = int(width_m * 100 / resolution_cm)  # Convert meters to centimeters
        self.length = int(length_m * 100 / resolution_cm)
        self.heights = np.zeros((self.length, self.width))  # Initialize height map with zeros
        self.center_x = self.width // 2
        self.center_y = self.length // 2

    def reset_map(self):
        self.heights = np.zeros((self.length, self.width))

    def spawn_object(self, x_m, y_m, l_m, w_m, h_m):
        # Convert object positions and dimensions from meters to centimeters
        x_cm = int(x_m * 100 / self.resolution_cm)
        y_cm = int(y_m * 100 / self.resolution_cm)
        l_cm = int(l_m * 100 / self.resolution_cm)
        w_cm = int(w_m * 100 / self.resolution_cm)
        h_cm = int(h_m * 100 / self.resolution_cm)

        # Calculate the adjusted position relative to the center
        x_adj = self.center_x + x_cm - l_cm // 2
        y_adj = self.center_y + y_cm - w_cm // 2

        # Check if object is within the bounds of the height map
        if x_adj + l_cm > self.width or y_adj + w_cm > self.length:
            raise EnvironmentError("Object is out of bounds!")

        # Add height of object to the corresponding region in the height map
        self.heights[y_adj:y_adj + w_cm, x_adj:x_adj + l_cm] = h_cm

    def plot_map(self):
        # Calculate x and y ticks centered around the new reference frame
        y_ticks = np.linspace(-self.width / 2 * self.resolution_cm / 100, self.width / 2 * self.resolution_cm / 100,
                              num=5)
        x_ticks = np.linspace(-self.length / 2 * self.resolution_cm / 100, self.length / 2 * self.resolution_cm / 100,
                              num=5)

        plt.imshow(self.heights.T, cmap='terrain', origin='lower', interpolation='nearest')
        plt.colorbar(label='Height (cm)')
        plt.xticks(np.arange(0, self.length + 1, self.length // 4), labels=x_ticks.round(2))
        plt.yticks(np.arange(0, self.width + 1, self.width // 4), labels=y_ticks.round(2))
        plt.xlabel('y Length (m)')
        plt.ylabel('x Width (m)')
        plt.title('Height Map with Objects')
        plt.grid(True)

    def draw_x_at_location(self, x_m, y_m):
        # Convert coordinates from meters to centimeters
        x_cm = int(x_m * 100 / self.resolution_cm)
        y_cm = int(y_m * 100 / self.resolution_cm)
        x_adj = self.center_x + x_cm
        y_adj = self.center_y + y_cm

        # Draw X mark at the specified location
        plt.scatter(y_adj, x_adj, marker='x', color='red')

    def draw_o_at_location(self, x_m, y_m):
        # Convert coordinates from meters to centimeters
        x_cm = int(x_m * 100 / self.resolution_cm)
        y_cm = int(y_m * 100 / self.resolution_cm)
        x_adj = self.center_x + x_cm
        y_adj = self.center_y + y_cm

        # Draw X mark at the specified location
        plt.scatter(y_adj, x_adj, marker='o', color='green')

    def show_plot(self):
        plt.show()

    def get_height_at_coordinate(self, x_m, y_m):
        # Convert coordinates from meters to centimeters
        x_cm = int(x_m * 100 / self.resolution_cm)
        y_cm = int(y_m * 100 / self.resolution_cm)
        x_adj = self.center_x + x_cm
        y_adj = self.center_y + y_cm
        # Check if coordinates are within the bounds of the height map
        if x_cm >= self.width or y_cm >= self.length:
            raise EnvironmentError("Coordinates are out of bounds!")

        # Retrieve the height at the specified coordinates
        return self.heights[
            y_adj, x_adj] / 100  # Convert back to meters in the plot method, rotate the axis so x is the vertical axis and y the horizontal


SUITE = containers.TaggedTasks()


def get_model_and_assets(model_path=None):
    """Returns a tuple containing the model XML string and a dict of assets."""
    if model_path is not None:
        return common.read_model(model_path), common.ASSETS

    return common.read_model('walker.xml'), common.ASSETS


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def hurdles_0(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(
        *get_model_and_assets("/home/jose-luis/Project/residual_rl/src/dm_control/suite/walker_obstacle.xml"))
    task = PlanarWalker(move_speed=_WALK_SPEED, random=random, hurdles=True, level=0)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add('benchmarking')
def stairs(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    path ="/home/jose-luis/Project/residual_rl/src/dm_control/suite/walker_stairs.xml"
    physics = Physics.from_xml_string(
        *get_model_and_assets(path))
    task = PlanarWalker(move_speed=_WALK_SPEED, random=random, hurdles=False, stairs=True, path=path, level=0)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)




@SUITE.add('benchmarking')
def hurdles(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    path = "/home/jose-luis/Project/residual_rl/src/dm_control/suite/walker_obstacle_level_1.xml"
    physics = Physics.from_xml_string(*get_model_and_assets(path))
    task = PlanarWalker(move_speed=_WALK_SPEED, random=random, hurdles=True, path=path, level=1)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

#
# @SUITE.add('benchmarking')
# def jump_0(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#     """Returns the Walk task."""
#     physics = Physics.from_xml_string(
#         *get_model_and_assets("/src/dm_control/suite/walker_jump.xml"))
#     task = PlanarWalker(move_speed=_WALK_SPEED, random=random, jump=True)
#     environment_kwargs = environment_kwargs or {}
#     return control.Environment(
#         physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
#         **environment_kwargs)


@SUITE.add('benchmarking')
def jump(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    path = "/home/jose-luis/Project/residual_rl/src/dm_control/suite/walker_jump.xml"

    physics = Physics.from_xml_string(*get_model_and_assets(path))
    task = PlanarWalker(move_speed=4, random=random, jump=True, path=path)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

#
# @SUITE.add('benchmarking')
# def jump_2(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#     """Returns the Walk task."""
#     physics = Physics.from_xml_string(
#         *get_model_and_assets("/home/jose-luis/Project/residual_rl/src/dm_control/suite/walker_jump_level_2_.xml"))
#     task = PlanarWalker(move_speed=_WALK_SPEED, random=random, jump=True)
#     environment_kwargs = environment_kwargs or {}
#     return control.Environment(
#         physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
#         **environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(
        *get_model_and_assets("/home/jose-luis/Project/residual_rl/src/dm_control/suite/walker.xml"))
    task = PlanarWalker(move_speed=2, random=random, run=True)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def control_signal(self):
        return self.data.ctrl.copy()

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['torso', 'zz']

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos['torso', 'z']

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def feet_height(self):
        """Returns the height of the torso."""
        return np.array([self.named.data.xpos['right_foot', 'z'].copy(), self.named.data.xpos['left_foot', 'z'].copy()])

    def torso_to_target(self):
        torso_to_target = (self.named.data.geom_xpos['target'] -
                           self.named.data.geom_xpos['torso'])

        return torso_to_target

    def torso_to_target_distance(self, norm=False):
        out = np.linalg.norm(self.torso_to_target())
        if norm:
            return out
        return self.torso_to_target()

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ['xx', 'xz']].ravel()

    def torso_pos(self):
        return self.named.data.xpos['torso', 'x'], self.named.data.xpos['torso', 'y'], self.named.data.xpos[
            'torso', 'z']


class PlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, move_speed, jump=False, hurdles=False, stairs=False,run=False, random=None, path=None, level=0):
        """Initializes an instance of `PlanarWalker`.

        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._move_speed = move_speed
        self._run = run
        self._jump = jump
        self._hurdles = hurdles
        self._stairs = stairs
        self.map = HeightMap(200, 10, 1)
        self.path = path

        self._level = level
        self.curriculum_level = 0

        self._curriculum_context = [0, 0, 0, 0, 0]  # max hight, min hight, max gap, min gap, block_thickness

        self._scandots_history = 5 * [50 * [0]]
        self.init_task = False

        # self._contacts = np.zeros(3)
        super().__init__(random=random)

    # def before_step(self, action, physics):

    #
    #
        # physics.set_control(np.zeros(6))
    #
    def after_step(self, physics):
        # physics.data.qpos[:3] = np.zeros(3)
        if self._jump:
            target_pos = np.array(self._targets_pos[self._target_id])
            current_pos = physics.torso_pos()
            distance = target_pos[0] - current_pos[0]
            if distance <= 0.2 and self._target_id <= len(self._targets_pos):
                physics.named.model.geom_rgba["target{}".format(self._target_id+1)] = [0,1,0,1]
                self._target_id += 1



    def update_platforms(self, physics, level):

        m1, m2 = get_model_and_assets(self.path)

        model = mjcf.from_xml_string(m1, assets=m2)

        min_gap = 0.3
        max_gap = np.clip(0.3 + level*0.17, a_max=1.25, a_min=None)
        min_platform_length = np.clip(1 - level*0.25, a_max=1, a_min=0.25)
        max_platform_length = 1.25
        platform_height = physics.named.model.geom_pos['box0', 'z'].copy() + physics.named.model.geom_size['box0', 'z'].copy() # platform height + thickness
        self._curriculum_context = [2 * min_platform_length, 2 * max_platform_length, min_gap, max_gap, platform_height]

        # print(self._curriculum_context)
        box_old = model.find('geom', 'box0')

        x1 = box_old.pos[0]
        s1 = box_old.size[0]
        for i in range(0, 21):

            if i >= 1:
                box = model.find('geom', 'box{}'.format(i))
                target = model.find('geom', 'target{}'.format(i))
                light = model.find('light', 'target_light{}'.format(i))
                gap = np.random.uniform(min_gap, max_gap)
                s2 = np.random.uniform(min_platform_length, max_platform_length)
                if i > 19:
                    s2 = 10
                x_pos = x1 + s1 + gap + s2
                box.pos = [x_pos, 0, 1]
                self._targets_pos.append([x_pos, 0, 2.3])
                target.pos = [x_pos, 0, 2.3]
                light.pos = [x_pos, 0, 2.3]
                box.size = [s2, 2, 0.1]
                x1 = x_pos
                s1=s2


        physics.reload_from_xml_string(model.to_xml_string())



    def update_hurdles(self, physics, level):
        m1, m2 = get_model_and_assets(self.path)

        model = mjcf.from_xml_string(m1, assets=m2)

        min_gap = 3.
        max_gap = 6.
        min_z = 0.05
        max_z = 0.1

        x1 = physics.named.model.geom_pos['box0', 'x'].copy()
        x_pos = x1
        for i in range(0, 16):

            if i >= 1:
                gap = np.random.uniform(min_gap, max_gap)

                # x1 = physics.named.model.geom_pos['box{}'.format(i - 1), 'x'].copy()
                s1 = physics.named.model.geom_size['box{}'.format(i - 1), 'x'].copy()
                s2 = physics.named.model.geom_size['box{}'.format(i), 'x'].copy()
                x_pos = x1 + s1 + gap + s2
                x1 = x_pos
            box = model.find('geom', 'box{}'.format(i))

            if level >= 1:
                min_z = 0.1
                max_z = 0.1 + 0.05*(level-1)

            z = np.random.uniform(min_z, max_z)
            box.size[2] = z
            box.pos = [x_pos, 0, z]
            # print(i, x_pos, gap)

        block_thickness = physics.named.model.geom_size['box0', 'x'].copy()
        self._curriculum_context = [2 * min_z, 2 * max_z, min_gap, max_gap, 2*block_thickness]
        physics.reload_from_xml_string(model.to_xml_string())

    def update_steps(self, physics, level):
        m1, m2 = get_model_and_assets(self.path)

        model = mjcf.from_xml_string(m1, assets=m2)

        min_step = -0.1 - level*0.05
        max_step = 0.1 + level*0.05
        max_h =2
        min_h = 0.01



        platform_length = 2 * physics.named.model.geom_size['box1', 'x'].copy()
        ref_h = physics.named.model.geom_size['box0', 'z'].copy()

        height = []
        for i in range(0, 31):
            if i >= 1:
                old_box = model.find('geom', 'box{}'.format(i-1))
                r_h = old_box.size[2]
                box = model.find('geom', 'box{}'.format(i))
                if r_h >= max_h:
                    delta_h = min_step
                elif r_h <= min_h:
                    delta_h = max_step
                else:
                    delta_h = np.random.choice([min_step, max_step])


                h_step = np.clip(r_h + delta_h, min_h, max_h)
                box.size[2] = h_step
                height.append(h_step)
                box.pos[2] = h_step


        self._curriculum_context = [2*min_h, 2*max_h, platform_length, 2*ref_h, max_step]





        physics.reload_from_xml_string(model.to_xml_string())

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.

        Args:
          physics: An instance of `Physics`.

        """
        # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        random = self.random or np.random
        self.init_task = True
        self._targets_pos = []
        self._target_id = 0
        self._scandots_history = 5 * [50 * [0]]
        hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
        slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
        ball = mjbindings.enums.mjtJoint.mjJNT_BALL

        qpos = physics.named.data.qpos

        for joint_id in range(physics.model.njnt):
            joint_name = physics.model.id2name(joint_id, 'joint')
            joint_type = physics.model.jnt_type[joint_id]
            is_limited = physics.model.jnt_limited[joint_id]
            range_min, range_max = physics.model.jnt_range[joint_id]

            if is_limited:
                if joint_type == hinge or joint_type == slide:
                    qpos[joint_name] = random.uniform(range_min, range_max)

        self.v_x = 0
        self.h = 0
        self.init_pose = physics.torso_pos()[0]

        self.height_platforms = []

        if self._jump:
            self.map.reset_map()
            level = self.curriculum_level
            self.update_platforms(physics, level=level)
            n_platforms = 21
            for i in range(n_platforms):
                pos_object = physics.named.model.geom_pos['box{}'.format(i)]
                w, l, h = physics.named.model.geom_size['box{}'.format(i)]

                z = h + pos_object[2]
                self.height_platforms.append(h + pos_object[2])

                self.map.spawn_object(pos_object[0],
                                      pos_object[1],
                                      2 * w,
                                      2 * l, z)

            #self.map.plot_map()
            #self.map.show_plot()
            # exit()


        if self._hurdles:

            self.map.reset_map()

            if self._level == 1:
                level = self.curriculum_level
                self.update_hurdles(physics, level)

            n_platforms = 16

            for i in range(n_platforms):

                pos_object = physics.named.model.geom_pos['box{}'.format(i)]
                w = physics.named.model.geom_size['box{}'.format(i)][0]
                if i >= 1:
                    gap =  physics.named.model.geom_pos['box{}'.format(i)][0] - physics.named.model.geom_pos['box{}'.format(i-1)][0] - 2*w
                    # print("platform: ", i, " pos ", pos_object[0], gap)

                w, l, h = physics.named.model.geom_size['box{}'.format(i)]

                z = 2 * h
                self.height_platforms.append(h + pos_object[2])

                self.map.spawn_object(pos_object[0],
                                      pos_object[1],
                                      2 * w,
                                      2 * l, z)
                # self.map.plot_map()
                # self.map.draw_x_at_location(pos_object[0], 0)
                # self.map.show_plot()
        if self._stairs:

            self.map.reset_map()


            level = self.curriculum_level
            self.update_steps(physics, level)

            n_platforms = 31

            for i in range(n_platforms):

                w, l, h = physics.named.model.geom_size['box{}'.format(i)]
                x, y, _ = physics.named.model.geom_pos['box{}'.format(i)]

                z = 2 * h
                self.height_platforms.append(z)

                self.map.spawn_object(x,
                                      y,
                                      2 * w,
                                      2 * l, z)
            #
            # self.map.plot_map()
            # self.map.show_plot()

        # self.t = 0
        self.torso_old_pos = physics.torso_pos()[0]
        if self._jump or self._hurdles or self._stairs:
            surfaces = ["floor"]
            boxes = ['box{}'.format(i) for i in range(n_platforms)]
            surfaces.extend(boxes)
            collision_groups = [("floor", ["floor"]),
                                ("surfaces", surfaces),
                                ("robot_legs", ["right_thigh", "left_thigh"]),
                                ("boxes", ['box{}'.format(i) for i in range(n_platforms)]),
                                ("robot_torso", ["torso"]),
                                ("robot_feet", ["right_foot", "left_foot"]),
                                ("robot_l_foot", ["left_foot"]),
                                ("robot_r_foot", ["right_foot"]),
                                ("robot", ["torso", "right_thigh", "left_thigh"])]
        else:
            collision_groups = [("floor",
                                 ["floor"]),
                                ("robot", ["torso", "right_thigh", "left_thigh"]),
                                ("robot_torso", ["torso"]),
                                ("robot_legs", ["right_thigh", "left_thigh"]),
                                ("robot_feet", ["right_foot", "left_foot"])]
        self.create_collision_groups(physics, collision_groups)

        # print([physics.model.geom(geom_name).id for geom_name in ["torso",'right_thigh','left_thigh']])
        super().initialize_episode(physics)

    def get_heights(self, min_x, max_x, robot_pose):

        sampling_array = np.zeros((50, 2))
        samples = 50
        z_samples = np.zeros(50)
        x = robot_pose
        min_value = x - min_x
        max_value = x + max_x
        sampling_array[:, 0] = np.linspace(min_value, max_value, samples)

        for i in range(50):
            x, y = sampling_array[i]
            z_samples[i] = self.map.get_height_at_coordinate(x, y)

        return z_samples

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        if self._jump or self._hurdles or self._stairs:
            x,y,_ =physics.torso_pos()
            obs['height'] = physics.torso_height() - self.map.get_height_at_coordinate(x,y)
        else:
            obs['height'] = physics.torso_height()

        obs['velocity'] = physics.velocity()

        if self._jump or self._hurdles or self._stairs:
            # if self.init_task:
            #     floor = "surfaces"
            #     foot_r = "robot_r_foot"
            #     foot_l = "robot_l_foot"
                # cr = self._check_collision(floor, foot_r, physics)
                # cl = self._check_collision(floor, foot_l, physics)
            #     obs['feet_contacts'] = np.array([cr, cl])
            #
            # else:
            #     obs['feet_contacts'] = np.zeros(2)
            # obs["feet_heights"] = physics.feet_height()

            min_x, max_x = 2, 5
            robot_pose = physics.torso_pos()[0]
            # print(self.map.get_height_at_coordinate(robot_pose, 0))
            heights_array = self.get_heights(min_x, max_x, robot_pose)
            self._scandots_history.append(heights_array)
            obs['scan_dots'] = np.array(self._scandots_history[-5:]).reshape((250,))  # 31-80, 50


            # if self.init_task:
            #     legs = "robot_legs"
            #     torso = "robot_torso"
            #
            #     boxes = "boxes"
            #     feet = "robot_feet"
            #     contact_legs = self._check_collision(floor, legs, physics)
            #     contact_torso_floor = self._check_collision(floor, torso, physics)
            #     contact_torso_boxes = self._check_collision(boxes, torso, physics)
            #     contact_feet = self._check_collision(boxes, feet, physics)
            #     contact_torso = contact_torso_floor or contact_torso_boxes
            #
            #     obs['privilege_info'] = np.array([contact_torso, contact_legs, contact_feet])
            # else:
            #     obs['privilege_info'] = np.zeros(5)

            obs['context'] = np.array(self._curriculum_context.copy())




        return obs

    #

    def reward_fall(self, delta_h):

        if delta_h[0] <= 0.3 and delta_h[1] <= 0.3:
            return True
        else:
            return False

    def get_collisions(self, physics, bodies, surface):
        collisions = []
        for body in bodies:
            col = self._check_collision(body, surface, physics)
            collisions.append(col)

        return np.any(collisions)

    def get_fall(self, physics):
        fall = False

        if physics.feet_height()[0] <= 0.9 or physics.feet_height()[1] <= 0.9:
            fall = True

        return fall



    def get_termination(self, physics):


        fall = False
        if self._jump:
            group_2 = "boxes"

            if physics.feet_height()[0] <= 0.9 or physics.feet_height()[1] <= 0.9:
                fall = True

            # feet_heigth = physics.feet_height()
            legs = "robot_legs"
            torso = "robot_torso"
            collision = self.get_collisions(physics, [legs, torso], group_2)
            # if feet_heigth[0] <= 0.9 * np.min(self.height_platforms) or feet_heigth[1] <= 0.9 * np.min(
            #         self.height_platforms):
            #     fall = True
            upside_down = physics.torso_upright() < -0.8
            if fall or collision or upside_down:
                # print("terminate")
                return 1

        elif self._hurdles:
            # group_2 = "boxes"
            legs = "robot_legs"
            torso = "robot_torso"
            floor = "floor"
            col1 = self._check_collision(floor, legs, physics)
            col2 = self._check_collision(floor, torso, physics)
            boxes = "boxes"
            col3 = self._check_collision(boxes, torso, physics)
            # feet = "robot_feet"
            # col4 = self._check_collision(boxes, feet, physics)
            collision = col1 or col2 or col3

        elif self._stairs:
            # group_2 = "boxes"
            legs = "robot_legs"
            torso = "robot_torso"
            boxes = "boxes"
            col1 = self._check_collision(boxes, legs, physics)

            col3 = self._check_collision(boxes, torso, physics)
            # feet = "robot_feet"
            # col4 = self._check_collision(boxes, feet, physics)
            collision = col1 or col3

        else:
            group_1 = "robot"
            group_2 = "floor"
            collision = self._check_collision(group_1, group_2, physics)

        if collision:
            #print("terminate")
            return 1

        else:
            return None

    def create_collision_groups(self, physics, collision_groups):
        # Pre-process the collision groups for "fast" detection of contacts
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {physics.model.geom(geom_name).id
                                               for geom_name in geom_names}

    def _check_collision(self, group1, group2, physics):
        """
        Check for collision between the specified groups.

        Args:
            group1 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor;
            group2 (string): A name referring to an entry contained in the
                collision_groups list handed to the constructor.

        Returns:
            A flag indicating whether a collision occurred between the given
            groups or not.

        """
        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        for coni in range(0, physics.data.ncon):
            con = physics.data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True
        return False

    def calculate_velocity_distance_reward(self, current_velocity, target_velocity, current_pos, target_pos):
        vect = np.array(target_pos) - current_pos
        norm = np.linalg.norm(vect)
        vect_norm = vect / (norm + 1e-5)
        vel_vect_sum = np.sum(vect_norm * current_velocity)
        r = np.min([vel_vect_sum, target_velocity]) / target_velocity
        return r


    def get_reward(self, physics):
        """Returns a reward to the agent."""
        x = physics.torso_pos()[0]

        # if self._jump or self._hurdles:
        #     if self._hurdles:
        #         z_line = self.map.get_height_at_coordinate(x, 0)
        #
        #     else:
        #         z_line = self.map.get_height_at_coordinate(x, 0)
        #         if z_line <= 0:
        #             z_line = np.min(self.height_platforms)
        #
        #     delta_h_v = physics.torso_height() - z_line
        #     h = delta_h_v

        # else:
            # h = physics.torso_height()
            # delta_h_v = physics.torso_height() - np.mean(physics.feet_height())
        # h = physics.torso_height()
        delta_h_v = physics.torso_height() - np.min(physics.feet_height())
        x, y, _ = physics.torso_pos()
        self.h = physics.torso_height() - self.map.get_height_at_coordinate(x,y)

        self.v_x = physics.horizontal_velocity()


        if self._run:
            # reward = np.clip(physics.horizontal_velocity()/self._move_speed,0,1)
            self._move_speed = 2
            # vel_reward = 10 * (physics.horizontal_velocity())
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(self._move_speed, float('inf')),
                                            margin=self._move_speed / 2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            vel_reward = (5 * move_reward + 1) / 6

        else:
            vel_reward = 10 * (physics.horizontal_velocity())

        standing_rewad = -1 * np.abs(delta_h_v - 1.2)
        torques_reward = -0.1 * np.linalg.norm(physics.control_signal())
        fall_reward = -10 * (delta_h_v < 0.3)
        upright_reward = 0.5 * physics.torso_upright()

        if self._jump:
            legs = "robot_legs"
            torso = "robot_torso"
            group_2 = "boxes"

            collision = self.get_collisions(physics, [legs, torso], group_2)
            fall = self.get_fall(physics)
            penalty_early_termination = -10 * (collision+fall)

            # target_pos = self._targets_pos[self._target_id][:2]
            # current_pos = physics.torso_pos()[:2]
            #
            # current_velocity = physics.horizontal_velocity()
            # target_velocity = self._move_speed
            # r = self.calculate_velocity_distance_reward(current_velocity, target_velocity, current_pos, target_pos)
            # vel_reward = 40 * np.clip(r,0, None)
            vel_reward = 10 * (physics.horizontal_velocity())



        elif self._stairs:
            legs = "robot_legs"
            torso = "robot_torso"
            group_2 = "boxes"

            collision = self.get_collisions(physics, [legs, torso], group_2)

            penalty_early_termination = -10 * collision

        else:

            legs = "robot_legs"
            torso = "robot_torso"
            group_2 = "floor"

            collision = self.get_collisions(physics, [legs, torso], group_2)

            penalty_early_termination = -10 * collision

        #if penalty_early_termination<0:
            # print(penalty_early_termination)

        return vel_reward + standing_rewad + fall_reward + torques_reward + upright_reward + penalty_early_termination

