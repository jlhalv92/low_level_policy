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
from scipy.signal import convolve2d


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def perlin(shape, res, tileable=(False, False), interpolant=interpolant):
  """Generate a 2D numpy array of perlin noise.

  Args:
      shape: The shape of the generated array (tuple of two ints).
          This must be a multiple of res.
      res: The number of periods of noise to generate along each
          axis (tuple of two ints). Note shape must be a multiple of
          res.
      tileable: If the noise should be tileable along each axis
          (tuple of two bools). Defaults to (False, False).
      interpolant: The interpolation function, defaults to
          t*t*t*(t*(t*6 - 15) + 10).

  Returns:
      A numpy array of shape shape with the generated noise.

  Raises:
      ValueError: If shape is not a multiple of res.
  """
  delta = (res[0] / shape[0], res[1] / shape[1])
  d = (shape[0] // res[0], shape[1] // res[1])
  grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
            .transpose(1, 2, 0) % 1
  # Gradients
  angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
  gradients = np.dstack((np.cos(angles), np.sin(angles)))
  if tileable[0]:
    gradients[-1,:] = gradients[0,:]
  if tileable[1]:
    gradients[:,-1] = gradients[:,0]
  gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
  g00 = gradients[    :-d[0],    :-d[1]]
  g10 = gradients[d[0]:     ,    :-d[1]]
  g01 = gradients[    :-d[0],d[1]:     ]
  g11 = gradients[d[0]:     ,d[1]:     ]
  # Ramps
  n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
  n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
  n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
  n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
  # Interpolation
  t = interpolant(grid)
  n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
  n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
  return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def stairs(
    spec=None,
    grid_loc=(0.0, 0.0),
    num_stairs=4,
    direction=1,
    name="stairs",
    tile_size=5.0,
    step_height=0.10,
    step_depth=0.24,
    base_z=0.0,
    base_thickness=0.10,
    floor_name="floor",
    terrain_group=1,
):
    """
    Square staircase with a central platform.

    direction = +1: ascend toward the center.
    direction = -1: descend toward the center.

    The outer border is at base_z.
    The center is at base_z + direction*num_stairs*step_height.

    Returns the MjSpec.
    """
    if spec is None:
        spec = mujoco.MjSpec()

    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1.")
    if not isinstance(num_stairs, int) or num_stairs < 0:
        raise ValueError("num_stairs must be a nonnegative integer.")
    if min(tile_size, step_depth, base_thickness) <= 0:
        raise ValueError("Dimensions must be positive.")
    if step_height < 0:
        raise ValueError("step_height must be nonnegative.")

    half = tile_size / 2
    center_half = half - num_stairs * step_depth

    if center_half <= 0:
        raise ValueError("Steps leave no space for the central platform.")

    floor = next(
        (geom for geom in spec.geoms if geom.name == floor_name),
        None,
    )

    body = spec.worldbody.add_body(
        name=name,
        pos=[float(grid_loc[0]), float(grid_loc[1]), base_z],
    )

    rgba = [0.460, 0.362, 0.216, 1.0]

    # Local elevation of the common bottom of all boxes.
    bottom = min(0.0, direction * num_stairs * step_height)
    bottom -= base_thickness

    def add_box(suffix, xy, half_xy, top):
        height = top - bottom

        geom = body.add_geom(
            name=f"{name}_{suffix}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[xy[0], xy[1], bottom + height / 2],
            size=[half_xy[0], half_xy[1], height / 2],
            rgba=rgba,
            group=terrain_group,
        )

        if floor is not None:
            for parameter in (
                "friction", "condim", "solref", "solimp",
                "solmix", "priority", "margin", "gap",
                "contype", "conaffinity",
            ):
                value = getattr(floor, parameter)
                setattr(
                    geom,
                    parameter,
                    value.copy() if isinstance(value, np.ndarray) else value,
                )

    # Each ring consists of four boxes with no overlapping volumes.
    for i in range(num_stairs):
        outer = half - i * step_depth
        inner = outer - step_depth
        offset = (outer + inner) / 2
        top = direction * i * step_height

        for label, sign in (("left", -1), ("right", 1)):
            add_box(
                f"{i}_{label}",
                [sign * offset, 0.0],
                [step_depth / 2, outer],
                top,
            )

        for label, sign in (("front", 1), ("back", -1)):
            add_box(
                f"{i}_{label}",
                [0.0, sign * offset],
                [inner, step_depth / 2],
                top,
            )

    add_box(
        "center",
        [0.0, 0.0],
        [center_half, center_half],
        direction * num_stairs * step_height,
    )

    return spec




def flat_floor(
    spec=None,
    grid_loc=(0.0, 0.0),
    name="flat",
    tile_size=(1,1),
    floor_name="floor",
    terrain_group=1,
):
    """
    Square staircase with a central platform.

    direction = +1: ascend toward the center.
    direction = -1: descend toward the center.

    The outer border is at base_z.
    The center is at base_z + direction*num_stairs*step_height.

    Returns the MjSpec.
    """
    if spec is None:
        spec = mujoco.MjSpec()



    floor = next(
        (geom for geom in spec.geoms if geom.name == floor_name),
        None,
    )

    body = spec.worldbody.add_body(
        name=name,
        pos=[float(grid_loc[0]), float(grid_loc[1]), 0],
    )

    rgba = [0.460, 0.362, 0.216, 1.0]


    def add_box(suffix, xy, half_xy, height ):


        geom = body.add_geom(
            name=f"{name}_{suffix}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[xy[0], xy[1], -height / 2],
            size=[half_xy[0] / 2, half_xy[1] / 2, height / 2],
            rgba=rgba,
            group=terrain_group,
        )

        if floor is not None:
            for parameter in (
                "friction", "condim", "solref", "solimp",
                "solmix", "priority", "margin", "gap",
                "contype", "conaffinity",
            ):
                value = getattr(floor, parameter)
                setattr(
                    geom,
                    parameter,
                    value.copy() if isinstance(value, np.ndarray) else value,
                )


    add_box(
        "center",
        [0.0, 0.0],
        tile_size,
        0.1,
    )

    return spec

def edge_slope(size, border_width=5, blur_iterations=20):
  """Creates a grayscale image with a white center and fading black edges using convolution."""
  img = np.ones((size, size), dtype=np.float32)
  img[:border_width, :] = 0
  img[-border_width:, :] = 0
  img[:, :border_width] = 0
  img[:, -border_width:] = 0

  kernel = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]) / 9.0

  for _ in range(blur_iterations):
    img = convolve2d(img, kernel, mode='same', boundary='symm')

  return img


def add_height_sampling_sites(
    spec,
    body_name="trunk",
    foot_plane_z=-0.34,
    prefix="height_sample",
):
    """
    Create a 17-by-11 grid attached to the robot body.

    foot_plane_z: height relative to the trunk frame, in metres.
    Returns site names arranged as a (17, 11) matrix.
    """
    body = next(
        (body for body in spec.bodies if body.name == body_name),
        None,
    )
    if body is None:
        raise ValueError(f"Body '{body_name}' not found.")

    x_points = np.linspace(-0.8, 0.8, 17)
    y_points = np.linspace(-0.5, 0.5, 11)

    site_names = np.empty(
        (len(x_points), len(y_points)),
        dtype=object,
    )

    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            name = f"{prefix}_{i}_{j}"

            body.add_site(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[float(x), float(y), foot_plane_z],
                size=[0.012, 0.012, 0.012],
                rgba=[0.0, 1.0, 0.0, 1.0],
                group=0,
            )

            site_names[i, j] = name

    return site_names


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




class TerrainHeightSampler:
    def __init__(
        self,
        model,
        site_ids,
        base_body_name="trunk",
        nominal_height=0.34,
        terrain_group=5,
        ray_top_z=3.0,
        max_distance=6.0,
        marker_offset=0.012,
    ):
        """
        site_ids:
            (17, 11) matrix of sites attached directly to the base body.
        ray_top_z:
            World height of ray origins; must exceed all terrain heights.
        marker_offset:
            Visual offset above the measured surface, in metres.
        """
        self.model = model
        self.site_ids = np.asarray(site_ids, dtype=np.int32)
        self.grid_shape = (17, 11)

        if self.site_ids.shape != self.grid_shape:
            raise ValueError("site_ids must have shape (17, 11).")
        if np.any(
            (self.site_ids < 0) | (self.site_ids >= model.nsite)
        ):
            raise ValueError("Invalid site ID.")
        if not 0 <= terrain_group < 6:
            raise ValueError("terrain_group must be between 0 and 5.")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive.")

        self.flat_site_ids = self.site_ids.ravel()

        self.base_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            base_body_name,
        )
        if self.base_body_id < 0:
            raise ValueError(f"Body '{base_body_name}' not found.")

        if not np.all(
            model.site_bodyid[self.flat_site_ids] == self.base_body_id
        ):
            raise ValueError(
                "Sampling sites must be attached directly to the base body."
            )

        self.nominal_height = float(nominal_height)
        self.ray_top_z = float(ray_top_z)
        self.max_distance = float(max_distance)
        self.marker_offset = float(marker_offset)

        # Ordering matches add_height_sampling_sites().
        x = np.linspace(-0.8, 0.8, 17)
        y = np.linspace(-0.5, 0.5, 11)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        self.local_xy = np.column_stack((xx.ravel(), yy.ravel()))

        # Include terrain only; exclude robot and old floor.
        self.geomgroup = np.zeros(6, dtype=np.uint8)
        self.geomgroup[terrain_group] = 1
        self.geomgroup[0]=1

        self.direction = np.array([0.0, 0.0, -1.0])
        self.hit_id = np.empty(1, dtype=np.int32)

        self.heights = np.full(self.grid_shape, np.nan)
        self.valid = np.zeros(self.grid_shape, dtype=bool)
        self.hit_geom_ids = np.full(
            self.grid_shape, -1, dtype=np.int32
        )
        self.sample_xy = np.zeros((*self.grid_shape, 2))

    def update(self, data):
        """Sample terrain and move markers. Return (heights, valid)."""
        mujoco.mj_kinematics(self.model, data)

        base_pos = data.xpos[self.base_body_id].copy()
        base_rot = data.xmat[self.base_body_id].reshape(3, 3).copy()

        # Rotate the sampling footprint with yaw only.
        yaw = np.arctan2(base_rot[1, 0], base_rot[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.array([[c, -s], [s, c]])

        world_xy = self.local_xy @ yaw_rotation.T + base_pos[:2]
        self.sample_xy[:] = world_xy.reshape(*self.grid_shape, 2)

        heights = self.heights.ravel()
        valid = self.valid.ravel()
        hit_geom_ids = self.hit_geom_ids.ravel()

        heights.fill(np.nan)
        valid.fill(False)
        hit_geom_ids.fill(-1)

        origin = np.array([0.0, 0.0, self.ray_top_z])

        for i, xy in enumerate(world_xy):
            origin[:2] = xy

            distance = mujoco.mj_ray(
                self.model,
                data,
                origin,
                self.direction,
                self.geomgroup,
                True,  # Include static geoms.
                -1,    # Robot excluded through geom groups.
                self.hit_id,
            )

            if 0.0 <= distance <= self.max_distance:
                heights[i] = self.ray_top_z - distance
                valid[i] = True
                hit_geom_ids[i] = self.hit_id[0]

        # Place valid markers on terrain.
        # Missing hits remain at nominal foot height.
        marker_z = np.full(
            len(heights),
            base_pos[2] - self.nominal_height,
        )
        marker_z[valid] = heights[valid] + self.marker_offset

        world_points = np.column_stack((world_xy, marker_z))

        # Convert desired world positions to trunk-local coordinates.
        self.model.site_pos[self.flat_site_ids] = (
            world_points - base_pos
        ) @ base_rot

        self.model.site_rgba[self.flat_site_ids[valid]] = [
            0.0, 1.0, 0.0, 1.0
        ]
        self.model.site_rgba[self.flat_site_ids[~valid]] = [
            1.0, 0.0, 0.0, 1.0
        ]

        mujoco.mj_kinematics(self.model, data)

        return self.heights.copy(), self.valid.copy()


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
        self.final_level = False
        self.current_level = 0
        self.env_id = env_id
        print("Created env {}".format(self.env_id))
        self._n_episode = 0
        self.episode_step = 0
        self.resampling_command_steps = 500 # 500
        self.joystick = joystick
        self.env_args = env_args
        self.tracking_velocities_array = []
        self.track_velocities = False
        self.heights_obs = np.zeros(187)
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
        self.initial_drop_height = 0.30 + self.current_level*0.1 #0.325 0.34, 0.30

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
        # spec = mujoco.MjSpec.from_file(filename=xml_path)
        # SQUARE_LENGTH = 5
        # HEIGHT = 0.5
        # BROWN_RGBA = [0.460, 0.362, 0.216, 1.0]
        #
        #
        #
        # size = 128
        # noise = perlin((size, size), (8, 8))
        #
        # # Remap noise to 0 to 1
        # noise = (noise + 1) / 2
        # noise -= np.min(noise)
        # noise /= np.max(noise)
        #
        # # Makes the edges slope down to avoid sharp boundary
        # noise *= edge_slope(size)
        #
        # # Create height field
        # hfield = spec.add_hfield(name="hfield",
        #                          size=[SQUARE_LENGTH, SQUARE_LENGTH,
        #                                HEIGHT, HEIGHT / 10],
        #                          nrow=noise.shape[0],
        #                          ncol=noise.shape[1],
        #                          userdata=noise.flatten())
        # grid_loc = [6, 0]
        # body = spec.worldbody.add_body(pos=grid_loc + [0], name="hfield")
        # body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="hfield",
        #               rgba=BROWN_RGBA)
        #
        # # xml_path = "src/envs/unitree_go1/navigation_go1_rays_test.xml"
        #
        # # self.model = mujoco.MjModel.from_xml_path(xml_path)
        # model = spec.compile()




        # SQUARE_LENGTH = 20.0
        # HEIGHT = 2.
        # BROWN_RGBA = [0.460, 0.362, 0.216, 1.0]
        #
        # # Constant zero elevations produce a flat surface.
        # # A 2-by-2 grid is sufficient for a flat heightfield.
        # size = 50
        # elevation = np.zeros((size, size), dtype=np.float32)
        #
        # spec.add_hfield(
        #     name="flat_hfield",
        #     size=[
        #         SQUARE_LENGTH / 2,
        #         SQUARE_LENGTH / 2,
        #         HEIGHT,
        #         HEIGHT / 10,
        #     ],
        #     nrow=size,
        #     ncol=size,
        #     userdata=elevation.ravel(),
        # )
        #
        # body = spec.worldbody.add_body(
        #     name="flat_terrain",
        #     pos=[0.0, 0.0, 0.0],
        # )
        #
        # # Find the existing floor.
        # floor = next(
        #     (geom for geom in spec.geoms if geom.name == "floor"),
        #     None,
        # )
        #
        # if floor is None:
        #     available_names = [geom.name for geom in spec.geoms]
        #     raise ValueError(
        #         f"Geom 'floor' not found. Available geoms: {available_names}"
        #     )
        # if floor is None:
        #     raise ValueError("Floor geom not found. Check its name in the XML.")
        #
        # terrain_geom = body.add_geom(
        #     name="flat_terrain_geom",
        #     type=mujoco.mjtGeom.mjGEOM_HFIELD,
        #     hfieldname="flat_hfield",
        #     rgba=BROWN_RGBA,
        # )
        #
        #
        #
        # terrain_geom.solref = floor.solref.copy()
        # terrain_geom.solimp = floor.solimp.copy()
        # terrain_geom.friction = floor.friction.copy()
        # terrain_geom.priority = floor.priority
        # terrain_geom.solmix = floor.solmix
        # terrain_geom.condim = floor.condim
        #
        # foot_names = {"FR_foot", "FL_foot", "RR_foot", "RL_foot"}
        #
        # # At physics timestep 0.005 s, use timeconst >= 0.010 s.
        # contact_solref = [0.010, 1.0]
        #
        # terrain_geom.solref = contact_solref
        #
        # for geom in spec.geoms:
        #     if geom.name in foot_names:
        #         geom.solref = contact_solref
        spec = mujoco.MjSpec.from_file(filename=xml_path)
        self.step_height_curriculum_level = np.linspace(0.0, 0.15, 8)

        tile_size = 5.0
        curriculum_spacing = tile_size * 3.0 # 15 m
        self.tile_poses = np.zeros((8,2))
        for level in range(8):
            center_x = level * curriculum_spacing
            center_y = 0.0
            tile_center_pos = [center_x, center_y]
            self.tile_poses[level] = tile_center_pos
            # Center tile: flat at level zero, stairs at higher levels.
            step_height = self.step_height_curriculum_level[level]
            if level > 0 :
                spec = stairs(
                    spec=spec,
                    grid_loc=tile_center_pos,
                    name=f"stairs_center_{level}",
                    tile_size=tile_size,
                    num_stairs=6,
                    direction=1,
                    step_height=step_height,
                    step_depth=0.24,
                    terrain_group=5,
                )
            # Eight neighboring flat tiles.
            if level < 7:
                step_height_sides = self.step_height_curriculum_level[level+1]

            else:
                step_height_sides = self.step_height_curriculum_level[level]

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    spec = stairs(
                        spec=spec,
                            grid_loc=(
                                center_x + dx * tile_size,
                                center_y + dy * tile_size,
                            ),
                        name=f"stairs_center_{level}_{dx + 1}_{dy + 1}",
                        tile_size=tile_size,
                        num_stairs=6,
                        direction=1,
                        step_height=step_height_sides,
                        step_depth=0.24,
                        terrain_group=5,
                    )

                    # spec = flat_floor(
                    #     spec=spec,
                    #     grid_loc=(
                    #         center_x + dx * tile_size,
                    #         center_y + dy * tile_size,
                    #     ),
                    #     name=f"flat_{level}_{dx + 1}_{dy + 1}",
                    #     tile_size=(tile_size, tile_size),
                    #     terrain_group=5,
                    # )

            # if step_height == 0.0:
            #     spec = flat_floor(
            #         spec=spec,
            #         grid_loc=(center_x, center_y),
            #         name=f"center_flat_{level}",
            #         tile_size=(tile_size, tile_size),
            #         terrain_group=5,
            #     )
            # else:
            #     spec = stairs(
            #         spec=spec,
            #         grid_loc=tile_center_pos,
            #         name=f"stairs_{level}",
            #         tile_size=tile_size,
            #         num_stairs=4,
            #         direction=1,
            #         step_height=step_height,
            #         step_depth=0.24,
            #         terrain_group=5,
            #     )
            #
            # # Eight neighboring flat tiles.
            # for dx in (-1, 0, 1):
            #     for dy in (-1, 0, 1):
            #         if dx == 0 and dy == 0:
            #             continue

                    # spec = flat_floor(
                    #     spec=spec,
                    #     grid_loc=(
                    #         center_x + dx * tile_size,
                    #         center_y + dy * tile_size,
                    #     ),
                    #     name=f"flat_{level}_{dx + 1}_{dy + 1}",
                    #     tile_size=(tile_size, tile_size),
                    #     terrain_group=5,
                    # )

        # step_height_curriculum_level = np.linspace(0.0, 0.15, 8)
        # for level in range(2):
        #     tile_size = 5.0
        #     tile_center_pos = [level * 7, 0.0]

            # spec = flat_floor(spec=spec,
            #                   grid_loc=(3+level * 7, 0),
            #                   name=f"flat_f{level}", tile_size=(2, 7),
            #                   terrain_group=5)
            # spec = flat_floor(spec=spec,
            #                   grid_loc=(-5+level * 10, 0),
            #                   name=f"flat_b{level}", tile_size=(5, 5),
            #                   terrain_group=5)
            # spec = flat_floor(spec=spec,
            #                   grid_loc=(level * 6, -3.5),
            #                   name=f"flat_l{level}", tile_size=(8, 2),
            #                   terrain_group=5)
            # spec = flat_floor(spec=spec,
            #                   grid_loc=(level * 6, 3.5),
            #                   name=f"flat_r{level}", tile_size=(5, 5),
            #                   terrain_group=5)




            # tile_center_pos = (level * 6.0, 6.0)
            # spec = stairs(
            #     spec=spec,
            #     grid_loc=tile_center_pos,
            #     num_stairs=4,
            #     direction=-1,
            #     name=f"stairs_down_level_{level}",
            #     tile_size=5.0,
            #     step_height=0.10,#0.1
            #     step_depth=0.24,
            #     terrain_group=5,
            # )

        self.height_site_names = add_height_sampling_sites(
            spec,
            body_name="trunk",
            foot_plane_z=-0.28,
        )
        model = spec.compile()
        self.model = model
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        # Store IDs in the same matrix arrangement.
        self.height_site_ids = np.array([
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
            for name in self.height_site_names.ravel()
        ]).reshape(self.height_site_names.shape)

        self.height_sampler = TerrainHeightSampler(
            model=self.model,
            site_ids=self.height_site_ids,
            nominal_height=0.30,
            terrain_group=5,
            ray_top_z=3.0,
        )



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


        self.height_map_idx = [self.observation_name_to_id["height_map_{}".format(i)] for i in range(187)]
        # self.gravity_idx = [self.observation_name_to_id["gravity"]]
        # self.tangential_friction_idx = [self.observation_name_to_id["tangential_friction"]]
        # self.torsional_friction_ground_idx = [self.observation_name_to_id["torsional_friction_ground"]]
        # self.rolling_friction_ground_idx = [self.observation_name_to_id["rolling_friction_ground"]]
        # self.rolling_friction_feet_idx = [self.observation_name_to_id["rolling_friction_feet"]]
        # self.torsional_friction_feet_idx = [self.observation_name_to_id["torsional_friction_feet"]]

        self.reward_function.init()




    def initial_state(self):
        if self.final_level or self.current_level==8:
            self.current_level = np.random.choice(8)
        nominal_joint_pos = self.nominal_joint_positions.copy()
        initial_joint_pos = nominal_joint_pos.copy()
        self.initial_drop_height =  0.30 + self.current_level * self.step_height_curriculum_level[self.current_level] + 0.15
        init_xpos,init_ypos = self.tile_poses[self.current_level,:]
        qpos = np.array([
            init_xpos,init_ypos, self.initial_drop_height,
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
        height_matrix, valid_matrix = self.height_sampler.update(self.data)

        self.heights_obs = height_matrix.flatten().copy()




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


        torque_limits = self.model.actuator_ctrlrange[:, 1].copy()
        torques = np.clip(self.controller.action2torques(action), -torque_limits, torque_limits)
        self.data.ctrl = torques.copy()
        for _ in range(self.nr_intermediate_steps):

            mujoco.mj_step(self.model, self.data, self.nr_substeps)
            self.data.qvel[6:] = np.clip(self.data.qvel[6:], -self.max_joint_velocities, self.max_joint_velocities)

        height_matrix, valid_matrix = self.height_sampler.update(self.data)
        self.heights_obs = height_matrix.flatten().copy()


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
            # evaluate curriculum

            self.final_level = self.current_level >= 8


            robot_xy = self.data.qpos[:2].copy()
            tile_origin_xy =self.tile_poses[self.current_level,:].copy()
            distance = np.linalg.norm(robot_xy - tile_origin_xy)
            command_xy = np.array([self.goal_x_velocity, self.goal_y_velocity])

            promote = distance > 2.5
            demote = (
                             distance < 0.5 * np.linalg.norm(command_xy) * 10.0
                     ) and not promote



            if promote:
                self.current_level += 1
                self.current_level = np.clip(self.current_level,0,9)
            if demote:
                self.current_level -= 1
                self.current_level = np.clip(self.current_level,0,9)






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

        for idx in range(187):
            observation_names.extend([f"height_map_{idx}"])
        space_low.extend([-np.inf]*187)
        space_high.extend([np.inf]*187)


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
        heights = self.heights_obs.copy()
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
            damping,
            heights

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

        observation[self.height_map_idx] = self.heights_obs.copy()

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