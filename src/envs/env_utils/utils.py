from src.utils.height_map import *
import json  # Function to read the JSON file
import pygame
from collections import deque
import numpy as np

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