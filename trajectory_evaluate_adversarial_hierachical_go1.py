import numpy as np
from src.utils.state_preprocessor import StateVariableConcatenation
import matplotlib.pyplot as plt
from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.go1_env import Go1
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from src.networks.nets import ActorNetwork, CriticNetwork, PolicyNetwork, ValueNetwork
import cv2


class OccupancyGrid:
    def __init__(self, width_meters, height_meters, resolution=0.01):
        """
        Initialize the occupancy grid.
        :param width_meters: Width of the grid in meters.
        :param height_meters: Height of the grid in meters.
        :param resolution: Grid resolution in meters per cell (default is 0.01 meters).
        """
        self.resolution = resolution  # meters per grid cell
        self.width_cells = int(width_meters / resolution)  # total columns
        self.height_cells = int(height_meters / resolution)  # total rows
        self.grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)  # 0 means free, 1 means occupied

        # The center of the grid corresponds to the origin (0, 0) in world coordinates
        self.origin_x = self.width_cells // 2
        self.origin_y = self.height_cells // 2

    def reset(self):
        self.grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)  # 0 means free, 1 means occupied



    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates.
        :param x: X coordinate in meters.
        :param y: Y coordinate in meters.
        :return: (grid_x, grid_y) in grid coordinates.
        """
        origin = np.array([self.origin_x, self.origin_y])
        point = np.array([x, -y])
        point_in_grid = origin + (point / self.resolution).astype(int)
        return point_in_grid[0], point_in_grid[1]

    def add_circle(self, x, y, radius):
        """
        Add a circular obstacle to the grid.
        :param x: X coordinate of the circle center in meters.
        :param y: Y coordinate of the circle center in meters.
        :param radius: Radius of the circle in meters.
        """
        grid_center = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution)
        cv2.circle(self.grid, grid_center, grid_radius, 1, -1)  # Draw filled circle (1 indicates occupied)

    def add_rectangle(self, x, y, width, height):
        """
        Add a rectangular obstacle to the grid.
        :param x: X coordinate of the center of the rectangle in meters.
        :param y: Y coordinate of the center of the rectangle in meters.
        :param width: Width of the rectangle in meters.
        :param height: Height of the rectangle in meters.
        """
        half_width = width / 2
        half_height = height / 2
        top_left = self.world_to_grid(x - half_width, y + half_height)
        bottom_right = self.world_to_grid(x + half_width, y - half_height)
        cv2.rectangle(self.grid, top_left, bottom_right, 1, -1)  # Draw filled rectangle

    def add_square(self, x, y, side_length):
        """
        Add a square obstacle to the grid.
        :param x: X coordinate of the center of the square in meters.
        :param y: Y coordinate of the center of the square in meters.
        :param side_length: Side length of the square in meters.
        """
        self.add_rectangle(x, y, side_length, side_length)

    def add_triangle(self, x, y, size):
        """
        Add a triangular obstacle to the grid.
        :param x: X coordinate of the center of the triangle in meters.
        :param y: Y coordinate of the center of the triangle in meters.
        :param size: Side length of the triangle in meters.
        """
        half_size = size / 2
        points = np.array([
            self.world_to_grid(x, y + half_size),  # Top point
            self.world_to_grid(x - half_size, y - half_size),  # Bottom left
            self.world_to_grid(x + half_size, y - half_size)  # Bottom right
        ], np.int32)
        cv2.fillPoly(self.grid, [points], 1)  # Draw filled triangle

    def get_grid_values(self, sampling_grid):
        weights = []
        #
        for i in range(sampling_grid.shape[1]):
            x_g, y_g = self.world_to_grid(sampling_grid[0, i], sampling_grid[1, i])
            if x_g >= self.width_cells or y_g >= self.height_cells:
                weights.append(0)
            weights.append(self.grid[y_g, x_g])
        return weights

    def display_grid(self):
        """
        Display the occupancy grid for visualization.
        """
        extent = [-self.width_cells // 2 * self.resolution, self.width_cells // 2 * self.resolution,
                  -self.height_cells // 2 * self.resolution, self.height_cells // 2 * self.resolution]
        plt.imshow(self.grid,origin="upper", extent=extent)

        plt.title("Occupancy Grid Map (0,0 at the center)")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.show()

def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate
def experiment(method_name=None, task_name="obstacles_navigation"):
    np.random.seed()

    logger = Logger(results_dir=None)


    # MDP

    horizon = 1000
    gamma = 0.99
    cfg_rewards=dict()

    cfg = dict(kp=40,
               kd=1,
               antagonist=False,
               max_level=False,
               curriculum=False,
               joystick=False,
               resample=False,
               random_obstacle=False,
               static_obstacle=True,
               domain_randomization=False,
               rewards=cfg_rewards)

    mdp = Go1(horizon, gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)






    path_c ="src/nominal_models/GO_1/PPO_low_level/epoch_last"
    if method_name == "MIN_MAX":
        path ="src/nominal_models/GO1/obstacles_navigation/Adversarial_SAC/0/Protagonist/epoch_last"
        path_adversary = "src/nominal_models/GO1/obstacles_navigation/Adversarial_SAC/0/ANTAGONIST/epoch_last"

    if method_name == "QARL":
        path ="src/nominal_models/GO1/obstacles_navigation/QARL/0/Protagonist/epoch_last"
        path_adversary = "src/nominal_models/GO1/obstacles_navigation/QARL/0/ANTAGONIST/epoch_last"

    control_agent = Agent.load(path_c)
    agent = Agent.load(path)
    adversary = Agent.load(path_adversary)
    if method_name=="QARL":
        preprocessor = StateVariableConcatenation()
        obs_dim = 18 +1
        core = Core([agent, adversary], mdp,preprocessors=[preprocessor],low_level_agent=control_agent, obs_dim=obs_dim)
        core.set_variables_for_training(0.1 * np.ones(5))
    else:
        obs_dim = 18
        core = Core([agent, adversary], mdp,low_level_agent=control_agent, obs_dim=obs_dim)

    dataset = core.evaluate(n_episodes=1, render=True, id_agent=0)
    s, a, r, s1, a, info, last = parse_dataset(dataset)

    E = agent.policy.entropy(s)
    E_a = adversary.policy.entropy(s)


    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    # success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, entropy_adversary=E_a)

    occupancy_grid = OccupancyGrid(15, 15, 0.01)
    occupancy_grid.reset()
    occupancy_grid.add_rectangle(0, -3, 10,
                                      0.2)
    occupancy_grid.add_rectangle(0, 3, 10,
                                      0.2)
    occupancy_grid.add_rectangle(-5, 0, 0.2,
                                      6)
    occupancy_grid.add_rectangle(5, 0, 0.2,
                                      6)

    for i in range(1,5):
        sx, sy = mdp._model.geom("obstacle_{}".format(i)).size[:2]
        x,y,_=mdp._data.body("obstacle_{}".format(i)).xpos
        occupancy_grid.add_rectangle(x, y, 2 * sx, 2 * sy)

    x_t,y_t = mdp._model.geom("target").pos[:2]

    plt.scatter(s[:,0], s[:,1])

    # Set up direction vectors (all pointing diagonally for this example)
    U = s[:,2]
    V = s[:,3]

    # Plot arrows
    plt.quiver(s[:,0], s[:,1], U, V, color='red')
    plt.scatter(x_t, y_t, marker='x', c="orange")
    occupancy_grid.display_grid()

    #


if __name__ == '__main__':

    methods = ["MIN_MAX", "QARL"]
    experiment(method_name="MIN_MAX", task_name="maze_evaluation")
