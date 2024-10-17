import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_rotated_ring_grid(robot_pose,theta, inner_radius=0.4, outer_radius=2., num_points_radius=15, num_points_angle=20):
    # Create the radius values between inner and outer radii
    r_values = np.linspace(inner_radius, outer_radius, num_points_radius)

    # Create the angle values from 0 to 2π
    theta_values = np.linspace(0, 2 * np.pi, num_points_angle, endpoint=False)

    # Create a grid of polar coordinates (r, θ)
    r_grid, theta_grid = np.meshgrid(r_values, theta_values)

    # Convert polar coordinates to Cartesian coordinates (x, y)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    # Flatten the grid points into a list of (x, y) tuples
    points = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Rotation matrix for angle theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Rotate each point
    rotated_points = np.dot(rotation_matrix, points)

    # Translate back to the center (x_c, y_c)
    rotated_points[0, :] += robot_pose[0]
    rotated_points[1, :] += robot_pose[1]

    return rotated_points


def create_rotated_grid(robot_pose ,theta):

    x = np.arange(-1.2,1.2,0.1)
    y = np.arange(-1.,1,0.1)

    xv, yv = np.meshgrid(x, y)

    # Flatten the grid points into a list of (x, y) tuples
    points = np.vstack([xv.ravel(), yv.ravel()])

    # Rotation matrix for angle theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Rotate each point around the center (x_c, y_c)
    rotated_points = np.dot(rotation_matrix, points)

    # Translate back to the center (x_c, y_c)
    rotated_points[0, :] += robot_pose[0]
    rotated_points[1, :] += robot_pose[1]

    return rotated_points

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
