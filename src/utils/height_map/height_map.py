import matplotlib.pyplot as plt
import numpy as np


def create_evenly_distributed_grid(grid_size, area_size, center):
    cx, cy = center
    grid_height, grid_width = grid_size
    area_height, area_width = area_size

    # Calculate the spacing between points
    x_spacing = area_width / (grid_width - 1)
    y_spacing = area_height / (grid_height - 1)

    # Calculate the starting points
    x_start = cx - (area_width / 2)
    y_start = cy - (area_height / 2)

    # Generate the grid points
    x_points = np.linspace(x_start, x_start + area_width, grid_width)
    y_points = np.linspace(y_start, y_start + area_height, grid_height)

    grid_points = np.array([(x, y) for x in x_points for y in y_points])

    return grid_points



class HeightMapParkour:
    def __init__(self,resolution_cm, width, length):
        # self.resolution_cm = resolution_cm
        # self.width = int(width_m * 100 / resolution_cm)  # Convert meters to centimeters
        # self.length = int(length_m * 100 / resolution_cm)

        #
        self.resolution_cm = resolution_cm
        self.width = width
        self.length = length
        self.heights = np.zeros((self.length, self.width))  # Initialize height map with zeros
        self.center_x = self.width // 2
        self.center_y = self.length // 2
        self.object_pos = []

    def reset_map(self):
        self.heights = np.zeros((self.length, self.width))

    def generate_ramp(self, ramp_type, ramp_ph, slope, offset, height):
        ramp = ramp_ph
        l,h = ramp.shape
        if ramp_type == 0:

            for i in range(0, l):
                for j in range(0, h):
                    ramp[i, j] = slope * i*(1/self.resolution_cm)

        if ramp_type == 1:

            for i in range(0, l):
                for j in range(0, h):
                    ramp[i, j] = height - slope * i*(1/self.resolution_cm)

        return ramp
    def add_ramp(self, x, y, l, w, h, ramp_type=0, offset=0):
        x_cm = int(x * self.resolution_cm)
        y_cm = int(y * self.resolution_cm)
        l_cm = int(l * self.resolution_cm)
        w_cm = int(w* self.resolution_cm)
        # Calculate the adjusted position relative to the center
        x_adj = self.center_x + x_cm - l_cm // 2
        y_adj = self.center_y + y_cm - w_cm // 2
        ramp = self.create_object(l,w)
        # ramp_ph = np.zeros(self.heights[y_start:y_end, x_start:x_end].shape)
        slope = h/w
        offset = offset

        ramp = self.generate_ramp(ramp_type,ramp, slope, offset, h)
        # Extract and return the window
        self.heights[y_adj:y_adj + w_cm, x_adj:x_adj + l_cm] = offset + ramp


    def create_object(self, l_m, w_m):

        l_cm = int(l_m * self.resolution_cm)
        w_cm = int(w_m * self.resolution_cm)
        return np.zeros((w_cm, l_cm))


    def spawn_object(self, x_m, y_m, l_m, w_m, h_m):
        # Convert object positions and dimensions from meters to centimeters
        x_cm = int(x_m * self.resolution_cm)
        y_cm = int(y_m *self.resolution_cm)
        l_cm = int(l_m * self.resolution_cm)
        w_cm = int(w_m * self.resolution_cm)
        h_cm = h_m

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
        y_ticks = np.linspace(-self.width / 2 * self.resolution_cm , self.width / 2 * self.resolution_cm ,
                              num=5)
        x_ticks = np.linspace(-self.length / 2 * self.resolution_cm, self.length / 2 * self.resolution_cm,
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
        x_cm = int(x_m * 100 / self.resolution_cm) //2
        y_cm = int(y_m * 100 / self.resolution_cm) //2
        x_adj = self.center_x + x_cm
        y_adj = self.center_y + y_cm

        # Draw X mark at the specified location
        plt.scatter(y_adj, x_adj, marker='x', color='red')

    def draw_o_at_location(self, x_m, y_m):
        # Convert coordinates from meters to centimeters
        x_cm = int(x_m * 100 / self.resolution_cm) //2
        y_cm = int(y_m * 100 / self.resolution_cm) //2
        x_adj = self.center_x + x_cm
        y_adj = self.center_y + y_cm

        # Draw X mark at the specified location
        plt.scatter(y_adj, x_adj, marker='o', color='green')

    def show_plot(self):
        plt.show()

    def get_height_at_coordinate(self, x_m, y_m):
        # Convert coordinates from meters to centimeters
        x_cm = int(x_m * self.resolution_cm)
        y_cm = int(y_m * self.resolution_cm)
        x_adj = self.center_x + x_cm
        y_adj = self.center_y + y_cm

        # Check if coordinates are within the bounds of the height map
        if x_cm >= self.width or y_cm >= self.length:
            raise EnvironmentError("Coordinates are out of bounds!")

        # Retrieve the height at the specified coordinates
        return self.heights[
            y_adj, x_adj]   # Convert back to meters in the plot method, rotate the axis so x is the vertical axis and y the horizontal




class HeightMap:
    def __init__(self, width_m, length_m, resolution_cm):
        self.resolution_cm = resolution_cm
        self.width = int(width_m * 100 / resolution_cm)  # Convert meters to centimeters
        self.length = int(length_m * 100 / resolution_cm)
        self.heights = np.zeros((self.length, self.width))  # Initialize height map with zeros
        self.center_x = self.width // 2
        self.center_y = self.length // 2
        self.object_pos = []

    def reset_map(self):
        self.heights = np.zeros((self.length, self.width))

    def generate_ramp(self, ramp_type, ramp_ph, slope, offset, height):
        ramp = ramp_ph
        l,h = ramp.shape
        if ramp_type == 0:

            for i in range(0, l):
                for j in range(0, h):
                    ramp[i, j] = slope * i*(30/1000)

        if ramp_type == 1:

            for i in range(0, l):
                for j in range(0, h):
                    ramp[i, j] = height+offset-slope * i*(30/1000)

        return ramp
    def add_ramp(self, x, y, l, w, h, ramp_type=0):
        window_height = int((1000 / 30) * l)
        window_width = int((1000 / 30) * w)
        offset = 0.1/5
        scale = int(1000/30)
        mat_height, mat_width = self.heights.shape
        cx = int(scale*x//2) + 1000//2
        cy = int(scale*y//2) + 1000//2

        # Calculate the boundaries of the window
        half_height = window_height // 2
        half_width = window_width // 2


        # Calculate start and end indices for the window, ensuring they are within matrix bounds
        x_start = max(0, cx - half_height)
        x_end = min(mat_height, cx + half_height + 1)

        y_start = max(0, cy - half_width)
        y_end = min(mat_width, cy + half_width + 1)
        slope = h/w
        ramp_ph = np.zeros(self.heights[y_start:y_end, x_start:x_end].shape)

        ramp = self.generate_ramp(ramp_type,ramp_ph, slope, offset, h)
        # Extract and return the window
        self.heights[y_start:y_end, x_start:x_end] = offset + ramp





    def spawn_object(self, x_m, y_m, l_m, w_m, h_m):
        # Convert object positions and dimensions from meters to centimeters
        x_cm = int(x_m * 100 / self.resolution_cm)//2
        y_cm = int(y_m * 100 / self.resolution_cm)//2
        l_cm = int(l_m * 100 / self.resolution_cm)
        w_cm = int(w_m * 100 / self.resolution_cm)
        h_cm = h_m

        self.object_pos.append((x_cm, y_cm))
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
        x_cm = int(x_m * 100 / self.resolution_cm) //2
        y_cm = int(y_m * 100 / self.resolution_cm) //2
        x_adj = self.center_x + x_cm
        y_adj = self.center_y + y_cm

        # Draw X mark at the specified location
        plt.scatter(y_adj, x_adj, marker='x', color='red')

    def draw_o_at_location(self, x_m, y_m):
        # Convert coordinates from meters to centimeters
        x_cm = int(x_m * 100 / self.resolution_cm) //2
        y_cm = int(y_m * 100 / self.resolution_cm) //2
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

