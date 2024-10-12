import numpy as np

def rotate_vector_with_quaternion(vector, quaternion):
    """
    Rotate a vector using a quaternion.

    :param vector: A NumPy array representing the vector to be rotated (shape: (3,)).
    :param quaternion: A NumPy array representing the quaternion (shape: (4,)).
    :return: The rotated vector as a NumPy array (shape: (3,)).
    """
    # Ensure quaternion and vector are NumPy arrays
    vector = np.asarray(vector)
    quaternion = np.asarray(quaternion)

    # Extract quaternion components
    q_w, q_x, q_y, q_z = quaternion
    q_v = np.concatenate(([0], vector))  # Create a quaternion for the vector with zero scalar part

    # Quaternion multiplication: q' = q * v * q*
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])  # Conjugate of the quaternion
    q_result = quaternion_multiply(quaternion_multiply(quaternion, q_v), q_conj)

    # Extract rotated vector (ignoring the first component which is the scalar part)
    rotated_vector = q_result[1:]

    return rotated_vector


def quaternion_multiply(q1, q2):
    """
    Perform quaternion multiplication.

    :param q1: First quaternion as a NumPy array (shape: (4,)).
    :param q2: Second quaternion as a NumPy array (shape: (4,)).
    :return: The resulting quaternion as a NumPy array (shape: (4,)).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def euler_to_quaternion(euler_angles):
    roll, pitch, yaw = euler_angles
    # Compute half-angles
    roll_half = roll / 2
    pitch_half = pitch / 2
    yaw_half = yaw / 2

    # Compute trigonometric functions
    sin_roll_half = np.sin(roll_half)
    cos_roll_half = np.cos(roll_half)
    sin_pitch_half = np.sin(pitch_half)
    cos_pitch_half = np.cos(pitch_half)
    sin_yaw_half = np.sin(yaw_half)
    cos_yaw_half = np.cos(yaw_half)

    # Compute quaternion components
    q_w = cos_roll_half * cos_pitch_half * cos_yaw_half + sin_roll_half * sin_pitch_half * sin_yaw_half
    q_x = sin_roll_half * cos_pitch_half * cos_yaw_half - cos_roll_half * sin_pitch_half * sin_yaw_half
    q_y = cos_roll_half * sin_pitch_half * cos_yaw_half + sin_roll_half * cos_pitch_half * sin_yaw_half
    q_z = cos_roll_half * cos_pitch_half * sin_yaw_half - sin_roll_half * sin_pitch_half * cos_yaw_half

    return np.array([q_w, q_x, q_y, q_z])

def quaternion_to_euler(quat, type="euler"):
    # Compute Euler angles from quaternion
    # Roll (x-axis rotation)
    q_w, q_x, q_y, q_z = quat

    sin_roll = 2 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x ** 2 + q_y ** 2)
    roll = np.arctan2(sin_roll, cos_roll)

    # Pitch (y-axis rotation)
    sin_pitch = 2 * (q_w * q_y - q_z * q_x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    # Yaw (z-axis rotation)
    sin_yaw = 2 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y ** 2 + q_z ** 2)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    if type == "euler":
        return np.array([roll, pitch, yaw])
    if type == "degree":
        roll_deg = np.rad2deg(roll)
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)
        return np.array([roll_deg, pitch_deg, yaw_deg])

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