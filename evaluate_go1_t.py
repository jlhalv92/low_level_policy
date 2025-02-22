import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from src.envs.gymnasium_env import GymnasiumGO1

from mushroom_rl.core import Logger
from src.core.core_quadruped import Core
from src.envs.go1_env_gym import Go1
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
import time

import mujoco
import mujoco.viewer


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 2*n_features)
        self._h2 = nn.Linear(2*n_features, n_features)
        self._h3 = nn.Linear(n_features, int(n_features/2))
        self._h4 = nn.Linear(int(n_features/2), n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._h4.weight, 0.01)
        nn.init.constant_(self._h4.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        features3 = F.elu(self._h3(features2))
        a = self._h4(features3)

        return a

def controller(action, data):
    action *= 0.25
    kp = 20
    kd = 0.5
    current_qpos = data.qpos[7:]
    current_qvel = data.qvel[6:]
    nominal_q = np.array([
            -0.2, -0.7, 1.25,
            0.2, 0.7, -1.25,
            0.2, -1, 1.35,
            -0.2, 1, -1.35,
        ])
    q_target = action + nominal_q

    torque = kp*(q_target - current_qpos) - kd*(current_qvel)

    return torque


class PDController:
    def __init__(self, data, nominal_joint_positions, control_frequency_hz=50, p_gain=20, d_gain=0.5, scaling_factor=0.25):
        self.data = data
        self.nominal_joint_positions = nominal_joint_positions
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

        current_qpos = self.data.qpos[7:].copy()
        current_qvel = self.data.qvel[6:].copy()

        scaled_action = action * self.action_scale
        if self.apply_scale_shoulder:
            scaled_action[[0, 3, 6, 9]] *= 0.5
        target_joint_positions = self.nominal_joint_positions.copy() + scaled_action

        torques = self.kp * self.kp_noise_factor * (target_joint_positions - current_qpos + self.joint_position_offset)- self.kd * self.kd_noise_factor * current_qvel
        return torques * self.motor_strength_factor

def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def update_orientation_attributes(data):
    orientation_quat = R.from_quat([data.qpos[4],
                                    data.qpos[5],
                                    data.qpos[6],
                                    data.qpos[3]])
    orientation_euler = orientation_quat.as_euler("xyz")
    orientation_quat_inv = orientation_quat.inv()
    return orientation_quat, orientation_euler, orientation_quat_inv

def get_robot_observation(data, nominal_joint_positions, current_action, orientation_quat_inv, commands):

    noise_joint_angles = np.random.uniform(low=-0.03, high=0.03, size=12)
    noise_joint_vel =  np.random.uniform(low=-1.5, high=1.5, size=12)
    vel_noise =  np.random.uniform(low=-0.1, high=0.1, size=3)
    ang_vel_noise =  np.random.uniform(low=-0.2, high=0.2, size=3)
    gravity_noise =  np.random.uniform(low=-0.05, high=0.05, size=3)

    qpos = (data.qpos[7:].copy() + noise_joint_angles) - nominal_joint_positions #(12)

    qvel = data.qvel[6:] + noise_joint_vel  #(12)
    previous_action = current_action   #(12)
    robot_linear_vel = orientation_quat_inv.apply(data.qvel[:3]).copy()  + vel_noise#(3)
    robot_angular_vel = data.qvel[3:6].copy() + ang_vel_noise #(3)
    commands = np.array(commands) #(3)
    projected_gravity = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0])).copy() + gravity_noise #(3)


    # Normalize and clip
    qpos /= 4.6
    qvel /= 35.0
    previous_action /= 10.0

    robot_linear_vel = np.clip(robot_linear_vel / 10.0, -1.0, 1.0)
    robot_angular_vel= np.clip(robot_angular_vel / 50.0, -1.0, 1.0)

    observation = np.concatenate([qpos,
                            qvel,
                            projected_gravity,
                            previous_action,
                            robot_linear_vel,
                            robot_angular_vel,
                            commands])

    return observation

def experiment():
    np.random.seed()

    logger = Logger(results_dir=None)


    # MDP

    horizon = 10000
    gamma = 0.99
    cfg_rewards=dict(curriculum_steps=1)






    path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_student/epoch_50"


    agent = Agent.load(path)
    # xml_path = "src/envs/unitree_go1/walk_go1.xml"
    control_frequency_hz = 50
    nr_substeps = int(round(1 /control_frequency_hz / 0.005))
    nr_intermediate_steps = 1
    timestep = 0.005
    dt = timestep * nr_substeps * nr_intermediate_steps
    # model = mujoco.MjModel.from_xml_path(xml_path)
    # model.opt.timestep = timestep
    # data = mujoco.MjData(model)
    env_args = dict(vx_limit=3, wz_limit=5)

    mdp_eval = GymnasiumGO1(gamma=gamma,
                            n_envs=1,
                            joystick=False,
                            domain_randomization=False,
                            horizon=1000, **env_args)
    data = mdp_eval.env.data
    model = mdp_eval.env.model
    nominal_q = np.array([
            -0.2, -0.7, 1.25,
            0.2, 0.7, -1.25,
            0.2, -1, 1.35,
            -0.2, 1, -1.35,
        ])

    obs,_ = mdp_eval.env.reset()

        # Close the viewer automatically after 30 wall-seconds.
    step = 0

    while step <horizon:

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mdp_eval.env.goal_x_velocity = 0
        mdp_eval.env.goal_y_velocity = 0
        mdp_eval.env.goal_yaw_velocity = 0
        observation = mdp_eval.env.get_robot_observation()
        action = agent.draw_action(observation)
        obs, *_ = mdp_eval.env.step(action)
        mdp_eval.render(record=False)

        # torques = controller.action2torques(action)
        # for _ in range(nr_intermediate_steps):
        #     data.ctrl = torques
        #     mujoco.mj_step(model, data, nr_substeps)
        #     data.qvel[6:] = np.clip(data.qvel[6:], -max_joint_velocities, max_joint_velocities)

        # orientation_quat, orientation_euler, orientation_quat_inv = update_orientation_attributes(data)
        # old_action = action.copy()


        step +=1
        time.sleep(dt)








if __name__ == '__main__':

    for _ in range(1):
        experiment()
