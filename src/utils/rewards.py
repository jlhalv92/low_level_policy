import numpy as np

class Rewards:
    def __init__(self, target_yaw, yaw, base_lin_vel, env_class, base_ang_vel, projected_gravity, last_dof_vel, dof_vel, dt, contact_forces, penalised_contact_indices, last_actions, actions, torques, last_torques, dof_pos, default_dof_pos, hip_indices, feet_indices, rigid_body_states, terrain, cfg, x_edge_mask, contact_filt, terrain_levels):
        self.target_yaw = target_yaw
        self.yaw = yaw
        self.base_lin_vel = base_lin_vel
        self.env_class = env_class
        self.base_ang_vel = base_ang_vel
        self.projected_gravity = projected_gravity
        self.last_dof_vel = last_dof_vel
        self.dof_vel = dof_vel
        self.dt = dt
        self.contact_forces = contact_forces
        self.penalised_contact_indices = penalised_contact_indices
        self.last_actions = last_actions
        self.actions = actions
        self.torques = torques
        self.last_torques = last_torques
        self.dof_pos = dof_pos
        self.default_dof_pos = default_dof_pos
        self.hip_indices = hip_indices
        self.feet_indices = feet_indices
        self.rigid_body_states = rigid_body_states
        self.terrain = terrain
        self.cfg = cfg
        self.x_edge_mask = x_edge_mask
        self.contact_filt = contact_filt
        self.terrain_levels = terrain_levels

    def _reward_tracking_yaw(self):
        rew = np.exp(-np.abs(self.target_yaw - self.yaw))
        return rew

    def _reward_lin_vel_z(self):
        rew = np.square(self.base_lin_vel[:, 2])
        rew[self.env_class != 17] *= 0.5
        return rew

    def _reward_ang_vel_xy(self):
        return np.sum(np.square(self.base_ang_vel[:, :2]), axis=1)

    def _reward_orientation(self):
        rew = np.sum(np.square(self.projected_gravity[:, :2]), axis=1)
        rew[self.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        return np.sum(np.square((self.last_dof_vel - self.dof_vel) / self.dt), axis=1)

    def _reward_collision(self):
        return np.sum(1. * (np.linalg.norm(self.contact_forces[:, self.penalised_contact_indices, :], axis=-1) > 0.1), axis=1)

    def _reward_action_rate(self):
        return np.linalg.norm(self.last_actions - self.actions, axis=1)

    def _reward_delta_torques(self):
        return np.sum(np.square(self.torques - self.last_torques), axis=1)

    def _reward_torques(self):
        return np.sum(np.square(self.torques), axis=1)

    def _reward_hip_pos(self):
        return np.sum(np.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), axis=1)

    def _reward_dof_error(self):
        dof_error = np.sum(np.square(self.dof_pos - self.default_dof_pos), axis=1)
        return dof_error

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = np.any(np.linalg.norm(self.contact_forces[:, self.feet_indices, :2], axis=2) > 4 * np.abs(self.contact_forces[:, self.feet_indices, 2]), axis=1)
        return rew.astype(float)

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().astype(int)
        feet_pos_xy[..., 0] = np.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = np.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * np.sum(self.feet_at_edge, axis=-1)
        return rew


