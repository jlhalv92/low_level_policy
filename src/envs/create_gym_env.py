import gymnasium as gym
from src.envs.gymnasium_go1_env import UnitreeGo1

def create_env(config):
    def make_env(seed):
        def thunk():
            env = UnitreeGo1(
                seed=seed,
                render=config.environment.render,
                mode=config.environment.mode,
                control_type=config.environment.control_type,
                command_type=config.environment.command_type,
                command_sampling_type=config.environment.command_sampling_type,
                initial_state_type=config.environment.initial_state_type,
                reward_type=config.environment.reward_type,
                termination_type=config.environment.termination_type,
                domain_randomization_sampling_type=config.environment.domain_randomization.sampling_type,
                domain_randomization_action_delay_type=config.environment.domain_randomization.action_delay_type,
                domain_randomization_mujoco_model_type=config.environment.domain_randomization.mujoco_model_type,
                domain_randomization_perturbation_type=config.environment.domain_randomization.perturbation_type,
                domain_randomization_perturbation_sampling_type=config.environment.domain_randomization.perturbation_sampling_type,
                domain_randomization_unseen_robot_type=config.environment.domain_randomization.unseen_robot_type,
                domain_randomization_seen_robot_type=config.environment.domain_randomization.seen_robot_type,
                observation_dropout_type=config.environment.observation.dropout_type,
                observation_noise_type=config.environment.observation.noise_type,
                terrain_type=config.environment.terrain.type,
                missing_value=config.environment.missing_value,
                timestep=config.environment.timestep,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                total_nr_envs=config.environment.nr_envs,
            )
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_envs == 1 else gym.vector.AsyncVectorEnv
    env = vector_environment_class([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    return env
    # env = make_env(0)
    # vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_envs == 1 else gym.vector.AsyncVectorEnv
    # env = vector_environment_class([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    # env.reset(seed=config.environment.seed)

