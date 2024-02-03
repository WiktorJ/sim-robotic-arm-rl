from dataclasses import dataclass


@dataclass
class Config:
    env_name = "FetchReachDense-v2"

    actor_lr = 3e-4
    critic_lr = 3e-4
    temperature_lr = 3e-4
    hidden_dims = (128, 128)
    gamma = 0.99
    tau = 0.005
    target_update_period = 1
    state_dependent_std = False
    target_entropy_multiplier = 1
    fixed_temperature = True
    temperature = 1

    seed = 123

    max_episode_steps = 100
    replay_buffer_capacity = 100000
    max_steps = 10000
    start_training_step = 2000
    update_steps = 1
    batch_size = 2048

    eval_interval = 250
    video_log_interval = 4
    num_eval_episodes = 1

    tqdm = True
    log_interval = 10
    log_videos = True
    logs_root = '/Users/wiktorjurasz/Projects/sim-robotic-arm-rl/logs'

    use_neptune=False
