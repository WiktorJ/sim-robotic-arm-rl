from dataclasses import dataclass


@dataclass
class Config:
    env_name: str = "Pusher-v4"

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    hidden_dims: tuple = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    target_update_period: int = 1
    state_dependent_std: bool = False
    target_entropy_multiplier: int = 1
    fixed_temperature: bool = False
    temperature: int = 1

    seed: int = 123

    max_episode_steps: int = 1000
    replay_buffer_capacity: int = 1000000
    max_steps: int = 1_000_000
    start_training_step: int = 2000
    update_steps: int = 1
    batch_size: int = 256

    eval_interval: int = 2000
    video_log_interval: int = 10
    num_eval_episodes: int = 1

    tqdm: bool = True
    log_interval: int = 10
    log_videos: bool = True
    logs_root: str = '/Users/wiktorjurasz/Projects/sim-robotic-arm-rl/logs'

    use_neptune: bool = True
