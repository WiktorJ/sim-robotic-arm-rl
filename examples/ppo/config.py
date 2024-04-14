from dataclasses import dataclass


@dataclass
class Config:
    env_name: str = "InvertedPendulum-v4"
    seed: int = 1234
    asynchronous = True

    hidden_dims: tuple = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_: float = 0.95
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    dropout_rate: float = None
    state_dependent_std: bool = False
    precalc_advantages: bool = True
    use_combined_loss: bool = True

    epochs: int = 10
    n_envs: int = 1
    rollout_length: int = 128

    max_steps: int = 3_000_000
    batch_size: int = 64
    max_episode_steps: int = 128

    eval_interval: int = 500
    video_log_interval: int = 5
    num_eval_episodes: int = 1

    tqdm: bool = True
    log_interval: int = 10
    log_videos: bool = True
    logs_root: str = "/Users/wiktorjurasz/Projects/sim-robotic-arm-rl/logs"

    use_neptune: bool = True
