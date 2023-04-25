# Flat is better than nested. - https://peps.python.org/pep-0020/
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class SimConfig:
    device: str = "${sim_device}"
    headless: bool = "${headless}"
    dt: float = 0.005
    substeps: int = 1
    gravity: Tuple[float, float, float] = (0., 0., -9.81) # [m/s^2]
    up_axis: int = 1 # 0 is y, 1 is z
    use_gpu_pipeline: bool = "${evaluate_use_gpu: ${task.sim.device}}"

# === Configs defined in RLGames ===
# for readability, they have been copied here and are imported from here
# these could alternatively be placed in their own file: configs/rl_games.py
@dataclass(frozen=True)
class RunnerConfig():
    seed: int = 42
    device: str = "cuda:0"
    multi_gpu: bool = False
    load_checkpoint: bool = False
    load_path: str = "" 
    max_epochs: int = 500
    minibatch_size: int = 4096
    mini_epochs: int = 8
    score_to_win: float = 20000
    save_best_after: int = 200
    save_frequency: int = 50


@dataclass(frozen=True)
class AlgorithmConfig():
    name: str = "a2c_continuous"
    ppo: str = True
    normalize_input: bool = True
    normalize_value: bool = True
    value_bootstrap: bool = False
    reward_shaper_scale_val: float = 0.01
    normalize_advantage: bool = True
    gamma: float = 0.99 
    tau: float = 0.95

    grad_norm: float = 1.0
    entropy_coef: float = 0.0
    truncate_grads: bool =True
    e_clip: float = 0.2
    horizon_length: int = 16
    critic_coef: int = 2
    clip_value: bool =True
    seq_len: int = 4
    bounds_loss_coef: float = 0.0001

    # learning rate
    schedule_type: str = "legacy"
    learning_rate: float = 3e-4
    lr_schedule: str = "adaptive"
    kl_threshold: float = 0.008

@dataclass(frozen=True)
class SpaceConfig():
    mu_activation: str = "None"
    sigma_activation: str = "None"
    mu_init: str = "default"
    sigma_init: str = "const_initializer"
    sigma_val: float = 0.0
    fixed_sigma: bool = True

@dataclass(frozen=True)
class MLPConfig():
    units: tuple = (256, 128, 64)
    activation: str = "elu"
    d2rl: bool = False
    initializer: str = "default"
    regularizer: str = "None"

@dataclass(frozen=True)
class NetworkConfig():
    name: str = "actor_critic"
    separate: bool = False 
    space: SpaceConfig = SpaceConfig()
    mlp: MLPConfig = MLPConfig()

@dataclass(frozen=True)
class PolicyConfig():
    name: str = "continuous_a2c_logstd"
    network: NetworkConfig = NetworkConfig()

