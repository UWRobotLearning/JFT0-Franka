from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(
    {"headless": False}
) # this has to happen before other imports
import logging

# === Env Imports ===:
from envs.franka_reach_env import ReachEnv
from configs.reach_env import ReachEnvConfig

# === RL Trainer Imports ===
from rl_games.torch_runner import Runner
#from rl_games.configs.definitions import RunnerConfig, AlgorithmConfig, PolicyConfig
from configs.definitions import RunnerConfig, AlgorithmConfig, PolicyConfig

# === Config Imports ===
from  dataclasses import dataclass, asdict
from omegaconf import OmegaConf # used only for to_yaml


# === Configure Everything Here ===
@dataclass(frozen=True)
class TopLevelConfig:
    env: ReachEnvConfig = ReachEnvConfig(
        num_envs=4,
        env_spacing=2.5,
        episode_length_s=20,
    )
    runner: RunnerConfig = RunnerConfig(
        minibatch_size = 4,
        max_epochs = 50,
    )
    algorithm: AlgorithmConfig = AlgorithmConfig()
    policy: PolicyConfig = PolicyConfig()
    log_dir: str = '../../FrankaExperimentLogs'
    rl_device: str = 'cuda:0'

def main():

    # === Create Config Object ===
    # NOTE: dont configure here,
    # after entering main(),
    # configuration should be frozen.

    cfg = TopLevelConfig()
    print(OmegaConf.to_yaml(cfg))

    log.info("config loaded")

    # === Create Env Object ===

    env = ReachEnv(
        cfg=cfg.env,
        clip_obs=100.0,
        clip_actions=100.0,
        rl_device=cfg.rl_device,
    )

    log.info("sim initialized")

    # === Create Trainer Object ===

    runner = Runner(
        env=env,
        runner_cfg=cfg.runner,
        algorithm_cfg=cfg.algorithm,
        policy_cfg=cfg.policy,
    )

    log.info("trainer initialized")

    # === Main Learning Loop ===

    runner.run()

    log.info(f"model saved to: {cfg.log_dir}")

    # === Exit Gracefully === 

    env.close()
    simulation_app.close()

    log.info(f"Ciao!")

if __name__ == "__main__":
    # === Setup Logging ===
    # use this logger (from the root) in all modules
    log = logging.getLogger('') # acquire root logger 
    console = logging.StreamHandler() 
    log.addHandler(console)
    log.setLevel(logging.INFO)

    main()
