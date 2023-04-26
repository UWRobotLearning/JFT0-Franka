# JFT0-Franka
Just Fork This! (L0) for the Franka Emika Panda
This repo provides a good starting point including the following dimensions:
| **_Category_**                      | **_Example in Slot_**                                               |
|-------------------------------------|---------------------------------------------------------------------|
| **Configuration**                   | `@dataclasses`                                                      |
| **Environments**                    | [Isaac-Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html) / [Orbit](https://github.com/NVIDIA-Omniverse/Orbit)                                                                  |
| **Optimization**                    | [RL Games](https://github.com/Denys88/rl_games) / TGS                |
| **Experiment Management & Logging** | [Hydra](https://hydra.cc/docs/intro/)                                |
| **Hardware Deployment**             | Interface for ROS2 / ZMQ                                             |

# Quick Start

## 1. Installing Isaac-Sim
From [Nvidia Ominiverse Installation](https://docs.omniverse.nvidia.com/prod_install-guide/prod_install-guide/workstation.html)
[Click Here](https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage) to download Linux image.
1. Right click on AppImage, then go to properties tab, then select "allow file to execute as program".
2. Log in to Nvidia Account.
3. Follow prompts (install nucleus and cache).
4. Go to "Exhcange" tab in Ominiverse launcher and type "Isaac Sim".
5. Install Isaac Sim. Currently newest version is 2022.2.0

## 2. Creating Isaac-Sim Conda Env (rather than running with python dist from Isaac download)
```bash
cd ~/.local/share/ov/pkg/isaac_sim*
conda env create -f environment.yml
conda activate isaac-sim
source setup_conda_env.sh
```

## 3. Run Training or Play Scripts
```bash
cd ./JFT0-Franka/scripts
python scripts/train.py
```

### Guides
Coming soon:
1. A diagram explaining the relationship of Nvidia's Isaac tools (including other derived repos)
2. A guide explaining the configuration philosphy / system
3. A guide showing how to deploy a policy on the real robot with one switch
