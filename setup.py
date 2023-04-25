from setuptools import find_packages
from setuptools import setup

requirements = [
  "slim-gym @ git+https://github.com/UWRobotLearning/slim-gym.git",
  "rl_games @ git+https://github.com/UWRobotLearning/rl_games.git@slimgym",
  "omni-isaac-orbit @ git+https://github.com/UWRobotLearning/Orbit.git@feature/intuitive_packaging#subdirectory=source/extensions/omni.isaac.orbit/",
  "omni-isaac-orbit_envs @ git+https://github.com/UWRobotLearning/Orbit.git@feature/intuitive_packaging#subdirectory=source/extensions/omni.isaac.orbit_envs/",
  "omegaconf",
]

dev_requirements = [
    "pytest",
    "pytest-cov",
    "black",
    "pre-commit",
    "pre-commit-hooks",
    "flake8",
    "flake8-bugbear",
    "pep8-naming",
    "reorder-python-imports",
    "Pygments",
]

setup(
    name="jft0_franka",
    version="0.1.0",
    packages=find_packages(),
    author=["UWRobotLearningLab"],
    author_email=["rosario@cs.uw.edu"],
    url="http://github.com/UWRobotLearning/JFT0-Franka",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
)
