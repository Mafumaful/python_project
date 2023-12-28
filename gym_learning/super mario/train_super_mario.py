# import Frame Stacker Wrapper and Gray Scaling Wrapper
from gym.wrappers import GrayScaleObservation
# import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# import Matplotlib
from matplotlib import pyplot as plt


import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# create the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim = True)

state, info = env.reset()