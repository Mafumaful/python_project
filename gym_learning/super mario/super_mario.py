import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# show the number of action of the env
# print(SIMPLE_MOVEMENT[env.action_space.sample()])
# Discrete(7)

done = True
for step in range(10000):
    if done:
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # show the game on the screen
    env.render()
env.close()
