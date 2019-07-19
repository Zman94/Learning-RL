import gym
env = gym.make('FrozenLake8x8-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
env.close()
