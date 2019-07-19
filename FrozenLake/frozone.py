import gym
import random
env = gym.make('FrozenLake8x8-v0')

explore = 1
epsilon = .04
gamma = .8
hot_tub = True

ant_board = [1.0]*64
zac_board = [[0.0 for i in range(4)] for j in range(64)]

for episode in range(50):

    obs = env.reset()
    while(100):
        # env.render()
        action = ant_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs)
        if done:
            print('finished after {} steps'.format(t+1))
            print(reward)
            break
env.close()

def ant_action (obs, e_board, l_board):
    moves = [up(obs)]

    if epsilon > random.random():
        return random.randrange(4)
    else return max(moves)

