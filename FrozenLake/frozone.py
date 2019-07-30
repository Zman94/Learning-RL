import gym
import random
import numpy as np

env = gym.make('FrozenLake-v0')

""" Train/Test Constants"""

MAX_EPS = 15000
TEST_EPS = 500
MAX_TURNS = 300

""" Env Constants"""

n_states = env.observation_space.n
n_actions = env.action_space.n

""" Learning Constants """
epsilon = 1
alpha = .65
gamma = .99
# Can I get a...
hot_tub = True


def ant_action(obs, e_board, l_board):
    pass
    # moves = [up(obs)]

    # if epsilon > random.random():
    # return random.randrange(4)
    # else:
    # return max(moves)


# Q = (1-alpha)*Q + alpha * (r + gamma * max(Q(s, a)))
def zac_q_update(obs, obs2, action, reward, zac_board):
    zac_board[obs, action] += alpha * \
        (reward + gamma * np.max(zac_board[obs2]) - zac_board[obs, action])


def zac_action(episode, obs, zac_board, exploit=False):
    global epsilon
    if exploit:
        return np.argmax(zac_board[obs])
    elif np.random.rand() > epsilon:
        action = np.argmax(
            (zac_board[obs] + np.random.randn(1, n_actions) / (episode/4)))
    else:
        action = env.action_space.sample()
        epsilon -= 10**-5
    return action


ant_board = [1.0] * n_states
zac_board = np.zeros((n_states, n_actions))

total_rewards = 0.0
for episode in range(1, MAX_EPS):

    obs = env.reset()
    sar_list = []
    t = 0
    done = False
    for step in range(1, MAX_TURNS):
        t += 1
        # env.render()
        # action = ant_action(obs)

        # Get Zach's action
        action = zac_action(episode, obs, zac_board)

        obs2, reward, done, info = env.step(action)

        # keep record of states and actions
        zac_q_update(obs, obs2, action, reward, zac_board)

        obs = obs2

        if done:
            total_rewards += reward
            if not episode % (MAX_EPS/20):
                print("Alpha", alpha, "Gamma", gamma, "Epsilon", epsilon)
                print("Episode:", episode,
                      "Avg. Reward:", total_rewards/episode)
            break


# print(zac_board)

tot_reward = 0.0
for episode in range(TEST_EPS):
    obs = env.reset()
    for t in range(MAX_TURNS):
        action = zac_action(episode, obs, zac_board, exploit=True)

        obs, reward, done, info = env.step(action)

        if done:
            if reward > 0:
                tot_reward += reward
            break

print("Average reward was", tot_reward/TEST_EPS)

env.close()
