import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from scores.score_logger import ScoreLogger

ENV_NAME = "Pong-ram-v0"

LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000

TOTAL_EPS = 500
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

render = True


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(
            observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA *
                            np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

# n_states = env.observation_space.shape
# tot_states = 0
# for d in n_states:
# tot_states += d
# n_actions = env.action_space.n


# n_hidden_1 = 200
# n_input = tot_states
BATCH_SIZE = 500
GAMMA = .99

self.model = Sequential()
self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
self.model.add(Dense(24, activation="relu"))
self.model.add(Dense(self.action_space, activation="linear"))
self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

# weights = {
# 'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
# 'out': tf.Variable(tf.random_normal([n_hidden_1, n_actions]))
# }
# biases = {
# 'b1': tf.Variable(tf.random_normal([n_hidden_1])),
# 'out': tf.Variable(tf.random_normal([n_actions]))
# }

# keep_prob = tf.placeholder("float")


def experience_replay(self):
    if len(self.memory) < BATCH_SIZE:
        return
    batch = random.sample(self.memory, BATCH_SIZE)
    for s, a, r, s_next, f in batch:
        q_update = r
        if not f:
            q_update = (r + GAMMA *
                        np.amax(self.model.predict(s_next)[0]))
        q_values = self.model.predict(s)
        q_values[0][a] = q_update
        self.model.fit(s, q_values, verbose=0)


def build_nn(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


def pong():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    ep = 0
    while ep < TOTAL_EPS:
        obs = env.reset()
        obs = np.reshape(obs, [1, observation_space])
        while True:
            if render:
                env.render()
            action = dqn_solver.act(obs)
            obs_next, r, f, info = env.step(action)
            r = r if not f else -r  # TODO not sure
            obs_next = np.reshape(obs_next, [1, observation_space])
            dqn_solver.remember(obs, action, r, obs_next, f)
            dqn_solver.experience_replay()
            obs = obs_next
            if f:
                break
        ep += 1
    env.close()
