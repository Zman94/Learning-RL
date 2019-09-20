import gym
import tensorflow as tf
import numpy as np
import pandas as pd

env = gym.make('Pong-ram-v0')
obs = env.reset()

n_states = env.observation_space.shape
tot_states = 0
for d in n_states:
    tot_states += d
n_actions = env.action_space.n

n_hidden_1 = 200
n_input = tot_states

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_actions]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_actions]))
}

keep_prob = tf.placeholder("float")


def build_nn(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# print(tot_states, n_actions): 128 6


# for _ in range(1000):
# env.render()
# env.step(env.action_space.sample()) # take a random action
env.close()
