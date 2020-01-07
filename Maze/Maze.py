import gym
import gym_maze
from gym import wrappers
from time import time
import numpy as np
import sys
import math

## program options
MIN_LEARNING_RATE = 0.1
MIN_EPSILON = 0
DISCOUNT = 0.95
EPISODES = 50000
STREAK_TO_END = 100
ENABLE_RECORDING = True

## environment rendering
# maze portal render function
def __draw_portals(self, transparency=160):

    if self.__enable_render is False:
        return

    colour_range = np.linspace(0, 255, len(self.maze.portals), dtype=int)
    colour_i = 0
    for portal in self.maze.portals:
        colour = ((100 - colour_range[colour_i])% 255, colour_range[colour_i], 0)
        colour_i += 40
        for location in portal.locations:
            ## TODO: use pygame to draw an portal
            self.__colour_cell(location, colour=colour, transparency=transparency)

gym_maze.__draw_portals = __draw_portals


## setup env for q learning
env = gym.make("maze-random-30x30-plus-v0")
env.reset()

MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
NUM_BUCKETS = MAZE_SIZE
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
SOLVED_T = np.prod(MAZE_SIZE, dtype=float)
MAX_T = np.prod(MAZE_SIZE, dtype=int) * 10

q_table = np.zeros(NUM_BUCKETS + (env.action_space.n,), dtype=float)


## q learning utils
def get_epsilon(t):
    return max(MIN_EPSILON, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

## recording
recording_folder = f"tmp/maze_{MAZE_SIZE}_{time()}"

if ENABLE_RECORDING:
    env = wrappers.Monitor(env, recording_folder, video_callable=lambda episode_id: True, force=True)

num_streaks = 0

env.render()

## perform q learning to learn to solve mazes
for episode in range(EPISODES):

    state = state_to_bucket(env.reset())
    episode_reward = 0
    learning_rate = get_learning_rate(episode)
    epsilon = get_epsilon(episode)
    done = False
    for t in range(MAX_T):

        if np.random.random() > epsilon:
            action = int(np.argmax(q_table[state]))
        else:
            action = env.action_space.sample()
        if not done:
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state_ = state_to_bucket(new_state)

            max_future_q = np.max(q_table[state_])

            current_q = q_table[state + (action, )]
            new_q = (1-learning_rate) * current_q + learning_rate * (reward + DISCOUNT * max_future_q)
            q_table[state + (action,)] = new_q

            state = state_

            env.render()

        if env.is_game_over():
            sys.exit()
        if done:
            print(f"Episode {episode} finished after {t} time steps with total reward = {episode_reward} (streak {num_streaks}).")

            if t <= SOLVED_T:
                num_streaks += 1
            else:
                num_streaks = 0
            break
        elif t >= MAX_T:
            print(f"Episode {episode} timed out at {t} with total reward = {episode_reward}.")
            break

        # early stopping: solved streaks
        if num_streaks > STREAK_TO_END:
                break
