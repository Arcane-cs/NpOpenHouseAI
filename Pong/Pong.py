import gym
from dddql_torch import DeepQNetwork, Agent
import numpy as np
from gym import wrappers

"""
TO DO:
- implement a better epsilon decay

gamma = discount
alpha = learning rate
"""

if __name__ == '__main__':
    env = gym.make('Pong-v0')
    brain = Agent(gamma=0.90, epsilon=1.0, 
                  alpha=0.004, maxMemorySize=5000)
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
                                np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_
    print('done initializing memory')

    scores = []
    epsHistory = []
    numGames = 50000
    batch_size=32
    num_streak = 0
    env = wrappers.Monitor(env, "tmp/pong-1", video_callable=lambda episode_id: True, force=True)
    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        observation = env.reset()
        brain.getEpsilon(i)
        frames = [np.sum(observation[15:200,30:125], axis=2)]
        score = 0
        lastAction = 0
        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = lastAction
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200,30:125], axis=2))
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
                                  np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_
            brain.learn(batch_size)
            lastAction = action
        scores.append(score)
        if score > 0:
            num_streak += 1
        else:
            num_streak = 0
        
        print('score:',score,' streak:',num_streak)

        if num_streak > 10:
            break
