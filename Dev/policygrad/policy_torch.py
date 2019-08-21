#!/usr/bin/env python3

# Policy algorithm by https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction
# Distributed there under the MIT license

import catch
import numpy as np
import torch

l1 = 10*10*3
l2 = 150
l3 = 3

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax(dim=0)
)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def loss_fn(preds, r): 
    # pred is output from neural network, a is action index
    # r is return (sum of rewards to end of episode), d is discount factor
    return -torch.sum(r * torch.log(preds)) # element-wise multipliy, then sum
    
MAX_DUR = 20
MAX_EPISODES = 20000
gamma_ = 0.95
time_steps = []

env = catch.Catch(grid_size=10)

win_stats = []
loss_stats = []

for episode in range(MAX_EPISODES):
    env.reset()
    curr_state = env.get_state().flatten()
    done = False
    transitions = [] # list of state, action, rewards
    
    for t in range(MAX_DUR): #while in episode
        act_prob = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0,1,2]), p=act_prob.data.numpy())
        prev_state = curr_state
        curr_state, reward, done = env.play(action)
        curr_state = curr_state.flatten()
        transitions.append((prev_state, action, reward))
        if done:
            win_stats.append(1 if reward == 1.0 else 0)
            break

    # Optimize policy network with full episode
    ep_len = len(transitions) # episode length
    time_steps.append(ep_len)
    preds = torch.zeros(ep_len)
    discounted_rewards = torch.zeros(ep_len)
    for i in range(ep_len): #for each step in episode
        discount = 1
        future_reward = 0
        # discount rewards
        for i2 in range(i, ep_len):
            future_reward += transitions[i2][2] * discount
            discount = discount * gamma_
        discounted_rewards[i] = future_reward
        state, action, _ = transitions[i]
        pred = model(torch.from_numpy(state).float())
        preds[i] = pred[action]
    loss = loss_fn(preds, discounted_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_stats.append(loss.item())
    if len(win_stats) >= 100:
        print("Episode {: 4d} Win perc {:2.4f} Loss {:2.6f}".format(episode, sum(win_stats)/100.0, sum(loss_stats)/100.0))
        win_stats = []
        loss_stats = []

