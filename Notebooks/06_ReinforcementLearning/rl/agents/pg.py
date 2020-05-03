#!/usr/bin/env python3

# Policy algorithm by https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction
# Distributed there under the MIT license

import catch
import numpy as np
import keras

grid_size = 10
l1 = grid_size*grid_size*3
l2 = 150
l3 = 3
learning_rate = 0.001

def generate_model():
    input_state = keras.layers.Input(shape=(l1,), name="Input_State")
    x = keras.layers.Dense(l2)(input_state)
    x = keras.layers.LeakyReLU()(x)
    actions = keras.layers.Dense(l3, activation='softmax')(x)

    def loss_fn(y_true, y_pred):
        return -1.0 * keras.backend.sum(y_true * keras.backend.log(y_pred))

    model = keras.models.Model(inputs=input_state, outputs=actions)
    model.compile(loss=loss_fn, optimizer=keras.optimizers.RMSprop(learning_rate))

    return model

model = generate_model()
model.summary()

MAX_DUR = 20
MAX_EPISODES = 10000
gamma_ = 0.95
time_steps = []

env = catch.Catch(grid_size=grid_size)

win_stats = []
loss_stats = []

for episode in range(MAX_EPISODES):
    env.reset()
    curr_state = env.get_state().flatten()
    done = False
    transitions = [] # list of state, action, rewards
    
    for t in range(MAX_DUR): #while in episode
        act_prob = model.predict(np.expand_dims(np.asarray(curr_state), axis=0))
        action = np.random.choice(np.array([0,1,2]), p=act_prob[0])
        prev_state = curr_state
        curr_state, reward, done = env.play(action)
        curr_state = curr_state.flatten()
        transitions.append((prev_state, action, reward))
        if done:
            win_stats.append(1 if reward == 1.0 else 0)
            break

    # Optimize policy network with full episode
    ep_len = len(transitions) # episode length
    discounted_rewards = np.zeros((ep_len, l3))
    train_states = []
    for i in range(ep_len): #for each step in episode
        discount = 1.0
        future_reward = 0.0
        # discount rewards
        for i2 in range(i, ep_len):
            future_reward += transitions[i2][2] * discount
            discount = discount * gamma_
        discounted_rewards[i][transitions[i][1]] = future_reward
        train_states.append(transitions[i][0])
    train_states = np.asarray(train_states)
    # Backpropagate model with preds & discounted_rewards here
    loss = model.train_on_batch(train_states, discounted_rewards)
    loss_stats.append(loss)
    
    if len(win_stats) >= 100:
        print("Episode {: 4d} Win perc {:2.4f} Loss {:2.4f}".format(episode, sum(win_stats)/100.0, sum(loss_stats)/100.0))
        win_stats = []
        loss_stats = []
        
