#!/usr/bin/env python3

import numpy as np

class Agent(object):
    def __init__(self, model, num_frames=1):
        self.model = model
        self.num_frames = num_frames
        
    def train(self, game, episodes=500, gamma=0.99):
        time_steps = []
        for episode in range(episodes):
            game.reset()
            F = np.expand_dims(game.get_frame(), axis=0)
            S = np.repeat(F, self.num_frames, axis=0)

            done = False
            transitions = [] # list of state, action, rewards
            
            while True:
                act_prob = self.model.predict(np.expand_dims(np.asarray(S), axis=0))
                action = np.random.choice(np.array(range(game.nb_actions)), p=act_prob[0])
                Fn, r, game_over = game.play(action)
                Sn = np.append(S[1:], np.expand_dims(Fn, axis=0), axis=0)
                transitions.append((S, act_prob, action, r))
                if game_over:
                    break

            print("Episode over, final reward {}, actions {}".format(r, [t[2] for t in transitions]))
            

