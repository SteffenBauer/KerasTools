#!/usr/bin/env python

import memory
import numpy as np
import os
import sys
import random

def update_progress(msg, progress):
    text = "\r{0}: {1: .2%}".format(msg, progress)
    sys.stdout.write(text)
    sys.stdout.flush()

class Agent(object):
    def __init__(self, model, mem=None, memory_size=1000, num_frames=None):
        self.model = model
        if mem == None:
            self.memory = memory.BasicMemory(memory_size = memory_size)
        else:
            self.memory = mem
            
        self.num_frames = num_frames
        
        
    def train(self, game, epochs=1, initial_epoch=1, episodes=256,
              batch_size=32, train_interval=32, gamma=0.9, epsilon=[1., .1],
              epsilon_rate=0.5, reset_memory=False, observe=0, callbacks=[]):

        for epoch in range(initial_epoch, epochs+1):
            win_count, turn_count, losses = 0, 0, []
            if reset_memory: self.memory.reset()
            for episode in range(episodes):
                game.reset()
                F = game.get_frame()
                S = [F]*self.num_frames
                while True:
                    action = self.act(game, S, epsilon)
                    Fn, r, game_over = game.play(action)
                    Sn = S[1:] + [Fn]
                    self.memory.remember(S, action, r, Sn, game_over)
                    S = Sn
                    turn_count += 1
                    if (turn_count > train_interval) or (episode == episodes-1 and game_over):
                        self.replay(losses, gamma, batch_size, game.nb_actions)
                        turn_count = 0
                        update_progress("{} {}".format(epoch, episode), float(episode+1)/episodes)
                    if game_over:
                        if game.is_won(): win_count += 1
                        break
            print " Loss {0: 2.4f} Win% {1: .2%} Mem {2: 5d}".format(sum(losses)/len(losses), float(win_count)/episodes, len(self.memory.memory))

    def act(self, game, state, epsilon=0.0):
        if random.random() <= epsilon:
            return random.randrange(game.nb_actions)
        act_values = self.model.predict(np.expand_dims(np.asarray(state), axis=0))
        return np.argmax(act_values[0])  # returns action

    def replay(self, losses, gamma, batch_size, actions):
        states, targets = self.create_training_set(batch_size, gamma)
        losses.append(self.model.train_on_batch(np.asarray(states), np.asarray(targets)))
    
    def create_training_set(self, batch_size, gamma):
        batch = self.memory.get_batch(self.model, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, game_over in batch:
            if not game_over:
                reward += gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            target[action] = reward 
            states.append(state)
            targets.append(target)
        return states, targets
    
if __name__ == '__main__':
    pass

