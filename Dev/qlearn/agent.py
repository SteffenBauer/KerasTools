#!/usr/bin/env python

import memory
import numpy as np
import os
import sys
import random

def update_progress(msg, progress):
    text = "\r{0} {1:>7.2%}".format(msg, progress)
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

        if type(epsilon) in {tuple, list}:
            delta =  ((epsilon[0] - epsilon[1]) / (epochs * epsilon_rate))
            epsilon, final_epsilon = epsilon
        else:
            delta = None
            final_epsilon = epsilon
        
        header_printed = False
        for epoch in range(initial_epoch, epochs+1):
            win_count, turn_count, losses, scores = 0, 0, [], []
            if reset_memory: self.memory.reset()
            for episode in range(episodes):
                game.reset()
                F = np.expand_dims(game.get_frame(), axis=0)
                S = np.repeat(F, self.num_frames, axis=0)
                current_score = 0.0
                while True:
                    action = self.act(game, S, epsilon)
                    Fn, r, game_over = game.play(action)
                    Sn = np.append(S[1:], np.expand_dims(Fn, axis=0), axis=0)
                    self.memory.remember(S, action, r, Sn, game_over)
                    S = np.copy(Sn)
                    turn_count += 1
                    current_score += r
                    if (turn_count >= train_interval) or (episode == episodes-1 and game_over):
                        result = self.replay(gamma, batch_size, game.nb_actions)
                        if result: losses.append(result)
                        turn_count = 0
                        if not header_printed:
                            print("{:^10s}|{:^9s}|{:^14s}|{:^9s}|{:^9s}|{:^15s}|{:^8s}".format("Epoch","Epsilon","Episode","Loss", "Win", "Avg/Max Score", "Memory"))
                            header_printed = True
                        update_progress("{0: 4d}/{1: 4d} |   {2:.2f}  | {3: 4d}".format(epoch, epochs, epsilon, episode), float(episode+1)/episodes)
                    if game_over:
                        scores.append(current_score)
                        if game.is_won(): win_count += 1
                        break
            print(" | {0: 2.4f} | {1:>7.2%} | {2: 5.2f} /{3: 5.2f}  | {4: 6d}".format(
                sum(losses)/len(losses), 
                float(win_count)/float(episodes), 
                sum(scores)/float(episodes),
                max(scores),
                len(self.memory.memory)))
            if epsilon > final_epsilon and delta:
                epsilon = max(final_epsilon, epsilon - delta)

    def act(self, game, state, epsilon=0.0):
        if random.random() <= epsilon:
            return random.randrange(game.nb_actions)
        act_values = self.model.predict(np.expand_dims(np.asarray(state), axis=0))
        return int(np.argmax(act_values[0]))

    def replay(self, gamma, batch_size, nb_actions):
        batch = self.memory.get_batch(self.model, batch_size)
        if batch:
            states, actions, rewards, next_states, game_over_s = zip(*batch)
            predicted_rewards = self.model.predict(np.asarray(states))
            predicted_next_rewards = self.model.predict(np.asarray(next_states))
            rewards = list(rewards)
            targets = np.zeros((len(rewards),nb_actions))
            for i in range(len(predicted_rewards)):
                targets[i] = predicted_rewards[i]
                targets[i,actions[i]] = rewards[i]
                if not game_over_s[i]:
                    targets[i,actions[i]] += gamma*np.max(predicted_next_rewards[i])
            return self.model.train_on_batch(np.asarray(states), targets)

if __name__ == '__main__':
    pass

