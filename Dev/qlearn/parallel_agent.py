#!/usr/bin/env python3

import memory
import numpy as np
import os
import sys
import random
import copy

def update_progress(msg, progress):
    text = "\r{0} {1:>7.2%}".format(msg, progress)
    sys.stdout.write(text)
    sys.stdout.flush()

class Agent(object):
    """
    DQN agent

    # Arguments
      model: Keras network. Its shape must be compatible with the game.
        - input_shape: 5D tensor with shape (batch, nb_frames, height, width, channels)
        - output_shape: 2D tensor with shape (batch, nb_actions)
      mem: Replay memory object (instance of qlearn.memory.Memory).
        Defaults to `BasicMemory` if not specified.
      memory_size: Size of replay memory. Default to 1000.
      num_frames: Integer. Number of past game state frames to show to the network.
        Defaults to 1.
    """

    def __init__(self, model, mem=None, memory_size=1000, num_frames=1):
        self.model = model
        if mem == None:
            self.memory = memory.BasicMemory(memory_size = memory_size)
        else:
            self.memory = mem
        self.num_frames = num_frames
        self.history = {'gamma': 0, 'epsilon': [], 'memory_fill': [],
                        'win_ratio': [], 'loss': [], 'finished': [],
                        'avg_score': [], 'max_score': []}

    def train(self, game, epochs=1, initial_epoch=1, turns_per_epoch=256,
              batch_size=32, train_interval=32, gamma=0.9, epsilon=[1., .1],
              epsilon_rate=0.5, reset_memory=False, observe=0,
              verbose=1, callbacks=[]):

        """
        Train the DQN agent on a game

        # Arguments
          game: Game object (instance of a qlearn.game.Game subclass)
          epochs: Integer. Number of epochs to train the model.
            When unspecified, the network is trained over one epoch.
          initial_epoch: Integer. Value to start epoch counting.
            If unspecified, `initial_epoch` will default to 1.
            This argument is useful when continuing network training.
          episodes: Integer. Number of game episodes to play during one epoch.
          batch_size: Integer. Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
          train_interval: Integer. Do one batch of gradient update
            after that number of game turns were played.
          gamma: Float between 0.0 and < 1.0. Discount factor.
          epsilon: Exploration factor. It can be:
            - Float between 0.0 and 1.0, to set a fixed factor for all epochs.
            - List or tuple with 2 floats, to set a starting and a final value.
          epsilon_rate: Float between 0.0 and 1.0. Decay factor for epsilon.
            Epsilon will reach the final value when this percentage of epochs is reached.
          reset_memory: Boolean. Sets if the replay memory should be reset before each epoch.
            Default to `False`.
          observe: Integer. When specified, fill the replay memory with random game moves for this number of epochs.
          verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
          callbacks: List of callback objects (instance of qlearn.callbacks.Callback)

        # Returns
          A History dictionary, containing records of training parameters during successive epochs:
            - epsilon:      Epsilon value
            - loss:         Training loss
            - avg_score:    Average game score
            - max_score:    Maximum reached game score
            - win_ratio:    Percentage of won games
            - memory_fill:  Records in the replay memory

        """

        self.history['gamma'] = gamma

        if observe > 0:
            self._fill_memory(game, observe)

        if type(epsilon) in {tuple, list}:
            delta =  ((epsilon[0] - epsilon[1]) / (epochs * epsilon_rate))
            epsilon, final_epsilon = epsilon
        else:
            delta = None
            final_epsilon = epsilon

        games = [copy.copy(game) for _ in range(train_interval)]
        for g in games: g.reset()
        current_scores = [0 for _ in games]
        F = [np.expand_dims(g.get_frame(), axis=0) for g in games] # Current game frames
        S = [np.repeat(f, self.num_frames, axis=0) for f in F] # Current game states

        header_printed = False
        for epoch in range(initial_epoch, epochs+1):
            win_count, losses, scores = 0, [], []
            if reset_memory: self.memory.reset()
            current_turns = 0
            current_episodes = 0
            while current_turns < turns_per_epoch:
                for i in range(train_interval):
                    if games[i].is_over():
                        games[i].reset()
                        F[i] = np.expand_dims(games[i].get_frame(), axis=0)
                        S[i] = np.repeat(F[i], self.num_frames, axis=0)
                        current_scores[i] = 0

                actions = self.act(games, S, epsilon)
                results = [g.play(a) for g,a in zip(games, actions)] # Fn, r, game_over

                Sn = []
                for i in range(train_interval):
                    fn, r, game_over = results[i]
                    sn = np.append(S[i][1:], np.expand_dims(fn, axis=0), axis=0)
                    self.memory.remember(S[i], actions[i], r, sn, game_over)
                    Sn.append(sn)
                    current_scores[i] += r
                    current_turns += 1
                    if game_over:
                        scores.append(current_scores[i])
                        if games[i].is_won(): win_count += 1
                        current_episodes += 1
                S = Sn
                result = self._replay(gamma, batch_size, game.nb_actions)
                if result: losses.append(result)
                if not header_printed and verbose > 0:
                    print("{:^10s}|{:^9s}|{:^16s}|{:^9s}|{:^10s}|{:^9s}|{:^17s}|{:^8s}".format(
                        "Epoch","Epsilon","Turns","Loss","Finished","Win","Avg/Max Score","Memory"))
                    header_printed = True
                if verbose == 1:
                    update_progress("{0: 4d}/{1: 4d} |   {2:.2f}  | {3: 6d}".format(epoch, epochs, epsilon, current_turns), float(current_turns/turns_per_epoch))

            loss = sum(losses)/len(losses)
            win_ratio = float(win_count)/float(current_episodes) if current_episodes > 0 else 0.0
            avg_score = sum(scores)/float(current_episodes) if current_episodes > 0 else float('nan')
            max_score = max(scores) if len(scores) > 0 else float('nan')
            memory_fill = len(self.memory.memory)
            if verbose == 2:
                print("{0: 4d}/{1: 4d} |   {2:.2f}  |    {3: 4d}    ".format(epoch, epochs, epsilon, current_turns), end=' ')
            if verbose > 0:
                print(" | {0: 2.4f} | {1: 8d} | {2:>7.2%} | {3: 6.2f} /{4: 6.2f}  | {5: 6d}".format(
                    loss, current_episodes, win_ratio, avg_score, max_score, memory_fill))

            self.history['epsilon'].append(epsilon)
            self.history['loss'].append(loss)
            self.history['finished'].append(current_episodes)
            self.history['win_ratio'].append(win_ratio)
            self.history['avg_score'].append(avg_score)
            self.history['max_score'].append(max_score)
            self.history['memory_fill'].append(memory_fill)
            if epsilon > final_epsilon and delta:
                epsilon = max(final_epsilon, epsilon - delta)

        return self.history

    def act(self, games, states, epsilon=0.0):
        """
        Choose actions on given game states.

        # Arguments
          games: List of game objects (instances of a qlearn.game.Game subclass)
          states: List of game states as numpy arrays of shape (nb_frames, height, width, channels)
          epsilon: Float between 0.0 and 1.0. Epsilon factor.
                   Probability that the agent chooses a random action instead of using the DQN.

        # Returns
          List of chosen game actions. Integer between 0 and `game.nb_actions`.

        """
        nb_games = len(games)
        nb_actions = games[0].nb_actions
        actions = [random.randrange(nb_actions) for _ in range(nb_games)]
        nb_calc = len([i for i in range(nb_games) if random.random() > epsilon])
        if nb_calc > 0:
            to_calc = sorted(random.sample(range(nb_games), nb_calc))
            calc_states = [states[i] for i in to_calc]
            act_values = self.model.predict(np.asarray(calc_states))

            for i, v in zip(to_calc, act_values):
                actions[i] = np.argmax(v)

        return actions

    def _replay(self, gamma, batch_size, nb_actions):
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

    def _fill_memory(self, game, turns):
        print("Fill memory with {} random action turns".format(turns))
        game.reset()
        F = np.expand_dims(game.get_frame(), axis=0)
        S = np.repeat(F, self.num_frames, axis=0)
        turn = 0
        #for turn in range(turns):
        print("{0:10s}| {1:8s}".format("Turns", "Memory fill"))
        while len(self.memory.memory) < turns:
            turn += 1
            action = random.randrange(game.nb_actions)
            Fn, r, game_over = game.play(action)
            Sn = np.append(S[1:], np.expand_dims(Fn, axis=0), axis=0)
            self.memory.remember(S, action, r, Sn, game_over)
            if game_over:
                game.reset()
                F = np.expand_dims(game.get_frame(), axis=0)
                S = np.repeat(F, self.num_frames, axis=0)
            update_progress("{0: 4d}/{1: 4d} | {2: 6d} | ".
                format(turn, turns, len(self.memory.memory)), float(len(self.memory.memory))/turns)
        print("")

if __name__ == '__main__':
    pass

