from .memory import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import sys

def update_progress(msg, progress):
    barLength = 20
    status = ""
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2:.2%} {3}".format(msg, "="*(block-1) + ">" + "-"*(barLength-block), progress, status)
    sys.stdout.write(text)
    sys.stdout.flush()

class Agent:

    def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
        assert len(model.get_output_shape_at(0)) == 2, "Model's output shape should be (nb_samples, nb_actions)."
        if memory:
            self.memory = memory
        else:
            self.memory = ExperienceReplay(memory_size)
        if not nb_frames and not model.get_input_shape_at(0)[1]:
            raise Exception("Missing argument : nb_frames not provided")
        elif not nb_frames:
            nb_frames = model.get_input_shape_at(0)[1]
        elif model.get_input_shape_at(0)[1] and nb_frames and model.get_input_shape_at(0)[1] != nb_frames:
            raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
        self.model = model
        self.nb_frames = nb_frames
        self.frames = None

    @property
    def memory_size(self):
        return self.memory.memory_size

    @memory_size.setter
    def memory_size(self, value):
        self.memory.memory_size = value

    def reset_memory(self):
        self.memory.reset_memory()

    def check_game_compatibility(self, game):
        if len(self.model.inputs) != 1:
            raise Exception('Multi node input is not supported.')
        game_output_shape = (1, None) + game.get_frame().shape
        if len(game_output_shape) != len(self.model.get_input_shape_at(0)):
            raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        else:
            for i in range(len(self.model.get_input_shape_at(0))):
                if self.model.get_input_shape_at(0)[i] and game_output_shape[i] and self.model.get_input_shape_at(0)[i] != game_output_shape[i]:
                    raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        if len(self.model.get_output_shape_at(0)) != 2 or self.model.get_output_shape_at(0)[1] != game.nb_actions:
            raise Exception('Output shape of model should be (nb_samples, nb_actions).')

    def get_game_data(self, game):
        frame = game.get_frame()
        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)
        return np.expand_dims(self.frames, 0)

    def clear_frames(self):
        self.frames = None

    def train(self, game, epochs=1, initial_epoch=1, episodes=256,
              batch_size=32, train_interval=32, gamma=0.9, epsilon=[1., .1],
              epsilon_rate=0.5, reset_memory=False, observe=0, callbacks=[]):
        self.check_game_compatibility(game)
        if type(epsilon)  in {tuple, list}:
            delta =  ((epsilon[0] - epsilon[1]) / (epochs * epsilon_rate))
            epsilon, final_epsilon = epsilon
        else:
            final_epsilon = epsilon
        nb_actions = self.model.get_output_shape_at(0)[-1]

        for epoch in range(initial_epoch, epochs+1):
            win_count, loss, score, max_score, loss_count, play_count = 0, 0.0, 0, -1000, 0, 0
            if reset_memory: self.reset_memory()
            for episode in range(episodes):
                game.reset()
                self.clear_frames()
                S = self.get_game_data(game)
                game_over, game_score = False, 0
                for c in callbacks: c.game_start(game.get_frame())
                while not game_over:
                    if np.random.random() < epsilon or epoch < observe:
                        a = int(np.random.randint(game.nb_actions))
                    else:
                        q = self.model.predict(S)
                        a = int(np.argmax(q[0]))
                    game.play(a)
                    r = game.get_score()
                    S_prime = self.get_game_data(game)
                    for c in callbacks: c.game_frame(game.get_frame())
                    game_over = game.is_over()
                    self.memory.remember(S, a, r, S_prime, game_over)
                    S = S_prime
                    play_count += 1
                    game_score += r
                    if epoch >= observe and (play_count >= train_interval or game_over):
                        play_count = 0
                        batch = self.memory.get_batch(model=self.model, batch_size=batch_size, gamma=gamma)
                        if batch:
                            inputs, targets = batch
                            loss += float(self.model.train_on_batch(inputs, targets))
                            loss_count += 1
                        state = "Epoch {:>4d}/{:>4d} | Epsilon {:.2f} | Episode {:>3d}/{:>3d} ".format(epoch, epochs, epsilon, episode+1, episodes)
                        update_progress(state, float(episode+1)/float(episodes))

                if game.is_won():
                    win_count += 1
                score += game_score
                if game_score > max_score: max_score = game_score
                for c in callbacks: c.game_over()
            if loss_count > 0: loss /= float(loss_count)
            win_ratio = float(win_count)/float(episodes)
            avg_score = float(score)/float(episodes)
            print(" Loss {: .4f} | Win {: 5.2%} | Avg/Max Score {: 5.2f}/{: 5.2f} | Store {:>5d}".format(loss, win_ratio, avg_score, float(max_score), len(self.memory.memory)))
            for c in callbacks: c.epoch_end(self.model, game.name, epoch, epsilon, loss, win_ratio, avg_score, max_score, len(self.memory.memory))
            if epsilon > final_epsilon and epoch >= observe:
                epsilon = max(final_epsilon, epsilon - delta)

    def play(self, game, nb_epoch=10, epsilon=0., verbose=1, visualize=False):
        self.check_game_compatibility(game)
        win_count = 0
        frames = []
        for epoch in range(nb_epoch):
            game.reset()
            self.clear_frames()
            S = self.get_game_data(game)
            if visualize:
                frames.append(game.draw())
            game_over = False
            while not game_over:
                if np.random.rand() < epsilon:
                    action = int(np.random.randint(0, game.nb_actions))
                else:
                    q = self.model.predict(S)[0]
                    possible_actions = game.get_possible_actions()
                    q = [q[i] for i in possible_actions]
                    action = possible_actions[np.argmax(q)]
                game.play(action)
                S = self.get_game_data(game)
                if visualize:
                    frames.append(game.draw())
                game_over = game.is_over()
            if game.is_won():
                win_count += 1
        if verbose > 0:
            print("Accuracy {} %".format(100. * win_count / nb_epoch))
        if visualize:
            if 'images' not in os.listdir('.'):
                os.mkdir('images')
            for i in range(len(frames)):
                plt.imshow(frames[i], interpolation='none')
                plt.savefig("images/{0}{1:05d}.png".format(game.name, i))

    def play_turn(self, game, epsilon=0):
        self.check_game_compatibility(game)
        S = self.get_game_data(game)
        if game.is_over():
            return None
        if np.random.rand() < epsilon:
            action = int(np.random.randint(0, game.nb_actions))
        else:
            q = self.model.predict(S)[0]
            possible_actions = game.get_possible_actions()
            q = [q[i] for i in possible_actions]
            action = possible_actions[np.argmax(q)]
        game.play(action)
        return action

    def get_action(self, game, epsilon=0):
        self.check_game_compatibility(game)
        S = self.get_game_data(game)
        if game.is_over():
            return None
        if np.random.rand() < epsilon:
            action = int(np.random.randint(0, game.nb_actions))
        else:
            q = self.model.predict(S)[0]
            possible_actions = game.get_possible_actions()
            q = [q[i] for i in possible_actions]
            action = possible_actions[np.argmax(q)]
        return action


