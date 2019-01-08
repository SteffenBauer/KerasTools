
import numpy as np
import os
import sys
import random
from keras import backend as K

class Game(object):
    def __init__(self): self.reset()

    @property
    def name(self): return "Game"
    @property
    def nb_actions(self): return 0

    def reset(self): pass
    def play(self, action): pass
    def get_state(self): return None
    def get_score(self): return 0
    def is_over(self): return False
    def is_won(self): return False
    def get_frame(self): return self.get_state()
    def draw(self): return self.get_state()
    def get_possible_actions(self): return range(self.nb_actions)


class Catch(Game):

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        n = random.randrange(0, self.grid_size-1)
        m = random.randrange(1, self.grid_size-2)
        self.state = [0, n, m]
        self.penalty = 0

    @property
    def name(self):
        return "Catch"

    @property
    def nb_actions(self):
        return 3

    def play(self, action):
        if self.is_over() or action not in range(self.nb_actions):
            return
        fy, fx, basket = self.state
        if action == 0:
            new_basket = max(1, basket-1)
        elif action == 2:
            new_basket = min(self.grid_size-2, basket+1)
        else:
            new_basket = basket
        self.state = [fy+1, fx, new_basket]
        self.penalty = 0.0 if action == 1 else -0.1

    def get_state(self):
        fy, fx, basket = self.state
        canvas = np.zeros((self.grid_size,self.grid_size,3))
        canvas[-1, basket-1:basket + 2, :] = (1,1,0)
        if self.is_won():
            canvas[fy, fx, :] = (0,1,0)
        elif self.is_over():
            canvas[fy, fx, :] = (1,0,0)
        else:
            canvas[fy, fx, :] = (1,1,0)

        return canvas

    def get_score(self):
        if self.is_won():
            return 1.0
        elif self.is_over():
            return -1.0
        else:
            return self.penalty

    def is_over(self):
        return self.state[0] == self.grid_size-1

    def is_won(self):
        fy, fx, basket = self.state
        return self.is_over() and abs(fx - basket) <= 1

class Snake(Game):

    def __init__(self, grid_size=10, snake_length=3, max_turn=100):
        self.grid_size = grid_size
        self.snake_length = snake_length
        self.max_turn = max_turn
        self.reset()

    @property
    def name(self):
        return "Snake"
    @property
    def nb_actions(self):
        return 3

    def play(self, action):
        if self.is_over() or action not in range(self.nb_actions):
            return
        self.scored = False
        self.turn += 1
        self.move_snake(action)
        if self.self_bite() or self.hit_border() or (self.max_turn > 0 and self.turn >= self.max_turn):
            self.snake.pop()
            self.game_over = True
        elif self.fruit != self.snake[0]:
            self.snake.pop()
        else:
            self.scored = True
            self.drop_fruit()

    def drop_fruit(self):
        if len(self.snake) >= (self.grid_size - 2) ** 2:
            self.fruit = (-1, -1)
            return
        while True:
            fruit_x = random.randrange(1, self.grid_size-1)
            fruit_y = random.randrange(1, self.grid_size-1)
            if (fruit_x, fruit_y) not in self.snake:
                self.fruit = (fruit_x, fruit_y)
                break

    def move_snake(self, action):
        h = self.snake[0]
        n = self.snake[1]
        if ((h[1] < n[1] and action == 2) or
            (h[0] < n[0] and action == 0) or
            (h[0] > n[0] and action == 1)):
            p = (h[0], h[1]-1)
        elif ((h[1] < n[1] and action == 1) or
              (h[1] > n[1] and action == 0) or
              (h[0] < n[0] and action == 2)):
            p = (h[0]-1, h[1])
        elif ((h[1] < n[1] and action == 0) or
              (h[1] > n[1] and action == 1) or
              (h[0] > n[0] and action == 2)):
            p = (h[0]+1, h[1])
        elif ((h[1] > n[1] and action == 2) or
              (h[0] < n[0] and action == 1) or
              (h[0] > n[0] and action == 0)):
            p = (h[0], h[1]+1)
        self.snake.insert(0, p)

    def get_state(self):
        canvas = np.zeros((self.grid_size,self.grid_size,3))

        # Red border
        canvas[0,:,:] = (1,0,0)
        canvas[:,0,:] = (1,0,0)
        canvas[-1,:,:] = (1,0,0)
        canvas[:,-1,:] = (1,0,0)

        # Yellow snake body
        for seg in self.snake:
            canvas[seg[0], seg[1], :] = (1,1,0)

        # Green snake head
        canvas[self.snake[0][0], self.snake[0][1], :] = (0,1,0)

        # Purple snake head if bitten
        if self.self_bite() or self.hit_border():
            canvas[self.snake[0][0], self.snake[0][1], :] = (1,0,1)

        # Blue fruit
        canvas[self.fruit[0], self.fruit[1], :] = (0,0,1)
        return canvas

    def get_score(self):
        if self.self_bite() or self.hit_border():
            score = -1.0
        elif self.scored:
            score = 1.0 # len(self.snake)
        else:
            score = 0.0
        return score

    def reset(self):
        grid_size = self.grid_size
        snake_length = self.snake_length
        head_x = (grid_size - snake_length) // 2
        self.snake = [(x, grid_size // 2) for x in range (head_x, head_x + snake_length)]
        self.game_over = False
        self.scored = False
        self.turn = 0
        self.drop_fruit()
        if random.randrange(2) == 0:
            self.snake.reverse()
        self.border = []
        for z in range(grid_size):
            self.border += [(z, 0), (z, grid_size - 1), (0, z), (grid_size - 1, z)]

    def self_bite(self):
        return len(self.snake) > len(set(self.snake))

    def hit_border(self):
        return self.snake[0] in self.border

    def is_over(self):
        return self.self_bite() or self.hit_border() or (self.max_turn > 0 and self.turn >= self.max_turn)

    def is_won(self):
        if self.max_turn > 0:
            return self.turn >= self.max_turn
        else:
            return len(self.snake) > self.snake_length

class Tromis(Game):

    def __init__(self, width=10, height=12, max_turn=None):
        self.width = width
        self.height = height
        self.max_turn = max_turn
        self.reset()

    @property
    def name(self):
        return "Tromis"
    @property
    def nb_actions(self):
        return 5
        
    def reset(self):
        self.lost = False
        self.turn = 0
        self.removed_rows = 0
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self._new_piece()
        self.drop_counter = 2

    trominos = ( 
                 ( ((0,0), (0,-1), (0,1)),
                   ((0,0), (-1,0), (1,0)),
                   ((0,0), (0,-1), (0,1)),
                   ((0,0), (-1,0), (1,0))
                 ),
                 ( ((0,0), (0,1), (1,0)),
                   ((0,0), (1,0), (1,1)),
                   ((0,1), (1,0), (1,1)),
                   ((0,0), (0,1), (1,1))
                 )
               )
    
    def _new_piece(self):
        self.p_type = random.randrange(2)
        self.p_orient = random.randrange(4)
        start_range = 0-min(t[1] for t in self.trominos[self.p_type][self.p_orient])
        end_range = self.width-max(t[1] for t in self.trominos[self.p_type][self.p_orient])
        self.p_row    = 1 if self.p_type == 0 else 0
        self.p_column = random.randrange(start_range, end_range)
        return

    def _valid_position(self, row, column, orient):
        for r,c in self.trominos[self.p_type][orient]:
            if (row+r < 0) or (row+r >= self.height):
                return False
            if (column+c < 0) or (column+c >= self.width):
                return False
            if self.grid[row+r][column+c] != 0:
                return False
        return True

    def _remove_completed_rows(self):
        self.removed_rows = 0
        while True:
            for i in range(len(self.grid)):
                if all(self.grid[i]): break
            else:
                break
            self.grid = [[0 for _ in range(self.width)]] + self.grid[0:i] + self.grid[i+1:]
            self.removed_rows = 1

    def play(self, action):
        if self.lost:
            return
        nrow,ncol,nori = self.p_row, self.p_column, self.p_orient
        
        actions = [lambda r,c,o: (r,c-1,o),
                   lambda r,c,o: (r,c+1,o),
                   lambda r,c,o: (r,c,o),
                   lambda r,c,o: (r,c,(o+1)%4),
                   lambda r,c,o: (r,c,(o-1)%4)]
        nrow,ncol,nori = actions[action](self.p_row, self.p_column, self.p_orient)

        if self._valid_position(nrow,ncol,nori):
            self.p_column, self.p_row, self.p_orient = ncol, nrow, nori

        if not self._valid_position(self.p_row+1, self.p_column, self.p_orient):
            for r,c in self.trominos[self.p_type][self.p_orient]:
                self.grid[self.p_row+r][self.p_column+c] = 1
            self._remove_completed_rows()
            self._new_piece()
        else:
            self.p_row += 1
        
        if not self._valid_position(self.p_row, self.p_column, self.p_orient):
            self.lost = True
        self.turn += 1
        
        return self.get_score()

    def get_score(self):
        if self.is_over() and self.lost:
            return -1
        elif self.is_over():
            return 1
        return self.removed_rows

    def get_state(self):
        canvas = np.zeros((self.height,self.width,3))
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == 1:
                    canvas[i,j,:] = (1,1,0)
        if self.lost: color = (1,0,0)
        elif self.p_type == 0: color = (0,0,1)
        else: color = (0,1,0)
        for r,c in self.trominos[self.p_type][self.p_orient]:
            canvas[self.p_row+r,self.p_column+c,:] = color
        return canvas
    
    def is_over(self):
        if self.max_turn is not None and self.turn >= self.max_turn:
            return True
        else:
            return self.lost

    def is_won(self):
        return self.is_over() and not self.lost


class Memory:

    def __init__(self):
        pass

    def remember(self, S, a, r, S_prime, game_over):
        pass

    def get_batch(self, model, batch_size):
        pass


class ExperienceReplay(Memory):

    def __init__(self, memory_size=100, fast=True):
        self.fast = fast
        self.memory = []
        self._memory_size = memory_size

    def remember(self, s, a, r, s_prime, game_over):
        self.input_shape = s.shape[1:]
        self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(), np.array(r).flatten(), s_prime.flatten(), 1 * np.array(game_over).flatten()]))
        if self.memory_size > 0 and len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def get_batch(self, model, batch_size, gamma=0.9):
        if self.fast:
            return self.get_batch_fast(model, batch_size, gamma)
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        nb_actions = model.get_output_shape_at(0)[-1]
        samples = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)
        S = samples[:, 0 : input_dim]
        a = samples[:, input_dim]
        r = samples[:, input_dim + 1]
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2]
        r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
        game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
        S = S.reshape((batch_size, ) + self.input_shape)
        S_prime = S_prime.reshape((batch_size, ) + self.input_shape)
        X = np.concatenate([S, S_prime], axis=0)
        Y = model.predict(X)
        Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
        delta = np.zeros((batch_size, nb_actions))
        a = np.cast['int'](a)
        delta[np.arange(batch_size), a] = 1
        targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)
        return S, targets

    @property
    def memory_size(self):
        return self._memory_size

    @memory_size.setter
    def memory_size(self, value):
        if value > 0 and value < self._memory_size:
            self.memory = self.memory[:value]
        self._memory_size = value

    def reset_memory(self):
        self.memory = []

    def set_batch_function(self, model, input_shape, batch_size, nb_actions, gamma):
        input_dim = np.prod(input_shape)
        samples = K.placeholder(shape=(batch_size, input_dim * 2 + 3))
        S = samples[:, 0 : input_dim]
        a = samples[:, input_dim]
        r = samples[:, input_dim + 1]
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2 : 2 * input_dim + 3]
        r = K.reshape(r, (batch_size, 1))
        r = K.repeat(r, nb_actions)
        r = K.reshape(r, (batch_size, nb_actions))
        game_over = K.repeat(game_over, nb_actions)
        game_over = K.reshape(game_over, (batch_size, nb_actions))
        S = K.reshape(S, (batch_size, ) + input_shape)
        S_prime = K.reshape(S_prime, (batch_size, ) + input_shape)
        X = K.concatenate([S, S_prime], axis=0)
        Y = model(X)
        Qsa = K.max(Y[batch_size:], axis=1)
        Qsa = K.reshape(Qsa, (batch_size, 1))
        Qsa = K.repeat(Qsa, nb_actions)
        Qsa = K.reshape(Qsa, (batch_size, nb_actions))
        delta = K.reshape(self.one_hot(a, nb_actions), (batch_size, nb_actions))
        targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)
        self.batch_function = K.function(inputs=[samples], outputs=[S, targets])

    def one_hot(self, seq, num_classes):
        return K.one_hot(K.reshape(K.cast(seq, "int32"), (-1, 1)), num_classes)

    def get_batch_fast(self, model, batch_size, gamma):
        if len(self.memory) < batch_size:
            return None
        samples = np.array(random.sample(self.memory, batch_size))
        if not hasattr(self, 'batch_function'):
            self.set_batch_function(model, self.input_shape, batch_size, model.get_output_shape_at(0)[-1], gamma)
        S, targets = self.batch_function([samples])
        return S, targets

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
            
            print(" Loss {:.4f} | Win {:5.2%} | Avg/Max Score {: 5.2f}/{: 5.2f} | Store {:>5d}".format(loss, win_ratio, avg_score, float(max_score), len(self.memory.memory)))
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


