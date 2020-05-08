import random
import numpy as np
from .game import Game

class Snake(Game):

    def __init__(self, grid_size=10, snake_length=3, max_turn=100):
        self.grid_size = grid_size
        self.snake_length = snake_length
        self.max_turn = max_turn
        self.reset()

    @property
    def name(self):       return "Snake"
    @property
    def nb_actions(self): return 3
    @property
    def actions(self):    return {0: 'rotateleft', 1: 'forward', 2: 'rotateright'}

    def play(self, action):
        if self.is_over() or (action not in range(self.nb_actions)):
            return
        self.scored = False
        self.turn += 1
        self.move_snake(action)
        if self.is_over():
            self.snake.pop()
            self.game_over = True
        elif self.fruit != self.snake[0]:
            self.snake.pop()
        else:
            self.scored = True
            self.drop_fruit()

        return (self.get_frame(), self.get_score(), self.is_over())

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
        if ((h[1] < n[1] and action == 1) or
            (h[0] < n[0] and action == 0) or
            (h[0] > n[0] and action == 2)):
            p = (h[0], h[1]-1)
        elif ((h[1] < n[1] and action == 2) or
              (h[1] > n[1] and action == 0) or
              (h[0] < n[0] and action == 1)):
            p = (h[0]-1, h[1])
        elif ((h[1] < n[1] and action == 0) or
              (h[1] > n[1] and action == 2) or
              (h[0] > n[0] and action == 1)):
            p = (h[0]+1, h[1])
        elif ((h[1] > n[1] and action == 1) or
              (h[0] < n[0] and action == 2) or
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
        if self.had_bumped():
            score = -1
        elif self.scored:
            score = 1
        else:
            score = 0
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

    def had_bumped(self):
        return self.self_bite() or self.hit_border()

    def timeout(self):
        return (self.max_turn is not None) and (self.turn >= self.max_turn)

    def is_over(self):
        return self.had_bumped() or self.timeout()

    def has_eaten(self):
        return len(self.snake) > self.snake_length

    def is_won(self):
        return self.is_over() and not self.had_bumped() and self.has_eaten()

