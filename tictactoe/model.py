import numpy as np

class Model:
    def __init__(self,name):
        self.name = name
        self.state = np.zeros((3,3))
        self.move = 0
        self.player = "x"
        self.state_history = [self.state.copy()]

    def update(self, action):
        # Action is the coordinate of the next move
        self.state = np.where(self.state == 0, self.state + action, self.state)
        self.state_history.append(self.state.copy())
        self.move += 1
        self.player = "o" if self.player == "x" else "x"

    def available_moves(self):
        available = list(zip(*np.where(self.state == 0)))
        return [action_function(coord, self.player) for coord in available]

    def reset(self, move):
        self.state = self.state_history[move]
        self.state_history = self.state_history[:move + 1]
        self.player = "x" if move % 2 == 0 else "o"


def action_function(coordinates,sign):
    action = np.zeros((3,3))
    if sign == "x":
        action[coordinates[0], coordinates[1]] = 1
        return action
    else:
        action[coordinates[0], coordinates[1]] = -1
        return action