import numpy as np

WINNING_STATES = [
    # Rows
    np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]),
    # Columns
    np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
    np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
    # Diagonals
    np.eye(3),
    np.fliplr(np.eye(3)),
]

def reward_function(state):
    for winning_state in WINNING_STATES:
        if np.sum(state * winning_state) == 3:
            return 1
    for winning_state in WINNING_STATES:
        if np.sum(state * winning_state) == -3:
            return -1
    return 0