import pickle
import numpy as np

from tictactoe.model import Model
from tictactoe.reward_function import reward_function

with open("tictactoe/value_function.pkl", "rb") as f:
    value_function = pickle.load(f)

SYMBOLS = {0: " ", 1: "X", -1: "O"}


def render(board):
    rows = []
    for i, row in enumerate(board):
        rows.append(" | ".join(SYMBOLS[int(v)] for v in row))
        if i < 2:
            rows.append("-" * 9)
    print("\n" + "\n".join(rows) + "\n")


def agent_move(model):
    best_value = -float("inf")
    best_action = None
    for action in model.available_moves():
        next_state = np.where(model.state == 0, model.state + action, model.state)
        key = tuple(map(tuple, next_state.astype(int)))
        value = value_function.get(key, 0.0)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def player_move(model):
    available = list(zip(*np.where(model.state == 0)))
    while True:
        try:
            raw = input("🎯 Your move (row col, e.g. '1 2'): ")
            r, c = map(int, raw.strip().split())
            coord = (r, c)
            if coord not in available:
                print("⚠️  Cell already taken or out of range, try again.")
                continue
            action = np.zeros((3, 3))
            action[r, c] = -1
            return action
        except (ValueError, IndexError):
            print("⚠️  Invalid input. Enter row and column as two numbers (0-2).")


def check_end(model):
    r = reward_function(model.state)
    if r == 1:
        render(model.state)
        print("🤖 X wins! Better luck next time.\n")
        return True
    if r == -1:
        render(model.state)
        print("🎉 O wins! You beat the agent!\n")
        return True
    if not list(zip(*np.where(model.state == 0))):
        render(model.state)
        print("🤝 It's a draw!\n")
        return True
    return False


def play():
    model = Model("tictacboard")
    print("\n🎮 Tic-Tac-Toe  |  You are O, agent is X  |  X goes first")
    print("   Coordinates: row and column from 0 to 2\n")
    print("   0 1 2  ← col")
    print("0  . . .")
    print("1  . . .")
    print("2  . . .")
    print("↑ row\n")

    while True:
        # Agent (X) move
        print("🤖 Agent (X) is thinking...")
        action = agent_move(model)
        model.update(action)
        render(model.state)
        if check_end(model):
            break

        # Player (O) move
        action = player_move(model)
        model.update(action)
        render(model.state)
        if check_end(model):
            break


if __name__ == "__main__":
    play()
