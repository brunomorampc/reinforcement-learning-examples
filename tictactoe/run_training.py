import logging
import pickle
import numpy as np

from tictactoe.model import Model
from tictactoe.reward_function import reward_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

states = np.load("tictactoe/states.npy")
logger.info("📂 Loaded %d states", len(states))

gamma = 1.0
toll = 0.001

# Initialize value function with immediate rewards
state_value_map = {}
for state in states:
    state_value_map[tuple(map(tuple, state.astype(int)))] = float(reward_function(state))
logger.info("🗺️  Value map initialised  |  terminal states: %d",
            sum(1 for v in state_value_map.values() if v != 0))

model = Model("tictacboard")
error = float("inf")
sweep = 0

while error > toll:
    error = 0
    sweep += 1
    for state in states:

        state_key = tuple(map(tuple, state.astype(int)))

        # Terminal states have a fixed value
        if reward_function(state) != 0:
            continue

        # Set up model at this state
        model.state = state.copy()
        model.state_history = [state.copy()]
        model.move = 0
        x_count = int(np.sum(state == 1))
        o_count = int(np.sum(state == -1))
        model.player = "x" if x_count == o_count else "o"
        current_player = model.player

        moves = model.available_moves()
        if not moves:  # full board, draw — value stays 0
            continue

        next_values = []
        for action in moves:
            model.update(action)
            next_key = tuple(map(tuple, model.state.astype(int)))
            next_values.append(state_value_map[next_key])
            model.reset(0)
            model.player = current_player  # restore for next action

        # X maximises, O minimises (from X's perspective)
        new_value = gamma * (max(next_values) if current_player == "x" else min(next_values))

        error += abs(state_value_map[state_key] - new_value)
        state_value_map[state_key] = new_value

    logger.info("🔄 Sweep %d  |  📉 error: %.6f", sweep, error)

logger.info("✅ Converged after %d sweeps", sweep)

with open("tictactoe/value_function.pkl", "wb") as f:
    pickle.dump(state_value_map, f)
logger.info("💾 Value function saved to tictactoe/value_function.pkl")
