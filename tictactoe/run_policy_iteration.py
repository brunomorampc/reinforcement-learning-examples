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

# --- Helpers ---

def state_key(state):
    return tuple(map(tuple, state.astype(int)))

def get_player(state):
    return "x" if int(np.sum(state == 1)) == int(np.sum(state == -1)) else "o"

def next_state(state, action):
    return np.where(state == 0, state + action, state)

# --- Initialise value function and model ---

state_value_map = {}
for state in states:
    state_value_map[state_key(state)] = float(reward_function(state))

model = Model("tictacboard")

def get_moves(state):
    model.state = state.copy()
    model.state_history = [state.copy()]
    model.move = 0
    model.player = get_player(state)
    return model.available_moves()

# --- Initialise policy: X states only, pick first available move ---

policy = {}
for state in states:
    if reward_function(state) != 0:
        continue
    if get_player(state) != "x":
        continue
    moves = get_moves(state)
    if moves:
        policy[state_key(state)] = moves[0]

logger.info("📋 Policy initialised for %d X-turn states", len(policy))

# --- Policy Iteration ---

iteration = 0
while True:
    iteration += 1

    # ── Policy Evaluation ──────────────────────────────────────────────────
    error = float("inf")
    eval_sweep = 0
    while error > toll:
        error = 0
        eval_sweep += 1
        for state in states:
            key = state_key(state)
            if reward_function(state) != 0:
                continue
            moves = get_moves(state)
            if not moves:
                continue

            if get_player(state) == "x":
                # Follow the current policy
                action = policy[key]
                ns = next_state(state, action)
                new_value = gamma * state_value_map[state_key(ns)]
            else:
                # O minimises
                next_values = [state_value_map[state_key(next_state(state, a))] for a in moves]
                new_value = gamma * min(next_values)

            error += abs(state_value_map[key] - new_value)
            state_value_map[key] = new_value

    logger.info("🔄 Iteration %d  |  📊 Policy evaluation done in %d sweeps", iteration, eval_sweep)

    # ── Policy Improvement ─────────────────────────────────────────────────
    policy_stable = True
    for state in states:
        key = state_key(state)
        if reward_function(state) != 0:
            continue
        if get_player(state) != "x":
            continue
        moves = get_moves(state)
        if not moves:
            continue

        best_action = max(moves, key=lambda a: state_value_map[state_key(next_state(state, a))])

        if not np.array_equal(policy[key], best_action):
            policy[key] = best_action
            policy_stable = False

    if policy_stable:
        logger.info("✅ Policy stable — converged after %d iterations", iteration)
        break
    else:
        logger.info("🔁 Policy updated, running another iteration")

# --- Save ---

with open("tictactoe/policy.pkl", "wb") as f:
    pickle.dump(policy, f)
logger.info("💾 Policy saved to tictactoe/policy.pkl")

with open("tictactoe/value_function_pi.pkl", "wb") as f:
    pickle.dump(state_value_map, f)
logger.info("💾 Value function saved to tictactoe/value_function_pi.pkl")
