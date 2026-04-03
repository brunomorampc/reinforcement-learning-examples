# Generate all possible states of a tic tac toe game.
# Assumptions:
## We start always with X
import logging
import numpy as np

from tictactoe.model import Model, action_function

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

model = Model("tictacboard")

counter = 0
max_positions = 9
coordinates = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
states = set()

logger.info("🎮 Starting tic-tac-toe state generation")
logger.debug("📐 Board size: %dx%d  |  Max positions: %d", 3, 3, max_positions)

for coord in coordinates:
    model.reset(0)
    logger.debug("🔄 [move 1 - X] Reset board, placing X at %s", coord)
    action = action_function(coord, "x")
    model.update(action)
    states.add(model.state.tobytes())
    remaining_coordinates = list(zip(*np.where(model.state == 0)))
    for coord in remaining_coordinates:
        model.reset(1)
        logger.debug("⭕ [move 2 - O] Placing O at %s", coord)
        action = action_function(coord, "o")
        model.update(action)
        states.add(model.state.tobytes())
        remaining_coordinates = list(zip(*np.where(model.state == 0)))
        for coord in remaining_coordinates:
            model.reset(2)
            action = action_function(coord, "x")
            model.update(action)
            states.add(model.state.tobytes())
            remaining_coordinates = list(zip(*np.where(model.state == 0)))
            for coord in remaining_coordinates:
                model.reset(3)
                action = action_function(coord, "o")
                model.update(action)
                states.add(model.state.tobytes())
                remaining_coordinates = list(zip(*np.where(model.state == 0)))
                for coord in remaining_coordinates:
                    model.reset(4)
                    action = action_function(coord, "x")
                    model.update(action)
                    states.add(model.state.tobytes())
                    remaining_coordinates = list(zip(*np.where(model.state == 0)))
                    for coord in remaining_coordinates:
                        model.reset(5)
                        action = action_function(coord, "o")
                        model.update(action)
                        states.add(model.state.tobytes())
                        remaining_coordinates = list(zip(*np.where(model.state == 0)))
                        for coord in remaining_coordinates:
                            model.reset(6)
                            action = action_function(coord, "x")
                            model.update(action)
                            states.add(model.state.tobytes())
                            remaining_coordinates = list(zip(*np.where(model.state == 0)))
                            for coord in remaining_coordinates:
                                model.reset(7)
                                action = action_function(coord, "o")
                                model.update(action)
                                states.add(model.state.tobytes())
                                remaining_coordinates = list(zip(*np.where(model.state == 0)))
                                for coord in remaining_coordinates:
                                    model.reset(8)
                                    action = action_function(coord, "x")
                                    model.update(action)
                                    states.add(model.state.tobytes())
                                    remaining_coordinates = list(zip(*np.where(model.state == 0)))

logger.info("✅ State generation complete  |  📦 Total states collected: %d", len(states))

states_array = np.array([np.frombuffer(s).reshape(3, 3) for s in states])
np.save("tictactoe/states.npy", states_array)
logger.info("💾 States saved to tictactoe/states.npy")


