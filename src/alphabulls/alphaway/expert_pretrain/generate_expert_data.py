# alphabulls/generate_expert_data.py

import pickle
from tqdm import tqdm

from alphabulls.classic.classic_solver import KnuthSolver
from alphabulls.game_env import BullsAndCowsGame
from alphabulls.utils import CODE_TO_INDEX, NUM_DIGITS

# --- Configuration ---
NUM_GAMES_TO_GENERATE = 2000  # Start with a smaller number like 200, can increase later
OUTPUT_FILE = "expert_data.pkl"


def generate_one_game_trajectory():
    """Plays one game with the KnuthSolver and records its decisions."""
    game = BullsAndCowsGame()
    solver = KnuthSolver()

    trajectory = []
    # First guess is always a good starting point like "0123" (if NUM_DIGITS=4), "01" (if NUM_DIGITS=2), etc.
    guess = ''.join(str(i) for i in range(NUM_DIGITS))

    for _ in range(10):  # Loop for each turn
        # Get the current state (the history before making the guess)
        current_history = game.get_current_state()

        # The expert's action is the chosen guess
        action_idx = CODE_TO_INDEX[guess]

        # Store this "flashcard": (situation, correct_action)
        trajectory.append((current_history, action_idx))

        # Make the move in the environment
        feedback, is_won = game.make_guess(guess)

        if is_won:
            return trajectory

        # Update the solver's internal state and get the next best guess
        solver.update(guess, feedback)
        guess = solver._find_next_best_guess()
        if not guess:  # Should not happen in a valid game
            return None

    return None  # Game was not solved (should not happen with Knuth)


def main():
    print(f"Generating expert game data for {NUM_GAMES_TO_GENERATE} games...")
    print("This will be VERY SLOW as it uses the optimal solver.")

    all_trajectories = []
    for _ in tqdm(range(NUM_GAMES_TO_GENERATE)):
        trajectory = generate_one_game_trajectory()
        if trajectory:
            all_trajectories.extend(trajectory)  # Flatten into a list of (state, action) tuples

    print(f"\nGenerated a total of {len(all_trajectories)} expert state-action pairs.")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_trajectories, f)

    print(f"Expert data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()