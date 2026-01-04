# alphabulls/classic_solver.py

from alphabulls.utils import ALL_POSSIBLE_CODES, code_to_tensor, CODE_TO_INDEX, INDEX_TO_CODE, NUM_DIGITS
from collections import defaultdict
import time
import re  # Import regular expressions for parsing user input


class KnuthSolver:
    """
    A classic, optimal solver based on Knuth's 5-guess algorithm (minimax).
    The goal is to make a guess that minimizes the size of the largest
    remaining set of possibilities.
    """

    def __init__(self):
        self.possible_codes = set(ALL_POSSIBLE_CODES)

    def _get_feedback(self, guess, secret):
        """Calculates feedback for two codes without game state."""
        bulls = sum(1 for g, s in zip(guess, secret) if g == s)
        cows = len(set(guess) & set(secret)) - bulls
        return (bulls, cows)

    def update(self, last_guess, feedback):
        """Filter the set of possible codes based on the last feedback."""
        if not self.possible_codes:
            return
        self.possible_codes = {
            code for code in self.possible_codes
            if self._get_feedback(last_guess, code) == feedback
        }

    def solve(self):
        """Solves a game by interacting with a game environment (automated)."""
        from alphabulls.game_env import BullsAndCowsGame

        game = BullsAndCowsGame()
        print(f"Classic Solver starting. Secret code is {game.secret_code_str}")

        # First guess is always a good starting point like "0123" (if NUM_DIGITS=4), "01" (if NUM_DIGITS=2), etc.
        guess = ''.join(str(i) for i in range(NUM_DIGITS))

        for i in range(1, 11):
            feedback, is_won = game.make_guess(guess)
            print(f"Guess {i}: {guess}, Feedback: {feedback[0]}B {feedback[1]}C")

            if is_won:
                print(f"Solved in {i} guesses!")
                return i

            self.update(guess, feedback)
            guess = self._find_next_best_guess()

    def _find_next_best_guess(self):
        """The core minimax logic. Very computationally expensive."""
        if not self.possible_codes:
            return None
        if len(self.possible_codes) == 1:
            return list(self.possible_codes)[0]

        min_max_partition = float('inf')
        best_guess = ""

        # Check all possible guesses, even ones we know are wrong
        # To speed up, we can limit the search space for a small trade-off in optimality
        search_space = ALL_POSSIBLE_CODES if len(self.possible_codes) > 1000 else self.possible_codes

        for guess in search_space:
            partitions = defaultdict(int)
            for secret in self.possible_codes:
                feedback = self._get_feedback(guess, secret)
                partitions[feedback] += 1

            max_partition = max(partitions.values())
            if max_partition < min_max_partition:
                min_max_partition = max_partition
                best_guess = guess

        return best_guess

    # ------------------- NEW METHOD FOR INTERACTIVE PLAY -------------------
    def play_interactively(self):
        """
        Drives the solver by asking the user for feedback.
        """
        print("\n--- Interactive Bulls and Cows Solver ---")
        print("Think of a 4-digit number with unique digits (e.g., 1234).")
        print("I will try to guess it. For each guess, provide the feedback.")
        print("For example, enter '2B 1C' for 2 bulls and 1 cow.")
        print("--------------------------------------------------")

        # First guess is always a good starting point like "0123" (if NUM_DIGITS=4), "01" (if NUM_DIGITS=2), etc.
        guess = ''.join(str(i) for i in range(NUM_DIGITS))

        for i in range(1, 11):
            if not guess:
                print("\nError: I could not find a consistent number based on your feedback.")
                print("Please double-check your previous answers and try again.")
                return

            print(f"\nMy guess #{i}: {guess}")

            # --- User input loop with validation ---
            while True:
                try:
                    user_input = input("Your feedback (e.g., '1B 2C'): ").upper().strip()

                    # Use regex to parse input like "1B 2C", "1B2C", "1 2" etc.
                    parts = re.findall(r'\d+', user_input)
                    if len(parts) != 2:
                        raise ValueError("Invalid format. Please enter two numbers for Bulls and Cows.")

                    bulls = int(parts[0])
                    cows = int(parts[1])

                    if not (0 <= bulls <= NUM_DIGITS and 0 <= cows <= NUM_DIGITS and bulls + cows <= NUM_DIGITS):
                        raise ValueError("Invalid feedback. Bulls and Cows must be between 0-4 and their sum cannot exceed 4.")

                    feedback = (bulls, cows)
                    break  # Exit validation loop if input is good
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}. Please try again.")

            # --- Check for win condition ---
            if feedback == (4, 0):
                print(f"\nGreat! I solved it in {i} guesses. The number was {guess}.")
                return

            # --- Update solver state and get next guess ---
            self.update(guess, feedback)
            guess = self._find_next_best_guess()

        print("\nI couldn't solve it in 10 guesses. There might be an inconsistency in the feedback provided.")


# --- MODIFIED MAIN BLOCK TO CHOOSE THE MODE ---
if __name__ == "__main__":

    # You can choose which mode to run
    mode = "interactive"  # or "solve"
    # mode = "solve"  # or "solve"

    if mode == "interactive":
        solver = KnuthSolver()
        solver.play_interactively()
    else:
        start_time = time.time()
        solver = KnuthSolver()
        solver.solve()
        print(f"Automated solver took {time.time() - start_time:.2f} seconds.")