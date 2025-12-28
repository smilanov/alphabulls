# alphabulls/classic_solver.py

from alphabulls.utils import ALL_POSSIBLE_CODES, code_to_tensor
from collections import defaultdict
import time


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
        self.possible_codes = {
            code for code in self.possible_codes
            if self._get_feedback(last_guess, code) == feedback
        }

    def solve(self):
        """Solves a game by interacting with a game environment."""
        from alphabulls.game_env import BullsAndCowsGame

        game = BullsAndCowsGame()
        print(f"Classic Solver starting. Secret code is {game.secret_code_str}")

        # First guess is always a good starting point like "0123"
        guess = "0123"

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
        for guess in ALL_POSSIBLE_CODES:
            partitions = defaultdict(int)
            for secret in self.possible_codes:
                feedback = self._get_feedback(guess, secret)
                partitions[feedback] += 1

            max_partition = max(partitions.values())
            if max_partition < min_max_partition:
                min_max_partition = max_partition
                best_guess = guess

        return best_guess


# Example of running the classic solver
if __name__ == "__main__":
    start_time = time.time()
    solver = KnuthSolver()
    solver.solve()
    print(f"Classic solver took {time.time() - start_time:.2f} seconds.")