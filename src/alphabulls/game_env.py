# alphabulls/game_env.py

import random
from alphabulls.utils import NUM_DIGITS, code_to_tensor


class BullsAndCowsGame:
    """
    Manages the state and logic of a single Bulls and Cows game.
    """

    def __init__(self):
        self.secret_code_str = ''.join(random.sample('0123456789', NUM_DIGITS))
        self.secret_code = code_to_tensor(self.secret_code_str)
        self.history = []
        self.guess_count = 0

    def make_guess(self, guess_str):
        """
        Makes a guess, returns feedback, and updates game state.
        Returns: (bulls, cows), is_won
        """
        if len(guess_str) != NUM_DIGITS or len(set(guess_str)) != NUM_DIGITS:
            raise ValueError("Invalid guess format.")

        self.guess_count += 1
        guess = code_to_tensor(guess_str)

        bulls = sum(1 for i in range(NUM_DIGITS) if guess[i] == self.secret_code[i])

        # Cows are shared digits but not in the same position
        cows = len(set(guess) & set(self.secret_code)) - bulls

        feedback = (bulls, cows)
        self.history.append((guess_str, feedback))

        is_won = (bulls == NUM_DIGITS)
        return feedback, is_won

    def make_guess_external_user(self, guess_str):
        """
        Makes a guess, returns feedback, and updates game state.
        Returns: (bulls, cows), is_won
        """
        if len(guess_str) != NUM_DIGITS or len(set(guess_str)) != NUM_DIGITS:
            raise ValueError("Invalid guess format.")

        self.guess_count += 1
        guess = code_to_tensor(guess_str)

        bulls = sum(1 for i in range(NUM_DIGITS) if guess[i] == self.secret_code[i])

        # Cows are shared digits but not in the same position
        cows = len(set(guess) & set(self.secret_code)) - bulls

        feedback = (bulls, cows)
        self.history.append((guess_str, feedback))

        is_won = (bulls == NUM_DIGITS)
        return feedback, is_won

    def get_current_state(self):
        """Returns the game history, which defines the current state."""
        return self.history