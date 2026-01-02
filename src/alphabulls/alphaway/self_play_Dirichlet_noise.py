# alphabulls/self_play.py

import torch
import torch.nn.functional as F
import numpy as np
from alphabulls.game_env import BullsAndCowsGame
from alphabulls.utils import state_to_tensor, INDEX_TO_CODE, CODE_TO_INDEX, MAX_GUESSES

# --- NEW: Add exploration parameters ---
DIRICHLET_ALPHA = 0.3
EXPLORATION_FRACTION = 0.25


def play_game(model):
    """
    Plays one full game of Bulls and Cows using the model to decide moves.
    Returns a list of training examples: (state, policy, value).
    """
    model.eval()  # Set the model to evaluation mode
    game = BullsAndCowsGame()
    game_history = []  # Stores (state_tensor, action_idx) for the whole game

    while True:
        # 1. Get current state and predict policy/value
        state_tensor = state_to_tensor(game.get_current_state())

        with torch.no_grad():
            policy_logits, _ = model(state_tensor.unsqueeze(0))

        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # ------------------- KEY CHANGE: ADD DIRICHLET NOISE FOR EXPLORATION -------------------
        # This is crucial. It forces the model to try new moves.
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(policy_probs))
        policy_probs = (1 - EXPLORATION_FRACTION) * policy_probs + EXPLORATION_FRACTION * noise
        # ------------------------------------------------------------------------------------

        # ------------------- THE FIX: RE-NORMALIZE THE PROBABILITIES -------------------
        # This corrects for tiny floating-point errors that can make the sum not exactly 1.
        policy_probs /= np.sum(policy_probs)
        # -----------------------------------------------------------------------------

        # 2. Choose an action based on the noisy policy
        action_idx = np.random.choice(len(policy_probs), p=policy_probs)
        guess = INDEX_TO_CODE[action_idx]

        # Prevent the model from making the same guess twice in a game
        # This is a simple heuristic that helps a lot in Bulls and Cows
        if guess in [h[0] for h in game.history]:
            # If guess is repeated, just pick the next best non-repeated option
            sorted_probs_idx = np.argsort(policy_probs)[::-1]
            for idx in sorted_probs_idx:
                if INDEX_TO_CODE[idx] not in [h[0] for h in game.history]:
                    action_idx = idx
                    break
            guess = INDEX_TO_CODE[action_idx]

        game_history.append((state_tensor, action_idx))

        # 3. Make the move in the environment
        _, is_won = game.make_guess(guess)

        if is_won:
            # print(f"Solved the game in {game.guess_count} guesses! The code was {game.secret_code_str}. The game_history is {game.history}")
            break
        elif game.guess_count >= MAX_GUESSES:  # End game if won or too long
            break

    # 4. Process game results to create training data
    training_data = []
    num_guesses = game.guess_count

    for i, (state, action_idx) in enumerate(game_history):
        # if num_guesses == MAX_GUESSES:
        #     continue  # Skip training data if the game was not solved
        target_value = float(num_guesses - i)

        target_policy = np.zeros(len(policy_probs), dtype=np.float32)
        target_policy[action_idx] = 1.0

        training_data.append((state, target_policy, target_value))

    return training_data, num_guesses