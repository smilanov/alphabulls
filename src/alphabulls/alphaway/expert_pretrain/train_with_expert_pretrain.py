# alphabulls/train.py
import os

import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import random
import numpy as np
from tqdm import tqdm

from alphabulls.alphaway.expert_pretrain.pretrain_with_expert import OUTPUT_MODEL_FILE
from alphabulls.alphaway.self_play_Dirichlet_noise import play_game
from alphabulls.alphaway.LSTM_net import AlphaBullsNet
from alphabulls.utils import DEVICE, MAX_GUESSES

# --- Hyperparameters ---
NUM_EPOCHS = 50
GAMES_PER_EPOCH = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
REPLAY_BUFFER_SIZE = 50000
PRETRAINED_MODEL_PATH = OUTPUT_MODEL_FILE #"pretrained_model.pth" # Path to our new model

def main():
    # 1. Initialize model, optimizer, and replay buffer
    model = AlphaBullsNet().to(DEVICE)

    # --- NEW: LOAD PRE-TRAINED WEIGHTS IF THEY EXIST ---
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Found pre-trained model at {PRETRAINED_MODEL_PATH}. Loading weights.")
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    else:
        print("No pre-trained model found. Starting training from scratch.")
    # ----------------------------------------------------

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Loss functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    print(f"Starting training on {DEVICE}...")

    # 2. Main training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # --- Self-Play Phase ---
        model.eval()
        print("Generating game data through self-play...")

        new_game_experiences = []
        successful_games = 0
        failed_games = 0
        total_guesses = []
        for _ in tqdm(range(GAMES_PER_EPOCH)):
            game_data, num_guesses = play_game(model)
            # --- THE FILTERING LOGIC ---
            if num_guesses < MAX_GUESSES:  # This was a successful game
                new_game_experiences.extend(game_data)
                successful_games += 1
                total_guesses.append(num_guesses)
            else:  # This was a failed game
                # Only add failed games with a small probability, e.g., 10%
                if random.random() < 0.5: # add almost all failed games; 90%
                    new_game_experiences.extend(game_data)
                    total_guesses.append(num_guesses)
                    failed_games += 1

        replay_buffer.extend(new_game_experiences)
        print(f"Added experiences from {successful_games} successful games and a {failed_games} failed games.")

        avg_guesses = np.mean(total_guesses)
        print(f"Average guesses per game: {avg_guesses:.2f}")

        # --- Training Phase ---
        if len(replay_buffer) < BATCH_SIZE:
            print("Replay buffer too small, skipping training for this epoch.")
            continue

        model.train()
        print("Training the model...")

        num_batches = len(replay_buffer) // BATCH_SIZE
        for _ in tqdm(range(num_batches)):
            # Sample a batch from the replay buffer
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, target_policies, target_values = zip(*batch)

            states = torch.stack(states).to(DEVICE)
            target_policies = torch.tensor(np.array(target_policies), dtype=torch.float32).to(DEVICE)
            target_values = torch.tensor(np.array(target_values), dtype=torch.float32).unsqueeze(1).to(DEVICE)

            # Forward pass
            pred_policy_logits, pred_values = model(states)

            # Calculate loss
            policy_loss = policy_loss_fn(pred_policy_logits, target_policies)
            value_loss = value_loss_fn(pred_values, target_values)
            total_loss = policy_loss + value_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch finished. Final Loss: {total_loss.item():.4f}")

        # Save model checkpoint
        # torch.save(model.state_dict(), f"../../../models_saved/alphabulls_model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()