# alphabulls/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import random
import numpy as np
from tqdm import tqdm

from alphabulls.alphaway.self_play import play_game
from alphabulls.alphaway.neural_net import AlphaBullsNet
from alphabulls.utils import DEVICE

# --- Hyperparameters ---
NUM_EPOCHS = 50
GAMES_PER_EPOCH = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 50000


def main():
    # 1. Initialize model, optimizer, and replay buffer
    model = AlphaBullsNet().to(DEVICE)
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
        total_guesses = []
        print("Generating game data through self-play...")
        for _ in tqdm(range(GAMES_PER_EPOCH)):
            game_data, num_guesses = play_game(model)
            replay_buffer.extend(game_data)
            total_guesses.append(num_guesses)

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
        torch.save(model.state_dict(), f"alphabulls_model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()