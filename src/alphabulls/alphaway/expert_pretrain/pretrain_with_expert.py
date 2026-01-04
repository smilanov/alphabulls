# alphabulls/pretrain_with_expert.py

import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import random
from tqdm import tqdm

from alphabulls.alphaway.LSTM_net import AlphaBullsNet
from alphabulls.utils import state_to_tensor, DEVICE

# --- Configuration ---
EXPERT_DATA_FILE = "expert_data.pkl"
OUTPUT_MODEL_FILE = "pretrained_model.pth"
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def main():
    print(f"Loading expert data from {EXPERT_DATA_FILE}...")
    with open(EXPERT_DATA_FILE, "rb") as f:
        expert_data = pickle.load(f)

    print(f"Loaded {len(expert_data)} state-action pairs.")

    # Initialize model, optimizer, and loss function
    model = AlphaBullsNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # For behavioral cloning, we only care about the policy head.
    policy_loss_fn = nn.CrossEntropyLoss()

    print("Starting supervised pre-training (Behavioral Cloning)...")

    for epoch in range(NUM_EPOCHS):
        random.shuffle(expert_data)  # Shuffle data each epoch
        total_loss = 0

        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        for i in tqdm(range(0, len(expert_data), BATCH_SIZE)):
            batch = expert_data[i:i + BATCH_SIZE]
            if not batch:
                continue

            histories, target_action_indices = zip(*batch)

            # Convert histories to state tensors
            states = torch.stack([state_to_tensor(h) for h in histories]).to(DEVICE)

            # Convert target actions to a tensor
            target_actions = torch.tensor(target_action_indices, dtype=torch.long).to(DEVICE)

            # --- Standard Supervised Learning Loop ---
            optimizer.zero_grad()

            # We only need the policy logits for this phase
            pred_policy_logits, _ = model(states)

            loss = policy_loss_fn(pred_policy_logits, target_actions)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(expert_data) / BATCH_SIZE)
        print(f"Epoch finished. Average Loss: {avg_loss:.4f}")

    # Save the pre-trained model
    torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
    print(f"\nPre-trained model saved to {OUTPUT_MODEL_FILE}")


if __name__ == "__main__":
    main()