"""
Hybrid training: Supervised pre-training + Self-play reinforcement learning.

This approach:
1. First trains on real chess games and puzzles (supervised learning)
2. Then continues with self-play (reinforcement learning)

Benefits:
- Faster initial learning (model starts with chess knowledge)
- Better final performance (combines human games + self-play)
- More efficient use of compute
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
from tqdm import tqdm

from train import ChessNet, encode_board, move_to_index, device
from train_continuous import continuous_training

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if running on Modal
if os.path.exists("/root/training_data") and os.path.exists("/root/src"):
    DATA_DIR = "/root/training_data"  # Modal: data baked into image
    WEIGHTS_DIR = "/root/weights"  # Modal: weights on volume
else:
    DATA_DIR = os.path.join(SCRIPT_DIR, "../data")  # Local
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "../weights")  # Local

os.makedirs(WEIGHTS_DIR, exist_ok=True)

class SupervisedChessDataset(Dataset):
    """Dataset for supervised learning from games and puzzles"""

    def __init__(self, games_file=None, puzzles_file=None):
        self.data = []

        # Load games
        if games_file and os.path.exists(games_file):
            print(f"Loading games from {games_file}...")
            with open(games_file, 'r') as f:
                games = json.load(f)
                for game in tqdm(games, desc="Processing games"):
                    self._process_game(game)
            print(f"  Loaded {len(self.data)} positions from games")

        # Load puzzles
        if puzzles_file and os.path.exists(puzzles_file):
            print(f"Loading puzzles from {puzzles_file}...")
            with open(puzzles_file, 'r') as f:
                puzzles = json.load(f)
                for puzzle in tqdm(puzzles, desc="Processing puzzles"):
                    self._process_puzzle(puzzle)
            print(f"  Total positions: {len(self.data)}")

    def _process_game(self, game):
        """Convert game moves to training positions"""
        if 'moves' not in game:
            return

        board = chess.Board()
        moves = game['moves']

        for move_uci in moves:
            try:
                # Encode current position
                board_encoded = encode_board(board)

                # Parse move
                move = chess.Move.from_uci(move_uci)

                if move not in board.legal_moves:
                    break

                # Create policy target (one-hot for the correct move)
                policy_target = np.zeros(4096)
                policy_target[move_to_index(move)] = 1.0

                # Placeholder value (we don't know game outcome here)
                value_target = 0.0

                self.data.append({
                    'board': board_encoded,
                    'policy': policy_target,
                    'value': value_target
                })

                board.push(move)

            except Exception as e:
                break

    def _process_puzzle(self, puzzle):
        """Convert puzzle to training position"""
        try:
            fen = puzzle.get('fen')
            solution = puzzle.get('solution', [])

            if not fen or not solution:
                return

            board = chess.Board(fen)

            # Use first move of solution as target
            first_move_uci = solution[0]
            move = chess.Move.from_uci(first_move_uci)

            if move not in board.legal_moves:
                return

            # Encode board
            board_encoded = encode_board(board)

            # Create policy target
            policy_target = np.zeros(4096)
            policy_target[move_to_index(move)] = 1.0

            # Puzzles are tactical wins, so positive value
            value_target = 1.0 if board.turn == chess.WHITE else -1.0

            self.data.append({
                'board': board_encoded,
                'policy': policy_target,
                'value': value_target
            })

        except Exception as e:
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.FloatTensor(item['board']),
            torch.FloatTensor(item['policy']),
            torch.FloatTensor([item['value']])
        )


def supervised_pretrain(
    model,
    games_file,
    puzzles_file,
    epochs=5,
    batch_size=64,
    learning_rate=0.001
):
    """
    Pre-train model on supervised data (games + puzzles)
    """
    print("\n" + "="*60)
    print("SUPERVISED PRE-TRAINING")
    print("="*60)

    # Load dataset
    dataset = SupervisedChessDataset(games_file, puzzles_file)

    if len(dataset) == 0:
        print("❌ No training data found!")
        print("Run: python preprocess.py --games 10000 --puzzles 5000")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss functions
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_policy_loss = 0
        epoch_value_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for boards, target_policies, target_values in pbar:
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device).squeeze(1)

            optimizer.zero_grad()

            # Forward pass
            if scaler:
                with torch.amp.autocast('cuda'):
                    policy_logits, values = model(boards)
                    target_moves = torch.argmax(target_policies, dim=1)
                    policy_loss = criterion_policy(policy_logits, target_moves)
                    value_loss = criterion_value(values.squeeze(), target_values)
                    loss = policy_loss + value_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                policy_logits, values = model(boards)
                target_moves = torch.argmax(target_policies, dim=1)
                policy_loss = criterion_policy(policy_logits, target_moves)
                value_loss = criterion_value(values.squeeze(), target_values)
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'policy': f'{policy_loss.item():.4f}',
                'value': f'{value_loss.item():.4f}'
            })

        avg_loss = epoch_loss / len(dataloader)
        avg_policy = epoch_policy_loss / len(dataloader)
        avg_value = epoch_value_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} "
              f"(Policy: {avg_policy:.4f}, Value: {avg_value:.4f})")

    # Save pre-trained model
    pretrain_path = os.path.join(WEIGHTS_DIR, "pretrained_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'iteration': 0,
        'stage': 'supervised_pretrain'
    }, pretrain_path)

    print(f"\n✓ Pre-trained model saved to {pretrain_path}")
    print("="*60 + "\n")


def hybrid_training(
    supervised_epochs=5,
    selfplay_iterations=500,
    games_per_iteration=20,
    mcts_simulations=30
):
    """
    Full hybrid training pipeline:
    1. Supervised pre-training on games/puzzles
    2. Self-play reinforcement learning
    """
    print("\n" + "="*60)
    print("HYBRID TRAINING PIPELINE")
    print("="*60)
    print("Stage 1: Supervised pre-training on human games")
    print("Stage 2: Self-play reinforcement learning")
    print("="*60 + "\n")

    # Create model
    model = ChessNet(
        input_channels=18,
        num_filters=64,
        num_residual_blocks=3
    ).to(device)

    # Try to compile model
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("✓ Model compiled with torch.compile")
    except:
        print("⚠ torch.compile not available")

    # Stage 1: Supervised pre-training
    games_file = os.path.join(DATA_DIR, "games.json")
    puzzles_file = os.path.join(DATA_DIR, "puzzles.json")

    if os.path.exists(games_file) or os.path.exists(puzzles_file):
        supervised_pretrain(
            model,
            games_file if os.path.exists(games_file) else None,
            puzzles_file if os.path.exists(puzzles_file) else None,
            epochs=supervised_epochs
        )
    else:
        print("\n⚠ No supervised data found, skipping pre-training")
        print("To use supervised pre-training:")
        print("  python preprocess.py --games 10000 --puzzles 5000")
        print("\nProceeding directly to self-play...\n")

    # Save the model before self-play
    best_model_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'iteration': 0
    }, best_model_path)

    # Stage 2: Self-play reinforcement learning
    print("\n" + "="*60)
    print("STAGE 2: SELF-PLAY REINFORCEMENT LEARNING")
    print("="*60 + "\n")

    # This will load the pre-trained model and continue training
    continuous_training(
        num_iterations=selfplay_iterations,
        games_per_iteration=games_per_iteration,
        mcts_simulations=mcts_simulations,
        train_epochs_per_iteration=2,
        batch_size=64,
        learning_rate=0.001,
        checkpoint_every=10
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid chess AI training")
    parser.add_argument("--supervised-epochs", type=int, default=5,
                        help="Epochs for supervised pre-training (default: 5)")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Self-play iterations (default: 500)")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per self-play iteration (default: 20)")
    parser.add_argument("--simulations", type=int, default=30,
                        help="MCTS simulations per move (default: 30)")
    parser.add_argument("--skip-supervised", action="store_true",
                        help="Skip supervised pre-training, go straight to self-play")

    args = parser.parse_args()

    if args.skip_supervised:
        print("Skipping supervised pre-training, starting self-play directly...")
        continuous_training(
            num_iterations=args.iterations,
            games_per_iteration=args.games,
            mcts_simulations=args.simulations
        )
    else:
        hybrid_training(
            supervised_epochs=args.supervised_epochs,
            selfplay_iterations=args.iterations,
            games_per_iteration=args.games,
            mcts_simulations=args.simulations
        )
