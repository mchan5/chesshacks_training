# main.py - ChessHacks bot using AlphaZero model
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.pgn
import random
import io
import torch
import numpy as np
import os
import sys

# ----------------------------
# Import training code
# ----------------------------
# Add the training repo to path so we can import train module
TRAINING_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ash-hf", "src")
sys.path.insert(0, TRAINING_PATH)

try:
    from train import ChessNet, MCTS, move_to_index
    print("✓ Successfully imported training modules")
except Exception as e:
    print(f"❌ Failed to import training modules: {e}")
    print(f"Looking in: {TRAINING_PATH}")
    ChessNet = None
    MCTS = None
    move_to_index = None

# ----------------------------
# Load AlphaZero model
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ash-hf", "weights", "best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
mcts = None

if ChessNet is not None and os.path.exists(MODEL_PATH):
    try:
        print(f"Loading AlphaZero model from {MODEL_PATH}...")

        # Create model with optimized architecture
        model = ChessNet(
            input_channels=18,
            num_filters=64,
            num_residual_blocks=3
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # Handle torch.compile prefix if present
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                cleaned_state_dict[k[len('_orig_mod.'):]] = v
            else:
                cleaned_state_dict[k] = v

        model.load_state_dict(cleaned_state_dict)
        model.eval()

        # Create MCTS instance
        # Use fewer simulations for speed in online play
        mcts = MCTS(model, num_simulations=50, batch_size=8)

        iteration = checkpoint.get('iteration', 'unknown')
        print(f"✓ Model loaded successfully (iteration {iteration})")
        print(f"✓ Using device: {device}")
        print(f"✓ MCTS simulations: 50")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
        mcts = None
else:
    if ChessNet is None:
        print("❌ Training modules not found - using random moves")
    elif not os.path.exists(MODEL_PATH):
        print(f"❌ Model checkpoint not found at {MODEL_PATH}")
        print("Download the model first:")
        print("  cd ash-hf/src")
        print("  modal run download_model.py --mode download")


# ----------------------------
# Get move from AlphaZero model
# ----------------------------
def get_move_from_model(board: chess.Board) -> tuple[Move, dict]:
    """
    Get best move using AlphaZero MCTS.

    Returns:
        tuple: (best_move, move_probabilities_dict)
    """
    if model is None or mcts is None or move_to_index is None:
        raise RuntimeError("Model not loaded")

    # Run MCTS search
    visit_counts = mcts.search(board)

    # Get legal moves and their visit counts
    legal_moves = list(board.legal_moves)
    move_visits = {}

    for move in legal_moves:
        idx = move_to_index(move)
        move_visits[move] = visit_counts[idx]

    # Normalize to probabilities
    total_visits = sum(move_visits.values())
    if total_visits > 0:
        move_probs = {m: v / total_visits for m, v in move_visits.items()}
    else:
        # If no visits (shouldn't happen), uniform distribution
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}

    # Pick move with highest visit count
    best_move = max(move_visits, key=move_visits.get)

    return best_move, move_probs


# ----------------------------
# Fallback: Random move
# ----------------------------
def get_random_move(board: chess.Board) -> tuple[Move, dict]:
    """Fallback random move selection"""
    legal_moves = list(board.legal_moves)
    move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
    best_move = random.choice(legal_moves)
    return best_move, move_probs


# ----------------------------
# ChessHacks entrypoint
# ----------------------------
@chess_manager.entrypoint
def bot(ctx: GameContext):
    print("\n" + "="*50)
    print("ChessHacks Bot - AlphaZero Model")
    print("="*50)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    print(f"Position: {ctx.board.fen()}")
    print(f"Legal moves: {len(legal_moves)}")

    # Try to use model, fallback to random
    try:
        if model is not None:
            print("Using AlphaZero model (50 MCTS simulations)...")
            move_obj, move_probs = get_move_from_model(ctx.board)
            print(f"✓ Model selected: {move_obj.uci()}")

            # Show top 3 moves
            sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top moves:")
            for m, prob in sorted_moves:
                print(f"  {m.uci()}: {prob:.2%}")

        else:
            print("⚠ Model not loaded, using random move")
            move_obj, move_probs = get_random_move(ctx.board)

    except Exception as e:
        print(f"❌ Model error: {e}")
        print("Falling back to random move")
        move_obj, move_probs = get_random_move(ctx.board)

    # Verify move is legal
    if move_obj not in legal_moves:
        print(f"❌ Model returned illegal move: {move_obj.uci()}")
        print("Using random legal move")
        move_obj, move_probs = get_random_move(ctx.board)

    # Log probabilities for devtools
    ctx.logProbabilities(move_probs)

    print(f"Playing: {move_obj.uci()}")
    print("="*50 + "\n")

    return move_obj


@chess_manager.reset
def reset(ctx: GameContext):
    """Called when a new game begins"""
    print("\n🔄 New game starting - resetting bot state")
    # MCTS is stateless, no reset needed
    pass
