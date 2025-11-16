#!/usr/bin/env python3
"""
Play chess against your trained model.
"""
import torch
import chess
import sys
import os

# Add src to path so we can import train
sys.path.insert(0, os.path.dirname(__file__))

from train import ChessNet, MCTS, move_to_index

def load_model(checkpoint_path='../weights/best_model.pt'):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model with optimized architecture
    model = ChessNet(
        input_channels=18,
        num_filters=64,
        num_residual_blocks=3
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    iteration = checkpoint.get('iteration', 'unknown')
    print(f"✓ Loaded model from iteration {iteration}")
    print(f"✓ Using device: {device}")

    return model, device

def get_model_move(board, model, simulations=100):
    """Get best move from model using MCTS"""
    mcts = MCTS(model, num_simulations=simulations)
    visit_counts = mcts.search(board)

    # Find legal move with highest visit count
    legal_moves = list(board.legal_moves)
    best_move = max(legal_moves, key=lambda m: visit_counts[move_to_index(m)])

    return best_move

def print_board(board):
    """Print board with nice formatting"""
    print()
    print("  a b c d e f g h")
    print(" ┌───────────────┐")
    for rank in range(7, -1, -1):
        print(f"{rank+1}│", end=" ")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                print(piece.symbol(), end=" ")
            else:
                print(".", end=" ")
        print(f"│{rank+1}")
    print(" └───────────────┘")
    print("  a b c d e f g h")
    print()

def play_game(model, human_color='white', simulations=100):
    """Play a game against the model"""
    board = chess.Board()

    print("\n" + "="*50)
    print("CHESS GAME vs AI")
    print("="*50)
    print(f"You are playing as: {human_color.upper()}")
    print(f"AI strength: {simulations} simulations (~{simulations//20} seconds per move)")
    print("\nEnter moves in UCI format (e.g., 'e2e4', 'e7e5')")
    print("Type 'quit' to exit, 'undo' to take back last move")
    print("="*50)

    move_history = []

    while not board.is_game_over():
        print_board(board)

        # Determine whose turn it is
        is_human_turn = (board.turn == chess.WHITE and human_color == 'white') or \
                        (board.turn == chess.BLACK and human_color == 'black')

        if is_human_turn:
            # Human move
            while True:
                try:
                    move_input = input(f"\n{human_color.capitalize()} to move: ").strip().lower()

                    if move_input == 'quit':
                        print("\nGame abandoned.")
                        return

                    if move_input == 'undo':
                        if len(move_history) >= 2:
                            # Undo both human and AI moves
                            board.pop()
                            board.pop()
                            move_history = move_history[:-2]
                            print("Undid last move.")
                        else:
                            print("Nothing to undo!")
                        break

                    # Parse move
                    move = chess.Move.from_uci(move_input)

                    if move in board.legal_moves:
                        board.push(move)
                        move_history.append(move)
                        print(f"✓ You played: {move.uci()}")
                        break
                    else:
                        print("❌ Illegal move! Try again.")
                        # Show legal moves if they want help
                        if input("Show legal moves? (y/n): ").strip().lower() == 'y':
                            legal = [m.uci() for m in board.legal_moves]
                            print(f"Legal moves: {', '.join(sorted(legal)[:20])}")

                except ValueError:
                    print("❌ Invalid format! Use UCI notation (e.g., 'e2e4')")
                except Exception as e:
                    print(f"❌ Error: {e}")

        else:
            # AI move
            print(f"\n🤖 AI is thinking ({simulations} simulations)...")
            try:
                move = get_model_move(board, model, simulations=simulations)
                board.push(move)
                move_history.append(move)
                print(f"✓ AI played: {move.uci()}")
            except Exception as e:
                print(f"❌ AI error: {e}")
                print("AI resigns due to error.")
                break

    # Game over
    print_board(board)
    print("\n" + "="*50)
    print("GAME OVER")
    print("="*50)
    print(f"Result: {board.result()}")

    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate - Draw!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    elif board.is_fifty_moves():
        print("Draw by fifty-move rule")
    elif board.is_repetition():
        print("Draw by threefold repetition")

    print("="*50)
    print(f"\nTotal moves: {len(move_history)}")
    print(f"Move history: {' '.join([m.uci() for m in move_history])}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Play chess against your trained model")
    parser.add_argument(
        '--checkpoint',
        default='../weights/best_model.pt',
        help='Path to model checkpoint (default: ../weights/best_model.pt)'
    )
    parser.add_argument(
        '--color',
        choices=['white', 'black'],
        default='white',
        help='Your color (default: white)'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=100,
        help='MCTS simulations (higher = stronger but slower, default: 100)'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nDownload the model first:")
        print("  cd src")
        print("  modal run download_model.py --mode download")
        return

    # Load model
    print("Loading model...")
    model, device = load_model(args.checkpoint)

    # Play game
    play_game(model, human_color=args.color, simulations=args.simulations)

if __name__ == "__main__":
    main()
