"""
Modal deployment script for chess AI training with MCTS.
Run with: modal run train_modal.py
"""
import modal

# Create Modal app
app = modal.App("chess-mcts-training")

# Create a volume for persistent storage of weights and data
volume = modal.Volume.from_name("chess-weights", create_if_missing=True)

# Define the image with all dependencies
# Copy local files into the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
        "python-chess",
        "datasets",
        "transformers",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("data", "/root/data")
)


@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G GPU with 24GB VRAM
    timeout=3600 * 3,  # 3 hour timeout
    volumes={"/root/weights": volume},
)
def train():
    """Train the chess model on Modal's cloud GPUs"""
    import sys
    sys.path.insert(0, "/root/src")

    # Import training script (already bundled in image)
    from train import main

    print("=" * 60)
    print("Starting Chess AI Training on Modal")
    print("=" * 60)

    # Run training
    main()

    # Commit volume to persist weights
    volume.commit()

    print("\n" + "=" * 60)
    print("Training complete! Weights saved to Modal volume.")
    print("=" * 60)


@app.function(
    image=image,
    gpu="T4",  # Smaller GPU for inference
    volumes={"/root/weights": volume},
)
def predict_move(fen: str, num_simulations: int = 100):
    """
    Use MCTS to predict best move for a given position.

    Args:
        fen: FEN string of the chess position
        num_simulations: Number of MCTS simulations to run

    Returns:
        Best move in UCI format
    """
    import sys
    sys.path.insert(0, "/root/src")

    import torch
    import chess
    from train import ChessNet, MCTS, encode_board, index_to_move

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ChessNet(
        input_channels=18,
        num_filters=128,
        num_residual_blocks=5
    ).to(device)

    try:
        checkpoint = torch.load("/root/weights/best_model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")
    except FileNotFoundError:
        print("No trained model found. Please run training first.")
        return None

    model.eval()

    # Parse FEN and run MCTS
    board = chess.Board(fen)
    mcts = MCTS(model, num_simulations=num_simulations)

    print(f"\nPosition: {fen}")
    print(f"Running MCTS with {num_simulations} simulations...")

    visit_counts = mcts.search(board)

    # Get top move
    import numpy as np
    top_idx = np.argmax(visit_counts)
    best_move = index_to_move(top_idx, board)

    if best_move:
        print(f"Best move: {best_move.uci()}")

        # Show top 5 moves
        top_indices = np.argsort(visit_counts)[-5:][::-1]
        print("\nTop 5 moves:")
        for i, idx in enumerate(top_indices, 1):
            move = index_to_move(idx, board)
            if move and visit_counts[idx] > 0:
                print(f"  {i}. {move.uci()}: {int(visit_counts[idx])} visits")

        return best_move.uci()
    else:
        print("No valid move found.")
        return None


@app.function(
    image=image,
    volumes={"/root/weights": volume},
)
def download_weights():
    """Download trained weights from Modal volume to local machine"""
    import os

    weights_dir = "/root/weights"

    if not os.path.exists(weights_dir):
        print("No weights directory found on Modal volume.")
        return {}

    files = os.listdir(weights_dir)
    print(f"Found {len(files)} files in weights directory:")
    for f in files:
        print(f"  - {f}")

    # Return file contents for download
    weights = {}
    for f in files:
        if f.endswith('.pt'):
            path = os.path.join(weights_dir, f)
            with open(path, 'rb') as file:
                weights[f] = file.read()

    return weights


@app.local_entrypoint()
def main(mode: str = "train"):
    """
    Main entry point for Modal deployment.

    Usage:
        modal run train_modal.py                    # Train the model
        modal run train_modal.py --mode predict     # Test prediction
        modal run train_modal.py --mode download    # Download weights
    """
    if mode == "train":
        print("Launching training job on Modal...")
        train.remote()

    elif mode == "predict":
        # Test prediction on starting position
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = predict_move.remote(starting_fen, num_simulations=100)
        print(f"\nPredicted move: {result}")

    elif mode == "download":
        print("Downloading weights from Modal...")
        weights = download_weights.remote()

        if weights:
            import os
            os.makedirs("ash-hf/weights", exist_ok=True)

            for filename, content in weights.items():
                local_path = os.path.join("ash-hf/weights", filename)
                with open(local_path, 'wb') as f:
                    f.write(content)
                print(f"Downloaded: {local_path}")

            print(f"\nSuccessfully downloaded {len(weights)} weight files!")
        else:
            print("No weights found to download.")

    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: train, predict, download")


# Serve as web endpoint (optional)
@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/weights": volume},
)
@modal.web_endpoint(method="POST")
def predict_api(request: dict):
    """
    Web API endpoint for move prediction.

    POST request body:
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "num_simulations": 100
    }
    """
    fen = request.get("fen")
    num_simulations = request.get("num_simulations", 100)

    if not fen:
        return {"error": "Missing 'fen' parameter"}

    move = predict_move.remote(fen, num_simulations)

    return {
        "fen": fen,
        "best_move": move,
        "num_simulations": num_simulations
    }
