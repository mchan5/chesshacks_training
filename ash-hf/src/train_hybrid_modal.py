"""
Modal deployment for hybrid training (supervised + self-play).
"""
import modal

app = modal.App("chess-hybrid-training")

# Persistent volume
volume = modal.Volume.from_name("chess-weights", create_if_missing=True)

# Image with all dependencies
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
        "python-chess",
    )
    .add_local_dir(script_dir, "/root/src")  # Copy entire src directory
    .add_local_dir(data_dir, "/root/training_data")  # Copy data directory to different path
)


@app.function(
    image=image,
    gpu="A100",  # Fast GPU for training
    timeout=86400,  # 24 hour timeout
    volumes={"/root/weights": volume},  # Mount volume at different path
    cpu=4,
)
def hybrid_train(
    supervised_epochs: int = 5,
    selfplay_iterations: int = 500,
    games_per_iteration: int = 20,
    mcts_simulations: int = 30,
):
    """
    Run hybrid training: supervised pre-training + self-play.

    Args:
        supervised_epochs: Epochs for supervised learning on games/puzzles
        selfplay_iterations: Number of self-play iterations
        games_per_iteration: Games per self-play iteration
        mcts_simulations: MCTS simulations per move
    """
    import sys
    sys.path.insert(0, "/root/src")

    from train_hybrid import hybrid_training

    print("=" * 60)
    print("Starting HYBRID training on Modal")
    print("=" * 60)
    print(f"Stage 1: Supervised pre-training ({supervised_epochs} epochs)")
    print(f"Stage 2: Self-play ({selfplay_iterations} iterations)")
    print("=" * 60)

    hybrid_training(
        supervised_epochs=supervised_epochs,
        selfplay_iterations=selfplay_iterations,
        games_per_iteration=games_per_iteration,
        mcts_simulations=mcts_simulations
    )

    # Commit volume to persist all changes
    volume.commit()

    print("\n" + "=" * 60)
    print("Hybrid training completed!")
    print("=" * 60)


@app.local_entrypoint()
def main(
    supervised_epochs: int = 5,
    iterations: int = 500,
    games: int = 20,
    simulations: int = 30,
):
    """
    Launch hybrid training on Modal.

    Usage:
        # Download data first (run locally):
        python preprocess.py --games 10000 --puzzles 5000

        # Then upload and run hybrid training:
        modal run train_hybrid_modal.py
        modal run train_hybrid_modal.py --supervised-epochs 10 --iterations 500
    """
    print(f"Launching HYBRID training on Modal...")
    print(f"  Supervised pre-training: {supervised_epochs} epochs")
    print(f"  Self-play iterations: {iterations}")
    print(f"  Games per iteration: {games}")
    print(f"  MCTS simulations: {simulations}")
    print("\nThis will run for a long time. Press Ctrl+C to stop monitoring.")
    print("The job will continue running on Modal even if you disconnect.\n")

    hybrid_train.remote(
        supervised_epochs=supervised_epochs,
        selfplay_iterations=iterations,
        games_per_iteration=games,
        mcts_simulations=simulations,
    )
