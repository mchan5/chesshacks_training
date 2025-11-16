# Parallel self-play using Python multiprocessing or Modal parallelism

import modal
from train_continuous_modal import app, volume, image

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/root/data": volume},
)
def parallel_selfplay_game(model_state, num_simulations):
    """Single self-play game - can be run in parallel"""
    import sys
    sys.path.insert(0, "/root/src")

    from train import ChessNet, device
    from train_continuous import play_self_play_game
    import torch

    # Load model
    model = ChessNet(input_channels=18, num_filters=64, num_residual_blocks=3).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Play game
    return play_self_play_game(model, num_simulations=num_simulations)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/root/data": volume},
)
def parallel_continuous_train(num_iterations=1000, games_per_iteration=25, mcts_simulations=50):
    """Continuous training with parallel self-play"""
    import sys
    sys.path.insert(0, "/root/src")

    from train import ChessNet, device
    from train_continuous import train_on_selfplay
    import torch
    import torch.optim as optim

    model = ChessNet(input_channels=18, num_filters=64, num_residual_blocks=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for iteration in range(num_iterations):
        print(f"\nITERATION {iteration + 1}/{num_iterations}")

        # Get model state for parallel workers
        model_state = model.state_dict()

        # Launch games in parallel across multiple containers!
        print(f"Generating {games_per_iteration} self-play games in parallel...")
        game_records = list(
            parallel_selfplay_game.map(
                [model_state] * games_per_iteration,
                [mcts_simulations] * games_per_iteration
            )
        )

        # Training (same as before)
        train_loss = train_on_selfplay(model, optimizer, game_records, epochs=2, batch_size=64)
        print(f"Training loss: {train_loss:.4f}")

        # Save checkpoint
        # ... (same as before)

    volume.commit()
