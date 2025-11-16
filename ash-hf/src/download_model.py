#!/usr/bin/env python3
"""
Download trained models from Modal volume.
"""
import modal
import os

app = modal.App("chess-download")
volume = modal.Volume.from_name("chess-weights", create_if_missing=False)

@app.function(volumes={"/root/weights": volume})
def list_checkpoints():
    """List all checkpoints in the volume"""
    import os
    weights_dir = "/root/weights"

    if not os.path.exists(weights_dir):
        print("No weights directory found yet - training hasn't started")
        return []

    files = os.listdir(weights_dir)
    checkpoint_files = [f for f in files if f.endswith('.pt')]

    if not checkpoint_files:
        print("No checkpoints found yet")
        return []

    print(f"\nFound {len(checkpoint_files)} checkpoint(s):")
    for f in sorted(checkpoint_files):
        path = os.path.join(weights_dir, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")

    return checkpoint_files

@app.function(volumes={"/root/weights": volume})
def download_checkpoint(checkpoint_name: str, local_dir: str = "."):
    """Download a specific checkpoint"""
    import os
    remote_path = f"/root/weights/{checkpoint_name}"

    if not os.path.exists(remote_path):
        print(f"Checkpoint {checkpoint_name} not found!")
        return None

    # Read the file
    with open(remote_path, 'rb') as f:
        data = f.read()

    return data

@app.local_entrypoint()
def main(mode: str = "list", checkpoint: str = "best_model.pt"):
    """
    Download models from Modal volume.

    Usage:
        modal run download_model.py                                    # List checkpoints
        modal run download_model.py --mode download                     # Download best_model.pt
        modal run download_model.py --mode download --checkpoint iteration_10.pt
    """

    if mode == "list":
        print("Listing checkpoints from Modal volume...")
        list_checkpoints.remote()

    elif mode == "download":
        print(f"Downloading {checkpoint}...")
        data = download_checkpoint.remote(checkpoint)

        if data is None:
            print("Download failed - checkpoint not found")
            return

        # Create local weights directory
        local_weights_dir = os.path.join(os.path.dirname(__file__), "../weights")
        os.makedirs(local_weights_dir, exist_ok=True)

        # Save locally
        local_path = os.path.join(local_weights_dir, checkpoint)
        with open(local_path, 'wb') as f:
            f.write(data)

        size_mb = len(data) / (1024 * 1024)
        print(f"[OK] Downloaded {checkpoint} ({size_mb:.1f} MB)")
        print(f"[OK] Saved to: {local_path}")

    else:
        print(f"Unknown mode: {mode}")
        print("Use --mode list or --mode download")
