#!/usr/bin/env python3
"""
Clear all checkpoints from Modal volume to start fresh.
"""
import modal
import os

app = modal.App("chess-clear")
volume = modal.Volume.from_name("chess-weights", create_if_missing=False)

@app.function(volumes={"/root/data": volume})
def clear_all_checkpoints():
    """Delete all checkpoints to start fresh"""
    import os
    import shutil

    weights_dir = "/root/data/weights"

    if os.path.exists(weights_dir):
        # List files before deletion
        files = os.listdir(weights_dir)
        checkpoint_files = [f for f in files if f.endswith('.pt')]

        if checkpoint_files:
            print(f"Found {len(checkpoint_files)} checkpoint(s) to delete:")
            for f in checkpoint_files:
                print(f"  - {f}")

            # Delete all .pt files
            for f in checkpoint_files:
                path = os.path.join(weights_dir, f)
                os.remove(path)
                print(f"✓ Deleted {f}")

            print("\nAll checkpoints cleared! Ready for fresh training.")
        else:
            print("No checkpoints found - already clean!")
    else:
        print("No weights directory found - volume is clean!")

@app.local_entrypoint()
def main():
    """Clear all checkpoints from Modal volume"""
    print("=" * 60)
    print("CLEARING ALL CHECKPOINTS")
    print("=" * 60)
    print("\nThis will delete all saved models from the Modal volume.")
    print("Training will start completely fresh.\n")

    response = input("Are you sure? (yes/no): ")

    if response.lower() == "yes":
        print("\nClearing checkpoints...")
        clear_all_checkpoints.remote()
        print("\n✓ Done! You can now start training from scratch.")
    else:
        print("Cancelled.")
