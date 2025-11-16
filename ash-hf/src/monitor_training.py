#!/usr/bin/env python3
"""
Monitor continuous training and automatically download checkpoints.
Run this while training is happening on Modal.
"""
import time
import os
import subprocess
import json
from datetime import datetime

def get_checkpoints():
    """Get list of checkpoints from Modal"""
    result = subprocess.run(
        ["modal", "run", "train_modal.py", "--mode", "list"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__)
    )
    return result.stdout

def download_checkpoint(checkpoint_name):
    """Download a specific checkpoint"""
    print(f"Downloading {checkpoint_name}...")
    result = subprocess.run(
        ["modal", "run", "train_modal.py", "--mode", "download", "--checkpoint", checkpoint_name],
        cwd=os.path.dirname(__file__)
    )
    return result.returncode == 0

def monitor_training(
    check_interval=300,  # Check every 5 minutes
    download_best=True,
    download_all_checkpoints=False,
    auto_download_every_n=5  # Download every 5 iterations
):
    """
    Monitor training and download checkpoints.

    Args:
        check_interval: Seconds between checks (default 300 = 5 minutes)
        download_best: Always download best_model.pt when it updates
        download_all_checkpoints: Download all checkpoint files
        auto_download_every_n: Auto-download every N iterations
    """
    print("=" * 60)
    print("CONTINUOUS TRAINING MONITOR")
    print("=" * 60)
    print(f"Check interval: {check_interval} seconds ({check_interval/60:.1f} minutes)")
    print(f"Download best model: {download_best}")
    print(f"Auto-download every {auto_download_every_n} iterations")
    print("\nPress Ctrl+C to stop monitoring\n")

    seen_checkpoints = set()
    last_best_model_time = None

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Checking training progress...")

            # Get current checkpoints
            output = get_checkpoints()
            print(output)

            # Parse for new checkpoints
            lines = output.split('\n')
            new_checkpoints = []

            for line in lines:
                if 'iteration_' in line and '.pt' in line:
                    # Extract checkpoint name
                    parts = line.split()
                    if parts:
                        checkpoint_name = parts[0]
                        if checkpoint_name not in seen_checkpoints:
                            new_checkpoints.append(checkpoint_name)
                            seen_checkpoints.add(checkpoint_name)

            # Download new checkpoints
            if new_checkpoints:
                print(f"\nFound {len(new_checkpoints)} new checkpoint(s)!")

                for cp in new_checkpoints:
                    # Extract iteration number
                    try:
                        iter_num = int(cp.split('_')[1].split('.')[0])

                        # Download if it's a multiple of auto_download_every_n
                        if download_all_checkpoints or (iter_num % auto_download_every_n == 0):
                            if download_checkpoint(cp):
                                print(f"Downloaded {cp}")
                    except:
                        pass

            # Always try to download best_model.pt
            if download_best:
                if download_checkpoint("best_model.pt"):
                    print(f"Downloaded latest best_model.pt")

            # Wait before next check
            print(f"\nNext check in {check_interval/60:.1f} minutes...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Training continues on Modal - you can restart monitoring anytime.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor continuous training")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--no-best",
        action="store_true",
        help="Don't auto-download best_model.pt"
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download ALL checkpoints (uses more disk space)"
    )
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        help="Auto-download every N iterations (default: 5)"
    )

    args = parser.parse_args()

    monitor_training(
        check_interval=args.interval,
        download_best=not args.no_best,
        download_all_checkpoints=args.download_all,
        auto_download_every_n=args.every
    )
