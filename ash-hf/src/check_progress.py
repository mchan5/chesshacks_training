#!/usr/bin/env python3
"""
Check training progress and data stored on Modal volume.
"""
import modal

app = modal.App("chess-progress-check")
volume = modal.Volume.from_name("chess-weights", create_if_missing=False)

@app.function(volumes={"/root/data": volume})
def check_storage():
    """Check what's stored in the Modal volume"""
    import os

    print("\n" + "="*60)
    print("MODAL VOLUME CONTENTS")
    print("="*60)

    base_dir = "/root/data"

    if not os.path.exists(base_dir):
        print("❌ No data directory found - training hasn't started yet")
        return

    total_size = 0
    file_count = 0

    # Check weights directory
    weights_dir = os.path.join(base_dir, "weights")
    if os.path.exists(weights_dir):
        print("\n📦 CHECKPOINTS (weights/):")
        weights = os.listdir(weights_dir)
        checkpoints = [f for f in weights if f.endswith('.pt')]

        if checkpoints:
            checkpoints.sort()
            for cp in checkpoints:
                path = os.path.join(weights_dir, cp)
                size = os.path.getsize(path)
                size_mb = size / (1024 * 1024)
                total_size += size
                file_count += 1

                # Try to read iteration number
                try:
                    import torch
                    checkpoint = torch.load(path, map_location='cpu')
                    iteration = checkpoint.get('iteration', '?')
                    print(f"  ✓ {cp:30s} {size_mb:6.1f} MB (iteration {iteration})")
                except:
                    print(f"  ✓ {cp:30s} {size_mb:6.1f} MB")
        else:
            print("  (No checkpoints found)")
    else:
        print("\n📦 CHECKPOINTS: None yet")

    # Check selfplay directory
    selfplay_dir = os.path.join(base_dir, "selfplay")
    if os.path.exists(selfplay_dir):
        print("\n🎮 SELF-PLAY GAMES (selfplay/):")
        games = os.listdir(selfplay_dir)
        if games:
            game_count = len(games)
            selfplay_size = sum(
                os.path.getsize(os.path.join(selfplay_dir, f))
                for f in games
            )
            selfplay_mb = selfplay_size / (1024 * 1024)
            total_size += selfplay_size
            file_count += game_count
            print(f"  {game_count} game files, {selfplay_mb:.1f} MB total")
        else:
            print("  (No game files)")
    else:
        print("\n🎮 SELF-PLAY GAMES: None")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_mb = total_size / (1024 * 1024)
    total_gb = total_size / (1024 * 1024 * 1024)
    print(f"Total files: {file_count}")
    print(f"Total size: {total_mb:.1f} MB ({total_gb:.2f} GB)")

    # Estimate training progress
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(weights_dir, x)))
        try:
            import torch
            checkpoint = torch.load(os.path.join(weights_dir, latest_checkpoint), map_location='cpu')
            iteration = checkpoint.get('iteration', 0)
            print(f"\n📊 Latest iteration: {iteration}")

            # Estimate based on common targets
            if iteration < 100:
                print(f"Progress: Early training ({iteration}/100 for basic bot)")
            elif iteration < 500:
                print(f"Progress: {iteration}/500 for competitive bot ({iteration/500*100:.1f}%)")
            elif iteration < 1000:
                print(f"Progress: {iteration}/1000 for strong bot ({iteration/1000*100:.1f}%)")
            else:
                print(f"Progress: {iteration}+ iterations (advanced training)")
        except:
            pass

    print("="*60 + "\n")

@app.local_entrypoint()
def main():
    """Check training progress"""
    print("Checking Modal volume for training data...")
    check_storage.remote()
