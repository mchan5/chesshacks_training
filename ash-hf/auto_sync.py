#!/usr/bin/env python3
"""
Auto-sync script: Downloads model from Modal and pushes to git every hour.

Usage:
    python auto_sync.py           # Run once
    python auto_sync.py --loop    # Run continuously every hour
"""
import subprocess
import time
import os
import argparse
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {description}...")
    try:
        # Set UTF-8 environment for subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace any problematic Unicode characters
            env=env
        )
        print(f"[OK] {description} completed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def sync_model():
    """Download model from Modal and push to git"""
    print("\n" + "="*60)
    print(f"SYNCING MODEL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Step 1: Download model from Modal
    success = run_command(
        "modal run ./src/download_model.py --mode download",
        "Downloading model from Modal"
    )

    if not success:
        print("[WARNING] Download failed, skipping git operations")
        return False

    # Step 2: Check if there are changes to commit
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(
        "git status --porcelain weights/",
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )

    if not result.stdout.strip():
        print("\n[OK] No changes to commit (model unchanged)")
        return True

    # Step 3: Git add weights
    success = run_command(
        "git add weights/",
        "Staging model weights"
    )

    if not success:
        return False

    # Step 4: Get latest checkpoint info for commit message
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            "modal run ./src/check_progress.py",
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        # Extract iteration number from output
        iteration = "unknown"
        for line in result.stdout.split('\n'):
            if "Latest iteration:" in line:
                # Format is: "[STATS] Latest iteration: 123"
                iteration = line.split(":")[-1].strip()
                break

        # If still unknown, checkpoint might not exist yet
        if iteration == "unknown":
            print("[INFO] Could not determine iteration (training may have just started)")
    except Exception as e:
        print(f"[INFO] Could not check iteration: {e}")
        iteration = "unknown"

    # Step 5: Commit
    commit_msg = f"Update model checkpoint (iteration {iteration})"
    success = run_command(
        f'git commit -m "{commit_msg}"',
        "Committing changes"
    )

    if not success:
        return False

    # Step 6: Push to remote
    success = run_command(
        "git push",
        "Pushing to remote"
    )

    print("\n" + "="*60)
    print(f"SYNC COMPLETED - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    return success

def main():
    parser = argparse.ArgumentParser(description="Auto-sync model from Modal to git")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously every 10 minutes"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=600,
        help="Sync interval in seconds (default: 600 = 10 minutes)"
    )
    args = parser.parse_args()

    if args.loop:
        print("Starting auto-sync loop...")
        print(f"Will sync every {args.interval} seconds ({args.interval/3600:.1f} hours)")
        print("Press Ctrl+C to stop")

        try:
            while True:
                sync_model()
                print(f"\n[WAIT] Waiting {args.interval/60:.0f} minutes until next sync...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nAuto-sync stopped by user")
    else:
        sync_model()

if __name__ == "__main__":
    main()
