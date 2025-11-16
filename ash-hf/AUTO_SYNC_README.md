# Auto-Sync Model from Modal to Git

Automatically downloads your trained model from Modal and pushes it to git every hour.

## Usage

### Windows (Easiest)
Just double-click or run:
```bash
auto_sync.bat
```

### Python (All platforms)

**Run once:**
```bash
python auto_sync.py
```

**Run continuously (every hour):**
```bash
python auto_sync.py --loop
```

**Custom interval (e.g., every 30 minutes):**
```bash
python auto_sync.py --loop --interval 1800
```

## What it does

Every hour (or your custom interval):
1. Downloads latest model from Modal (`modal run src/download_model.py --mode download`)
2. Checks if model changed
3. If changed:
   - Stages weights: `git add weights/`
   - Commits: `git commit -m "Update model checkpoint (iteration X)"`
   - Pushes: `git push`
4. If unchanged: skips commit/push

## Requirements

- Modal CLI authenticated (`modal token set`)
- Git configured with remote repository
- Training must be running on Modal

## Stopping

Press `Ctrl+C` to stop the auto-sync loop.

## Troubleshooting

**"Download failed"**: Check that training is running on Modal
```bash
modal app list
```

**"Git push failed"**: Make sure you have push access to the remote
```bash
git remote -v
```

**Want to change interval**: Use `--interval SECONDS`
```bash
python auto_sync.py --loop --interval 7200  # Every 2 hours
```
