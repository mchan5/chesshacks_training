# Continuous Training Guide

Make your chess AI train continuously and improve forever using self-play.

## What is Continuous Training?

Instead of training once on static data, the model:
1. **Plays games against itself** using MCTS
2. **Learns from those games**
3. **Repeats forever**, getting stronger each iteration

This is how AlphaZero and other top chess AIs work!

## Option 1: Run Locally (Free, but needs your computer on)

```bash
cd ash-hf/src
python train_continuous.py
```

This will:
- Run 1000 iterations (can be changed)
- Generate 25 self-play games per iteration
- Train on the new games
- Save checkpoints every 5 iterations
- Keep improving indefinitely

**Pros:** Free
**Cons:** Needs your computer running, slower on CPU

## Option 2: Run on Modal (Recommended for serious training)

```bash
cd ash-hf/src
modal run train_continuous_modal.py
```

This will:
- Deploy to Modal's cloud GPUs (A10G)
- Run for 24 hours (configurable)
- Continue even if you close your laptop
- Save progress to Modal volume

**Cost:** ~$1-2/hour, but produces much stronger models

### Custom Configuration

```bash
# Train for 10,000 iterations with 50 games each
modal run train_continuous_modal.py --iterations 10000 --games 50 --simulations 100
```

## How It Works

### Iteration Loop

Each iteration:
1. **Self-Play** (5-10 min): Model plays 25 games against itself
2. **Training** (2-3 min): Trains on ~500-1000 positions from those games
3. **Save** (instant): Saves checkpoint
4. **Repeat**: Goes to next iteration

### Total Time

- **Per iteration:** ~10-15 minutes on A10G GPU
- **100 iterations:** ~20-25 hours
- **1000 iterations:** ~10 days of continuous training

### Self-Play Statistics

You'll see output like:
```
ITERATION 42/1000
============================================================

Generating 25 self-play games...
Self-play: 100%|████████| 25/25 [08:32<00:00, 20.5s/game]

Self-play stats:
  Total positions: 1,247
  Avg moves/game: 49.9
  Results: W:12 B:9 D:4

Training on 1,247 positions...
Epoch 1/2: 100%|████████| 20/20 [01:15<00:00]
Training loss: 3.2451

✓ Saved checkpoint: iteration_42.pt
Iteration time: 10.3 minutes
Estimated time remaining: 164.7 hours
```

## Configuration Options

Edit [train_continuous.py:333-340](ash-hf/src/train_continuous.py#L333-L340):

```python
continuous_training(
    num_iterations=1000,         # Total iterations (higher = longer)
    games_per_iteration=25,      # Games per iteration (higher = more data, slower)
    mcts_simulations=50,         # MCTS sims per move (higher = stronger play, slower)
    train_epochs_per_iteration=2,  # Training epochs (2-5 typical)
    batch_size=64,               # Batch size (64-128)
    learning_rate=0.001,         # Learning rate (0.0001-0.01)
    checkpoint_every=5           # Save frequency
)
```

## Resuming Training

If training stops, just run again:
```bash
modal run train_continuous_modal.py
```

It will automatically:
- Load the latest `best_model.pt`
- Resume from where it left off
- Continue training

## Monitoring Progress

### Check Iteration Number

The model saves iteration info in checkpoints:
```python
checkpoint = torch.load("best_model.pt")
print(f"Current iteration: {checkpoint['iteration']}")
```

### View Self-Play Games

Game metadata is saved to `selfplay_games/games_iter_*.json`:
```bash
# Download from Modal
modal run train_modal.py --mode download
```

### Compare Model Versions

Pit different iterations against each other:
```python
# Load two models and have them play
model_v1 = load_checkpoint("iteration_10.pt")
model_v2 = load_checkpoint("iteration_50.pt")
# Play 100 games, see which wins more
```

## When to Stop

You can stop anytime! The model is saved every iteration. Stop when:
- **Validation loss plateaus** (stops improving)
- **Out of budget** (Modal costs)
- **Good enough for your needs**

Typical stopping points:
- **50-100 iterations:** Basic competent play
- **500-1000 iterations:** Strong amateur level
- **5000+ iterations:** Expert level (expensive!)

## Cost Estimation (Modal)

| Iterations | Time | Cost (A10G @ $1.10/hr) |
|------------|------|------------------------|
| 10 | 2-3 hours | $2-3 |
| 100 | 20-25 hours | $20-25 |
| 1000 | 10 days | $250-300 |

**Tip:** Start with 50-100 iterations to test, then scale up.

## Advanced: Infinite Training

For truly continuous training:

```python
num_iterations=999999  # Effectively infinite
```

Set a longer timeout in Modal:
```python
timeout=86400 * 7  # 7 days
```

The job will run until:
- You stop it manually
- Modal timeout is reached
- Budget limit is hit

## Comparison: Regular vs Continuous Training

| Aspect | Regular Training | Continuous Training |
|--------|------------------|---------------------|
| Data | Fixed (5k games) | Growing (infinite) |
| Time | 20 min | Hours to days |
| Improvement | Learns patterns | Gets stronger continuously |
| Cost | $0.30 | $1-300+ |
| Best for | Prototyping | Production quality |

## Next Steps

1. **Start small:** Run 10 iterations locally to verify it works
2. **Test on Modal:** Run 50 iterations on Modal
3. **Scale up:** If satisfied, run 500-1000 iterations
4. **Evaluate:** Test model against Stockfish or online bots
5. **Iterate:** Adjust hyperparameters and continue

Happy training! 🚀
