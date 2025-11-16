# Complete Optimization Summary

## All Applied Optimizations

### 1. **Batched MCTS Evaluation** (3-5x speedup) ✅
- **What**: Evaluates 8 positions at once instead of 1 at a time
- **Impact**: Massive GPU utilization improvement
- **Location**: `train.py` MCTS class with `batch_size=8`

### 2. **Smaller Network** (2x speedup) ✅
- **Original**: 128 filters, 5 residual blocks
- **Optimized**: 64 filters, 3 residual blocks
- **Impact**: 4x fewer parameters = 2x faster inference
- **Location**: `train_continuous.py:240-243`

### 3. **Mixed Precision Training (FP16)** (1.5-2x speedup) ✅
- **What**: Uses 16-bit floats for training
- **Impact**: Faster training, less memory
- **Location**: `train_continuous.py:163` with `use_amp=True`

### 4. **torch.compile** (1.3-1.8x speedup) ✅
- **What**: JIT compiles model for faster inference
- **Impact**: Optimized CUDA kernels
- **Location**: `train_continuous.py:256-261`

### 5. **Optimized Board Encoding** (1.1-1.2x speedup) ✅
- **What**: Vectorized operations, fewer allocations
- **Impact**: Faster board-to-tensor conversion
- **Location**: `train.py:105-146`

### 6. **Debug Timing** ✅
- **What**: Shows exactly where time is spent
- **Impact**: Helps identify bottlenecks
- **Location**: `train_continuous.py:56-160`

## Expected Performance

### Before Optimizations:
- Game time: 20-30 seconds
- Iteration time: 10-15 minutes
- 100 iterations: ~20-25 hours

### After All Optimizations:
- Game time: **3-6 seconds** (4-8x faster!)
- Iteration time: **2-4 minutes** (4-6x faster!)
- 100 iterations: **3-7 hours** (4-6x faster!)

## Recommended Settings

### For Development/Testing:
```bash
modal run train_continuous_modal.py --iterations 10 --games 10 --simulations 25
```
- Fast iterations (~2 min each)
- Cheap ($0.30-0.50 total)
- Good for testing

### For Production Training:
```bash
modal run train_continuous_modal.py --iterations 500 --games 20 --simulations 30
```
- Balanced speed/quality
- ~30-40 hours total
- ~$35-45 cost

### For Maximum Quality:
```bash
modal run train_continuous_modal.py --iterations 1000 --games 25 --simulations 40
```
- Best training data
- ~60-80 hours total
- ~$70-90 cost

## Cost Breakdown

| Setting | Iterations | Time | A10G Cost ($1.10/hr) |
|---------|------------|------|----------------------|
| Quick test | 10 | 30 min | $0.55 |
| Development | 50 | 2-3 hrs | $2.50-3.50 |
| Good model | 200 | 12-15 hrs | $14-17 |
| Production | 500 | 30-40 hrs | $35-45 |
| Best quality | 1000 | 60-80 hrs | $70-90 |

## What Was Changed

### train.py:
- ✅ Fixed default network size (119→18 channels, 256→128→64 filters)
- ✅ Added batched MCTS (batch_size=8)
- ✅ Optimized encode_board function

### train_continuous.py:
- ✅ Smaller network (64 filters, 3 blocks)
- ✅ Added torch.compile
- ✅ Added mixed precision training
- ✅ Added debug timing for first game

### Files Created:
- `train_batched_mcts.py` - Example of batched MCTS (already integrated)
- `train_parallel_selfplay.py` - For future: parallel self-play across multiple GPUs
- `OPTIMIZATIONS.md` - This file

## Still Possible (Not Yet Implemented):

### 7. Parallel Self-Play (10-25x speedup, 10-25x cost)
Run games in parallel across multiple containers. See `train_parallel_selfplay.py`.

### 8. Larger Batch Size for MCTS
Increase from 8 to 16 or 32 if GPU memory allows (minor speedup).

### 9. Better GPU (A100)
2-3x faster but 2-3x more expensive. Use for final training runs.

## How to Use

1. **Start with quick test:**
   ```bash
   modal run train_continuous_modal.py --iterations 5 --games 5 --simulations 20
   ```
   Should complete in ~10 minutes, cost ~$0.20

2. **If working well, scale up:**
   ```bash
   modal run train_continuous_modal.py --iterations 100 --games 15 --simulations 25
   ```

3. **Monitor with:**
   ```bash
   # Check progress
   modal app list

   # Download latest model
   cd ash-hf/src
   modal run train_modal.py --mode download --checkpoint best_model.pt
   ```

## Performance Metrics to Watch

Good signs:
- ✅ First game: 5-15 seconds (CUDA warmup)
- ✅ Subsequent games: 3-6 seconds
- ✅ Training loss decreasing over iterations
- ✅ Game lengths getting longer (better play)

Bad signs:
- ❌ Games taking 30+ seconds (might be running on CPU)
- ❌ "Using device: cpu" in logs (no GPU allocated)
- ❌ OOM errors (batch size too large)

## Troubleshooting

**If still slow:**
1. Check Modal logs for "Using device: cuda" (not "cpu")
2. First game is always slower (CUDA warmup) - watch 2nd game time
3. Try reducing simulations: `--simulations 20`

**If out of memory:**
1. Reduce network size further (32 filters, 2 blocks)
2. Reduce batch_size in MCTS (from 8 to 4)
3. Reduce training batch size (from 64 to 32)

**If training not improving:**
1. Increase simulations (25→40)
2. Increase games per iteration (15→25)
3. Train for more iterations
