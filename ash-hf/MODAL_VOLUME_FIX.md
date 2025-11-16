# Modal Volume Persistence Fix

## The Problem

**Symptom**: Model was training successfully (loss decreasing), but when downloading the model from Modal, it appeared unchanged across iterations.

**Root Cause**: Modal volumes use lazy writes and only persist changes when `volume.commit()` is explicitly called. Previously, we only called `volume.commit()` once at the very end of training (after all 500 iterations), which meant:

1. Training runs for hours/days
2. Model weights are saved to the volume every iteration
3. BUT changes only exist in Modal's cache layer
4. If you download the model mid-training, you get the old version
5. Only when the entire training completes does `volume.commit()` save everything

## The Attempted Fix (Didn't Work)

Previously tried adding `os.sync()` after saving the model:
```python
torch.save(checkpoint, best_model_path)
os.sync()  # This doesn't help with Modal volumes!
```

**Why it failed**: `os.sync()` only flushes the OS filesystem cache, but Modal volumes have their own caching layer that ignores this.

## The Proper Fix (Now Implemented)

**Solution**: Pass the Modal volume object into the training function and call `volume.commit()` after every checkpoint save.

### Changes Made

1. **train_continuous.py** - Added `modal_volume` parameter and commit logic:
   ```python
   def continuous_training(..., modal_volume=None):
       # ... training loop ...

       # Save checkpoint
       torch.save(checkpoint, best_model_path)

       # Commit Modal volume to persist changes
       if modal_volume is not None:
           modal_volume.commit()
           print("[OK] Modal volume committed (changes persisted)")
   ```

2. **train_hybrid.py** - Pass volume through to continuous_training:
   ```python
   def hybrid_training(..., modal_volume=None):
       # ... supervised pre-training ...

       continuous_training(
           ...,
           modal_volume=modal_volume  # Pass it through
       )
   ```

3. **train_hybrid_modal.py** - Pass volume object from Modal function:
   ```python
   @app.function(volumes={"/root/weights": volume}, ...)
   def hybrid_train(...):
       hybrid_training(
           ...,
           modal_volume=volume  # Pass the volume object
       )

       volume.commit()  # Final commit at end
   ```

4. **train_continuous_modal.py** - Same changes for pure self-play training

## How It Works Now

With `checkpoint_every=10`:

- **Iteration 1-9**: Model trains, saves to volume cache
- **Iteration 10**: Model saves, `volume.commit()` persists ALL changes
- **Iteration 11-19**: More training, saves to cache
- **Iteration 20**: `volume.commit()` persists changes again
- And so on...

This means:
- ✅ Model is available for download every 10 iterations
- ✅ If training crashes, you only lose progress since last commit (max 10 iterations)
- ✅ Auto-sync script can download the latest model every 20 minutes
- ✅ No wasted compute - all progress is saved

## Verification

To verify the fix is working:

1. Check that training prints `[OK] Modal volume committed (changes persisted)` every 10 iterations
2. Run auto-sync and verify the model file changes
3. Check iteration number in downloaded checkpoint matches training output

## Performance Impact

Minimal - `volume.commit()` is fast (< 1 second) and only happens every 10 iterations (roughly every 30-60 minutes of training).
