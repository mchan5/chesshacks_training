# Chess AI Training on Modal
Deploy and train your chess AI with MCTS on cloud GPUs using Modal.

## Setup

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

This will open your browser to authenticate. Create a free account if you don't have one.

### 3. Verify Your Data

Make sure you have preprocessed data files:
- `ash-hf/data/games.json`
- `ash-hf/data/puzzles.json`

If not, run the preprocessing first:
```bash
cd ash-hf/src
python preprocess.py
```

## Usage

### Training on Modal GPUs

Launch training on a cloud GPU (A10G with 24GB VRAM):

```bash
modal run train_modal.py
```

This will:
- Upload your code and data to Modal
- Spin up an A10G GPU instance
- Train for 10 epochs (~15-20 minutes)
- Save weights to a persistent Modal volume
- Shut down automatically when done

**Cost:** ~$0.30-0.50 for a full training run (A10G is ~$1.10/hour)

### Monitor Training

You'll see live output in your terminal:
```
Epoch 1/10
Training: 100%|████████| 150/150 [00:45<00:00]
Train Loss: 8.3421 (Policy: 8.1234, Value: 0.2187)
Val Loss: 7.9876, Val Accuracy: 0.0523
✓ Saved best model (val_loss: 7.9876)
```

### Test Inference with MCTS

After training, test move prediction:

```bash
modal run train_modal.py --mode predict
```

This will run MCTS on the starting position and show the top moves.

### Download Trained Weights

Download weights from Modal to your local machine:

```bash
modal run train_modal.py --mode download
```

Weights will be saved to `ash-hf/weights/`:
- `best_model.pt` - Best model by validation loss
- `final_model.pt` - Model after all epochs
- `checkpoint_epoch_*.pt` - Intermediate checkpoints

## Advanced Usage

### Custom Inference

Use the Python API to get move predictions:

```python
import modal

app = modal.App.lookup("chess-mcts-training", create_if_missing=False)
predict_move = modal.Function.lookup("chess-mcts-training", "predict_move")

# Predict move for a position
fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
best_move = predict_move.remote(fen, num_simulations=200)
print(f"Best move: {best_move}")
```

### Web API Endpoint

Deploy as a REST API (run once):

```bash
modal deploy train_modal.py
```

Then make POST requests:

```bash
curl -X POST https://your-username--chess-mcts-training-predict-api.modal.run \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "num_simulations": 100}'
```

Response:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "best_move": "e2e4",
  "num_simulations": 100
}
```

### Change GPU Type

Edit `train_modal.py` and modify the `@app.function` decorator:

```python
@app.function(
    gpu="T4",      # Cheaper option (~$0.60/hr)
    # gpu="A10G",  # Current (24GB, ~$1.10/hr)
    # gpu="A100",  # Powerful (40GB, ~$4/hr)
)
```

## Training Configuration

Modify hyperparameters in `ash-hf/src/train.py`:

```python
BATCH_SIZE = 64            # Increase if you have more VRAM
LEARNING_RATE = 0.001      # Adjust learning rate
NUM_EPOCHS = 10            # More epochs = better (but longer)
NUM_RESIDUAL_BLOCKS = 5    # Model depth (5-20)
NUM_FILTERS = 128          # Model width (128-256)
```

For MCTS simulations, modify in `train_modal.py`:

```python
num_simulations=100  # More simulations = stronger play (but slower)
```

## Cost Estimation

| GPU Type | Cost/Hour | 10 Epoch Training | 100 MCTS Sims |
|----------|-----------|-------------------|---------------|
| T4       | $0.60     | ~30 min (~$0.30)  | ~5 sec        |
| A10G     | $1.10     | ~20 min (~$0.35)  | ~3 sec        |
| A100     | $4.00     | ~10 min (~$0.65)  | ~1 sec        |

**Free Tier:** Modal gives $30/month free credits for new users.

## Troubleshooting

### Out of Memory Error

Reduce batch size in `train.py`:
```python
BATCH_SIZE = 32  # or 16
```

### Data Not Found

Make sure you've run preprocessing:
```bash
cd ash-hf/src
python preprocess.py
```

### Modal Authentication Issues

Re-authenticate:
```bash
modal setup --force
```

### Check Modal Volume

List files in the persistent volume:
```bash
modal volume ls chess-weights
```

## File Structure

```
chesshacks_training/
├── train_modal.py              # Modal deployment script
├── ash-hf/
│   ├── data/
│   │   ├── games.json          # Training data (uploaded to Modal)
│   │   └── puzzles.json
│   ├── src/
│   │   ├── train.py            # Training logic (uploaded to Modal)
│   │   └── preprocess.py
│   └── weights/                # Downloaded weights (local)
│       ├── best_model.pt
│       └── final_model.pt
└── MODAL_DEPLOYMENT.md         # This file
```

## Next Steps

1. **Increase Dataset Size**: Process more games/puzzles for better performance
2. **Self-Play Training**: Implement AlphaZero-style self-play loop
3. **Hyperparameter Tuning**: Experiment with network architecture
4. **Arena Testing**: Pit different model versions against each other

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Discord](https://discord.gg/modal) - Get help from the community
- [Your Modal Dashboard](https://modal.com/apps) - Monitor jobs and costs
