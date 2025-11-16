# Chess AI Model - Usage Guide

## Model Architecture

**Type:** AlphaZero-style Chess Neural Network
**Framework:** PyTorch
**Input:** 18-channel 8×8 board representation
**Output:** Policy (4096 move probabilities) + Value (position evaluation)

### Network Structure
- **Input channels:** 18
  - Channels 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
  - Channels 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
  - Channel 12: Current player turn (1.0 for white, 0.0 for black)
  - Channels 13-16: Castling rights (white kingside, white queenside, black kingside, black queenside)
  - Channel 17: En passant squares

- **Residual blocks:** 3 (optimized from 5)
- **Filters:** 64 (optimized from 128)
- **Policy head:** Outputs 4096 move probabilities (64×64 from-to square encoding)
- **Value head:** Outputs single number from -1 (black winning) to +1 (white winning)

## File Format

Checkpoints are saved as PyTorch `.pt` files with this structure:

```python
{
    'model_state_dict': model.state_dict(),  # Model weights
    'iteration': 42,                         # Training iteration number
    'optimizer_state_dict': optimizer.state_dict()  # Optimizer state (optional)
}
```

## Loading the Model

### Basic Loading

```python
import torch
from train import ChessNet

# Create model with same architecture
model = ChessNet(
    input_channels=18,
    num_filters=64,
    num_residual_blocks=3
)

# Load checkpoint
checkpoint = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

print(f"Loaded model from iteration {checkpoint['iteration']}")
```

### Loading with GPU

```python
import torch
from train import ChessNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ChessNet(
    input_channels=18,
    num_filters=64,
    num_residual_blocks=3
).to(device)

checkpoint = torch.load('best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Using the Model

### 1. Get Move Recommendations (with MCTS)

```python
import chess
from train import MCTS, encode_board, move_to_index
import numpy as np

# Setup
board = chess.Board()  # or chess.Board("fen string here")
mcts = MCTS(model, num_simulations=100)

# Run search
visit_counts = mcts.search(board)

# Get best move
legal_moves = list(board.legal_moves)
best_move = max(legal_moves, key=lambda m: visit_counts[move_to_index(m)])
print(f"Best move: {best_move.uci()}")
```

### 2. Direct Neural Network Inference (No MCTS)

```python
import torch
import chess
from train import encode_board, move_to_index
import numpy as np

# Encode board
board = chess.Board()
board_tensor = torch.FloatTensor(encode_board(board)).unsqueeze(0)  # Add batch dimension
board_tensor = board_tensor.to(device)

# Get predictions
with torch.no_grad():
    policy_logits, value = model(board_tensor)
    policy_probs = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
    value = value.item()

print(f"Position evaluation: {value:.3f}")  # -1.0 to +1.0
print(f"Top move probabilities:")

# Get probabilities for legal moves only
legal_moves = list(board.legal_moves)
move_probs = []
for move in legal_moves:
    idx = move_to_index(move)
    move_probs.append((move, policy_probs[idx]))

# Sort by probability
move_probs.sort(key=lambda x: x[1], reverse=True)
for move, prob in move_probs[:5]:
    print(f"  {move.uci()}: {prob:.4f}")
```

### 3. Play Against the Model

```python
import chess
from train import MCTS, move_to_index

def get_model_move(board, model, simulations=100):
    """Get best move from model using MCTS"""
    mcts = MCTS(model, num_simulations=simulations)
    visit_counts = mcts.search(board)

    # Get legal moves with their visit counts
    legal_moves = list(board.legal_moves)
    best_move = max(legal_moves, key=lambda m: visit_counts[move_to_index(m)])
    return best_move

# Play a game
board = chess.Board()
while not board.is_game_over():
    print(board)
    print()

    if board.turn == chess.WHITE:
        # Human move
        move_uci = input("Your move (UCI format, e.g. 'e2e4'): ")
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Illegal move!")
                continue
        except:
            print("Invalid format!")
            continue
    else:
        # AI move
        print("AI is thinking...")
        move = get_model_move(board, model, simulations=100)
        print(f"AI plays: {move.uci()}")
        board.push(move)

print(f"Game over: {board.result()}")
```

### 4. Batch Evaluation (Multiple Positions)

```python
import torch
import chess
from train import encode_board
import numpy as np

# Multiple positions
boards = [
    chess.Board(),
    chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3")
]

# Encode all boards
encoded = np.array([encode_board(b) for b in boards])
boards_tensor = torch.FloatTensor(encoded).to(device)

# Batch inference
with torch.no_grad():
    policy_logits, values = model(boards_tensor)
    values = values.cpu().numpy().flatten()

for i, (board, value) in enumerate(zip(boards, values)):
    print(f"Position {i+1}: {value:.3f}")
```

## Model Strength by Training Stage

| Iterations | Expected Strength | Characteristics |
|-----------|------------------|-----------------|
| 0-50 | Random/Beginner | Mostly draws, basic moves |
| 50-100 | Novice (500-800 Elo) | Learns piece values, fewer draws |
| 100-200 | Beginner (800-1200 Elo) | Basic tactics, opening principles |
| 200-500 | Amateur (1200-1600 Elo) | Tactical combinations, positional play |
| 500-1000 | Intermediate (1600-2000 Elo) | Strategic understanding, endgame skill |
| 1000+ | Advanced (2000+ Elo) | Deep strategic play, strong tactics |

## Optimizations Applied

This model includes several performance optimizations:

1. **Batched MCTS** - Evaluates 8 positions simultaneously
2. **Mixed Precision (FP16)** - Faster training with automatic mixed precision
3. **torch.compile** - JIT compilation for faster inference
4. **TensorFloat32** - Hardware acceleration on Ampere GPUs
5. **Optimized board encoding** - Vectorized operations
6. **Reduced network size** - 3 blocks × 64 filters (vs 5 blocks × 128 filters)

**Expected performance:**
- Self-play games: 3-8 seconds (with optimizations)
- Training iteration: 2-4 minutes (20 games)
- Inference latency: ~10-20ms per position (batched)

## Common Issues

### Issue: "Missing key(s) in state_dict: '_orig_mod...'"

**Cause:** Model was saved with `torch.compile`, keys have `_orig_mod.` prefix

**Fix:**
```python
checkpoint = torch.load('best_model.pt', map_location=device)
state_dict = checkpoint['model_state_dict']

# Remove _orig_mod. prefix
cleaned_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('_orig_mod.'):
        cleaned_state_dict[k[len('_orig_mod.'):]] = v
    else:
        cleaned_state_dict[k] = v

model.load_state_dict(cleaned_state_dict)
```

### Issue: "size mismatch for conv_initial.0.weight"

**Cause:** Model architecture doesn't match checkpoint (different num_filters or num_residual_blocks)

**Fix:** Use matching architecture parameters when creating the model:
```python
# For new optimized checkpoints (default):
model = ChessNet(input_channels=18, num_filters=64, num_residual_blocks=3)

# For old checkpoints (if you have any from before optimization):
model = ChessNet(input_channels=18, num_filters=128, num_residual_blocks=5)
```

## File Locations

- **Modal volume:** Checkpoints saved to `/root/data/weights/` on Modal
- **Local download:** Downloaded to `ash-hf/weights/` locally
- **Checkpoint naming:**
  - `best_model.pt` - Best model so far (lowest loss)
  - `iteration_10.pt`, `iteration_20.pt`, etc. - Saved every 10 iterations

## Advanced: Converting to Other Formats

### Export to ONNX

```python
import torch
from train import ChessNet

model = ChessNet(input_channels=18, num_filters=64, num_residual_blocks=3)
checkpoint = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 18, 8, 8)

# Export
torch.onnx.export(
    model,
    dummy_input,
    'chess_model.onnx',
    input_names=['board'],
    output_names=['policy', 'value'],
    dynamic_axes={
        'board': {0: 'batch_size'},
        'policy': {0: 'batch_size'},
        'value': {0: 'batch_size'}
    }
)
```

### Save for TorchScript

```python
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('chess_model.pt')

# Load later
loaded = torch.jit.load('chess_model.pt')
```

## FAQ

**Q: How do I know which checkpoint is best?**
A: `best_model.pt` is automatically updated with the model that has the lowest training loss. Use this for deployment.

**Q: Should I use MCTS or direct inference?**
A: Always use MCTS for best play quality. Direct inference is faster but much weaker (~200-300 Elo difference).

**Q: How many MCTS simulations should I use?**
A:
- Fast play: 25-50 simulations (~1-2 seconds)
- Standard play: 100 simulations (~3-5 seconds)
- Strong play: 200-400 simulations (~8-15 seconds)
- Tournament play: 800+ simulations (30+ seconds)

**Q: Can I run this on CPU?**
A: Yes, but it's ~10-50x slower. MCTS search will take 30-60 seconds per move on CPU vs 3-5 seconds on GPU.

**Q: How big are the checkpoint files?**
A: ~5-10 MB per checkpoint (small network size makes them portable)

**Q: Why are all games drawing in early training?**
A: This is completely normal! Untrained models play poorly, so both sides make mistakes equally and games end in draws (stalemate, threefold repetition, 50-move rule). After 50-100 iterations, you'll see more decisive games.
