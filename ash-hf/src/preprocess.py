from datasets import load_dataset, Dataset
import json
import os
from tqdm import tqdm


def _normalize_moves(value):
    """Return a flat list of move strings from a raw field.

    Handles cases where the dataset provides:
    - a single whitespace-delimited string
    - a list of strings
    - nested lists (rare, but defensive)
    - None or other types -> returns empty list
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [m for m in value.strip().split() if m]
    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, str):
                if item:
                    flat.append(item)
            elif isinstance(item, list):
                for inner in item:
                    if isinstance(inner, str) and inner:
                        flat.append(inner)
                    elif inner is not None:
                        flat.append(str(inner))
            elif item is not None:
                flat.append(str(item))
        return flat
    # Fallback: coerce to string
    return [str(value)]

os.makedirs("../data", exist_ok=True)

# ---- Load chess games dataset ----
print("Loading chess games dataset using streaming mode...")
print("This will download data as needed (much faster startup!)")

# Use streaming to avoid downloading entire dataset
games_ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)

# Take first 5000 games using streaming
processed_games = []
print("Processing 5000 games...")
for i, game in enumerate(tqdm(games_ds, total=5000, desc="Games")):
    if i >= 5000:
        break
    try:
        raw_moves = game.get("moves_uci")
        moves = _normalize_moves(raw_moves)
        if not moves:
            # Skip empty move lists
            continue
        processed_games.append({
            "type": "game",
            "moves": moves,
        })
    except Exception as e:
        print(f"Skipping game {i}: {e}")
        continue

with open("../data/games.json", "w") as f:
    json.dump(processed_games, f)

print("Saved games.json with", len(processed_games), "games")

# ---- Load chess puzzles dataset ----
print("\nLoading puzzles dataset using streaming mode...")
puzzles_ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

# Take first 2000 puzzles using streaming
processed_puzzles = []
print("Processing 2000 puzzles...")
for i, p in enumerate(tqdm(puzzles_ds, total=2000, desc="Puzzles")):
    if i >= 2000:
        break
    try:
        fen = p.get("FEN")
        solution_raw = p.get("Moves")
        solution_moves = _normalize_moves(solution_raw)
        if not fen or not solution_moves:
            continue
        processed_puzzles.append({

            "type": "puzzle",
            "fen": fen,
            "solution": solution_moves,
        })
    except Exception as e:
        print(f"Skipping puzzle {i}: {e}")
        continue

with open("../data/puzzles.json", "w") as f:
    json.dump(processed_puzzles, f)

print("Saved puzzles.json with", len(processed_puzzles), "puzzles")
