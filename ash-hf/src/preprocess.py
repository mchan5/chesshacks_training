from datasets import load_dataset, Dataset
import json
import os
from tqdm import tqdm
import argparse


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

parser = argparse.ArgumentParser(description="Stream and preprocess chess games and puzzles")
parser.add_argument(
    "--games",
    type=str,
    default="20000",
    help="Number of games to process, or 'all' for entire dataset (default: 20000)",
)
parser.add_argument(
    "--puzzles",
    type=str,
    default="10000",
    help="Number of puzzles to process, or 'all' for entire dataset (default: 10000)",
)
args = parser.parse_args()


def _parse_count(value: str | int | None):
    if value is None:
        return None
    if isinstance(value, int):
        return max(0, value)
    s = str(value).strip().lower()
    if s in {"all", "max", "inf", "infinite", "none"}:
        return None  # None indicates no limit (use all)
    try:
        return max(0, int(s))
    except ValueError:
        raise SystemExit(f"Invalid count: {value!r}. Use a non-negative integer or 'all'.")


games_target = _parse_count(args.games)
puzzles_target = _parse_count(args.puzzles)

os.makedirs("../data", exist_ok=True)

# ---- Load chess games dataset ----
print("Loading chess games dataset using streaming mode...")
print("This will download data as needed (much faster startup!)")

# Use streaming to avoid downloading entire dataset
games_ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)

# Take first N games using streaming (or all if None)
processed_games = []
games_jsonl_path = None
games_written = 0
print(f"Processing {'all' if games_target is None else games_target} games...")
if games_target is None:
    games_jsonl_path = os.path.join("..", "data", "games.jsonl")
    # Fresh file for this run
    if os.path.exists(games_jsonl_path):
        os.remove(games_jsonl_path)

for i, game in enumerate(tqdm(games_ds, total=games_target, desc="Games")):
    if games_target is not None and i >= games_target:
        break
    try:
        raw_moves = game.get("moves_uci")
        moves = _normalize_moves(raw_moves)
        if not moves:
            # Skip empty move lists
            continue
        record = {"type": "game", "moves": moves}
        if games_jsonl_path:
            with open(games_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            games_written += 1
        else:
            processed_games.append(record)
    except Exception as e:
        print(f"Skipping game {i}: {e}")
        continue

if games_jsonl_path:
    print("Saved games to JSONL:", games_jsonl_path, "records:", games_written)
else:
    with open("../data/games.json", "w") as f:
        json.dump(processed_games, f)
    print("Saved games.json with", len(processed_games), "games")

# ---- Load chess puzzles dataset ----
print("\nLoading puzzles dataset using streaming mode...")
puzzles_ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

# Take first M puzzles using streaming (or all if None)
processed_puzzles = []
puzzles_jsonl_path = None
puzzles_written = 0
print(f"Processing {'all' if puzzles_target is None else puzzles_target} puzzles...")
if puzzles_target is None:
    puzzles_jsonl_path = os.path.join("..", "data", "puzzles.jsonl")
    if os.path.exists(puzzles_jsonl_path):
        os.remove(puzzles_jsonl_path)

for i, p in enumerate(tqdm(puzzles_ds, total=puzzles_target, desc="Puzzles")):
    if puzzles_target is not None and i >= puzzles_target:
        break
    try:
        fen = p.get("FEN")
        solution_raw = p.get("Moves")
        solution_moves = _normalize_moves(solution_raw)
        if not fen or not solution_moves:
            continue
        record = {"type": "puzzle", "fen": fen, "solution": solution_moves}
        if puzzles_jsonl_path:
            with open(puzzles_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            puzzles_written += 1
        else:
            processed_puzzles.append(record)
    except Exception as e:
        print(f"Skipping puzzle {i}: {e}")
        continue

if puzzles_jsonl_path:
    print("Saved puzzles to JSONL:", puzzles_jsonl_path, "records:", puzzles_written)
else:
    with open("../data/puzzles.json", "w") as f:
        json.dump(processed_puzzles, f)
    print("Saved puzzles.json with", len(processed_puzzles), "puzzles")
