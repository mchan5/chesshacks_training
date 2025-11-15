from datasets import load_dataset, Dataset
import json
import os

os.makedirs("../data", exist_ok=True)

# ---- Load chess games dataset ----
print("Loading chess games...")
games_ds = load_dataset("angeluriot/chess_games", split="train")

# We'll take a small subset for demonstration
games_sample = games_ds.shuffle(seed=42).select(range(5000))  # 5k games for now

processed_games = []
for game in games_sample:
    moves = game["moves_uci"].split()
    processed_games.append({
        "type": "game",
        "moves": moves
    })

with open("../data/games.json", "w") as f:
    json.dump(processed_games, f)

print("Saved games.json with", len(processed_games), "games")

# ---- Load chess puzzles dataset ----
print("Loading puzzles dataset...")
puzzles_ds = load_dataset("Lichess/chess-puzzles", split="train")

puzzles_sample = puzzles_ds.shuffle(seed=42).select(range(2000))  # 2k puzzles

processed_puzzles = []
for p in puzzles_sample:
    # We'll store FEN + solution moves
    processed_puzzles.append({
        "type": "puzzle",
        "fen": p["FEN"],
        "solution": p["Moves"].split()
    })

with open("../data/puzzles.json", "w") as f:
    json.dump(processed_puzzles, f)

print("Saved puzzles.json with", len(processed_puzzles), "puzzles")
