#!/bin/bash
# Development training - fast iterations for testing
# Good for experimenting with hyperparameters

echo "=========================================="
echo "DEVELOPMENT TRAINING (FAST)"
echo "=========================================="
echo ""
echo "Settings:"
echo "- 100 iterations"
echo "- 10 games per iteration"
echo "- 25 MCTS simulations"
echo ""
echo "Expected:"
echo "- Time: 3-5 hours"
echo "- Cost: ~$4-6 (A10G)"
echo "- Good for testing/development"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

cd "$(dirname "$0")/../src"

echo ""
echo "Starting development training..."
echo ""

modal run train_continuous_modal.py \
  --iterations 100 \
  --games 10 \
  --simulations 25

echo ""
echo "Development training complete!"
