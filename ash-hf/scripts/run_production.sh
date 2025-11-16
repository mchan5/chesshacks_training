#!/bin/bash
# Production training with optimized settings
# Balanced for speed and quality

echo "=========================================="
echo "PRODUCTION TRAINING (OPTIMIZED)"
echo "=========================================="
echo ""
echo "Settings:"
echo "- 500 iterations"
echo "- 20 games per iteration"
echo "- 30 MCTS simulations"
echo ""
echo "Expected:"
echo "- Time: 30-40 hours"
echo "- Cost: ~$35-45 (A10G)"
echo "- Strong amateur-level model"
echo ""
echo "The job will continue even if you close this terminal."
echo "Monitor progress at: https://modal.com/apps"
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
echo "Starting training on Modal..."
echo "Press Ctrl+C to stop monitoring (training continues)"
echo ""

modal run train_continuous_modal.py \
  --iterations 500 \
  --games 20 \
  --simulations 30

echo ""
echo "=========================================="
echo "Monitor stopped. Training continues on Modal!"
echo ""
echo "Download model with:"
echo "  cd ash-hf/src"
echo "  modal run train_modal.py --mode download --checkpoint best_model.pt"
echo ""
echo "Check status:"
echo "  modal app list"
echo "  modal run train_modal.py --mode list"
echo "=========================================="
