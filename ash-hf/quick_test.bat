@echo off
echo ==========================================
echo QUICK OPTIMIZATION TEST
echo ==========================================
echo.
echo This will:
echo - Run 5 iterations
echo - 5 games per iteration
echo - 20 MCTS simulations
echo.
echo Expected:
echo - Complete in ~10 minutes
echo - Cost: ~$0.20
echo - Games: 3-8 seconds each
echo.
echo Watch for 'Using device: cuda' in logs!
echo.
pause

cd src
modal run train_continuous_modal.py --iterations 5 --games 5 --simulations 20

echo.
echo ==========================================
echo Test complete!
echo.
echo Check timing output:
echo - First game should be 5-15s (CUDA warmup)
echo - Later games should be 3-8s
echo.
echo If working well, run full training:
echo   cd src
echo   modal run train_continuous_modal.py --iterations 500 --games 20 --simulations 30
echo ==========================================
pause
