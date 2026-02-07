#!/bin/bash

# Enhanced script with process tracking

echo "========================================"
echo "Starting MMM Simulations"
echo "Start time: $(date)"
echo "========================================"

# Array of simulation files
simulations=(
    "main_true_hill_c.py"
    "main_true_hill_s.py"
    "main_true_weibull_c.py"
    "main_true_weibull_s.py"
    "main_true_sigmoid.py"
    "main_true_error.py"
)

# Start all simulations
for sim in "${simulations[@]}"; do
    logfile="log_${sim%.py}.txt"
    echo "Starting: $sim -> $logfile"
    nohup python -u "$sim" > "$logfile" 2>&1 &
    pid=$!
    echo "  PID: $pid"
done

echo ""
echo "========================================"
echo "All simulations started!"
echo "========================================"
echo ""
echo "Useful commands:"
echo "  Monitor all logs:      tail -f log_*.txt"
echo "  Check processes:       ps aux | grep main_true"
echo "  Stop all:              pkill -f main_true"
echo "  Check completion:      ls -lh results_*.csv"
echo ""
