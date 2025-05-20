#!/bin/bash

LOG_FOLDER="logs"
LOG_ALPHA="${LOG_FOLDER}/mpc_sm.txt"
LOG_HORIZON="${LOG_FOLDER}/mpc_hor.txt"

if [ -d "$LOG_FOLDER" ]; then rm -r $LOG_FOLDER; fi
mkdir $LOG_FOLDER

HORIZON_ARG=("15" "20" "25" "30" "35" "40" "50")
ALPHA_ARG=("20" "30" "40" "50")
CONT_ARG=("naive" "zerovel" "st" "htwa" "receding" "parallel")

for CONT in "${CONT_ARG[@]}"; do
    # Print controller type
    echo "Running $CONT" | tee -a "$LOG_HORIZON"

    # Horizons
    for HORIZON in "${HORIZON_ARG[@]}"; do 
        echo "running mpc with horizon $HORIZON" | tee -a "$LOG_HORIZON"
        nohup python -u mpc.py -c=$CONT --horizon="$HORIZON" >> "$LOG_HORIZON" 2>&1
        echo "completed execution" | tee -a "$LOG_HORIZON"
        echo "----------------------------------------" | tee -a "$LOG_HORIZON"
    done

    # Safety margin
    echo "Running $CONT" | tee -a "$LOG_HORIZON"

    for ALPHA in "${ALPHA_ARG[@]}"; do 
        echo "running mpc with safety margin $ALPHA" | tee -a "$LOG_ALPHA"
        nohup python -u mpc.py -c=$CONT --alpha="$ALPHA" >> "$LOG_ALPHA" 2>&1
        echo "completed execution" | tee -a "$LOG_ALPHA"
        echo "----------------------------------------" | tee -a "$LOG_ALPHA"
    done

done