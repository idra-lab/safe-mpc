#!/bin/bash

# Total number of verification cases
# Get total size from the Python script
TOTAL=$(python3 -c "
import numpy as np
arr = np.load('data_results/N_step_unsafe_states.npy')  
print(arr.shape[0])
")

echo "Total verification cases: $TOTAL"

N_JOBS=15
CHUNK=$(((TOTAL / N_JOBS)+1))

trap "echo 'Killing all jobs...'; kill 0; exit 1" SIGINT

for i in $(seq 0 $((N_JOBS - 1))); do
    START=$((i * CHUNK))
    END=$(((i+1) * CHUNK))
    python3 back_and_forth_two_traj.py --start_array $START --end_array $END > logs/back_and_forth_two_traj${START}_${END}.log &
    sleep 5  # wait 30 seconds before launching next job
done


wait
echo "All jobs done, merging results..."
python3 merge_results.py verification_results_back_and_forth_within_N_CIS_ipopt_two_traj_start
echo "Merged results saved to verification_results_back_and_forth_within_N_CIS_ipopt_two_traj_start.npy"