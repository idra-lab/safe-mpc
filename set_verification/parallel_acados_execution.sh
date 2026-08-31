#!/bin/bash

# Total number of verification cases
# Get total size from the Python script
TOTAL=$(python3 -c "
import numpy as np
arr = np.load('data_results/sampled_states.npy')  
print(arr.shape[0])
")

echo "Total verification cases: $TOTAL"

N_JOBS=10
CHUNK=$((TOTAL / N_JOBS))

trap "echo 'Killing all jobs...'; kill 0; exit 1" SIGINT

for i in $(seq 0 $((N_JOBS - 1))); do
    START=$((i * CHUNK))
    END=$(((i+1) * CHUNK))
    python3 acados_set_verification_N_CIS.py --start_array $START --end_array $END > logs/N_STEP_acados_log_${START}_${END}.log &
    sleep 5  # wait 30 seconds before launching next job
done


wait
echo "All jobs done, merging results..."
python3 merge_results.py acados
echo "Merged results saved to verification_results_N_CIS_acados.npy"