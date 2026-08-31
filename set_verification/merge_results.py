import numpy as np
import glob
import sys
import re


def _chunk_sort_key(path):
    """Sort files by numeric start/end indices in the filename."""
    m = re.search(r'_start(\d+)_end(\d+)\.npy$', path)
    if m is None:
        return (float('inf'), float('inf'), path)
    return (int(m.group(1)), int(m.group(2)), path)

# if sys.argv[1] == 'acados':
#     files = sorted(
#         glob.glob('data_results/verification_results_N_CIS_acados_*.npy'),
#         key=_chunk_sort_key,
#     )
#     print(files)
#     results = np.concatenate([np.load(f) for f in files], axis=0)
#     np.save('data_results/verification_results_N_CIS_acados.npy', results)
#     print(f'Successful states = {np.sum(results.any(axis=1))} / {results.shape[0]}')
#     # concatenate also log files
#     log_files = sorted(glob.glob('data_results/N_STEP_acados_*.log'),key=_chunk_sort_key) 
#     with open("combined_log_N_step_acados.log", "w") as outfile:
#         for log_file in log_files:
#             with open(log_file, "r") as infile:
#                 outfile.write(infile.read())    
# elif sys.argv[1] == 'ipopt':.

file_key_name = sys.argv[1]

files = sorted(
    glob.glob(f'data_results/{file_key_name}*.npy'),
    key=_chunk_sort_key,
)
print(files)
results = np.concatenate([np.load(f) for f in files], axis=0)
np.save(f'data_results/{file_key_name}.npy', results)
print(f'Successful states = {np.sum(results.any(axis=1))} / {results.shape[0]}')
# # concatenate also log files
# log_files = sorted(glob.glob('data_results/N_STEP_log_*.log'),key=_chunk_sort_key) 
# with open(f"combined_log_{file_key_name}.log", "w") as outfile:
#     for log_file in log_files:
#         with open(log_file, "r") as infile:
#             outfile.write(infile.read())   


