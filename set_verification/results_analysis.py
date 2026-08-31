import numpy as np
import matplotlib.pyplot as plt

solver = 'ipopt' # choose between 'acados' and 'ipopt' data

# load tested initial conditions
states_to_verify = np.load('data_results/sampled_states_large.npy')

# load data from acados or ipopt
# if solver == 'acados':
#     results_N_stepCIS = np.load('data_results/verification_results_N_CIS_acados.npy')
#     results_CIS = np.load('data_results/verification_results_CIS_acados.npy')
# elif solver == 'ipopt':
#     results_N_stepCIS = np.load('data_results/verification_results_N_stepCIS.npy')
#     results_CIS = np.load('data_results/verification_results_CIS.npy')

results_CIS = np.load(f'data_results/verification_results_CIS_large.npy')
results_N_stepCIS = np.load('data_results/4dofverification_results_N_stepCIS_large.npy')
data_N_CIS = np.load('data_results/CIS_Unsafe_states_large.npy')

print(f'Successful states for CIS: {np.sum(results_CIS)} / {results_CIS.shape[0]}')
print(f'Successful states for N-step CIS (at least one problem feasible for horizons in 1-45): {np.sum(results_N_stepCIS.any(axis=1))} / {results_N_stepCIS.shape[0]}')

# find inexes at which CIS is not feasible
infeasible_indices = np.where(results_CIS == False)[0]
# print(f'Indices that are not safe for CIS: {infeasible_indices}')

# find indices at which N-step CIS is not feasible for any of the tested horizons
infeasible_N_step_indices = np.where(results_N_stepCIS.any(axis=1)==False)[0]
# print(f'Indices that are not safe for N-step CIS for any of the tested horizons: {infeasible_N_step_indices} and len {infeasible_N_step_indices.shape}')
np.save('data_results/4dofN_step_unsafe_states.npy',data_N_CIS[infeasible_N_step_indices])

random_state_indx = np.random.choice(infeasible_N_step_indices)
print(f'An unsafe state: index {random_state_indx} with state {data_N_CIS[random_state_indx]}')

print('\n\n\n')

# # at CIS infeasible states, find at which horizons N-step CIS becomes feasible, if it is the case
# for idx in infeasible_indices:
#     string_to_print = f'State {states_to_verify[idx]}:  is Unsafe for CIS\n'
#     for horizon in range(results_N_stepCIS[idx,:].shape[0]):
#         string_to_print += f' | Horizon {horizon+1}: {"Safe" if results_N_stepCIS[idx,horizon] else "Unsafe"}\n'  
#     print(string_to_print) 

horizons_result = []
for i in range(results_N_stepCIS.shape[1]):
    horizons_result.append(np.sum(results_N_stepCIS[:,i]))

plt.figure()
plt.bar(range(1, len(horizons_result)+1), horizons_result, width=0.75)
plt.axhline(y=max(horizons_result), color='red', linestyle='--', label=f'Max: {max(horizons_result)}')
plt.xlabel('Horizon')
plt.grid(True)
plt.ylabel('Number of successful states for each horizon')
plt.title(f'Number of successful states for each horizon, {solver} data')
plt.xticks(range(1, len(horizons_result)+1,2), rotation=45)
plt.yticks(range(0, max(horizons_result)+1,10), rotation=0)
plt.show()
plt.savefig(f'data_results/successful_states_per_horizon_{solver}.png')  


print(f'Maximum number of successful states is {max(horizons_result)} at horizon {np.argmax(horizons_result)+1}')
# analysis of N-step CIS excluded unfeasible states for CIS
