import numpy as np
import matplotlib.pyplot as plt

# load tested initial conditions
# states_to_verify = np.load('data_results/N_step_unsafe_states.npy')

states_tested_back_and_forth = np.load('data_results/4dofN_step_unsafe_states.npy')

results_back_and_forth = np.load('data_results/4dofverification_results_back_and_forth_within_N_CIS_ipopt_large_start.npy') #verification_results_back_and_forth_within_N_CIS_ipopt_two_traj_start.npy')
results_CIS = np.load(f'data_results/verification_results_CIS_large.npy')
results_N_stepCIS = np.load('data_results/4dofverification_results_N_stepCIS_large.npy')

print(f'Successful states for CIS: {np.sum(results_CIS)} / {results_CIS.shape[0]}')
print(f'Successful states for N-step CIS (at least one problem feasible for horizons in 1-45): {np.sum(results_N_stepCIS.any(axis=1))} / {results_N_stepCIS.shape[0]}')
print(f'Successful states for back-and-forth CIS (at least one problem feasible for horizons in 1-45): {results_back_and_forth.any(axis=(1, 2)).sum() + np.sum(results_N_stepCIS.any(axis=1))} / {results_N_stepCIS.shape[0]} ')

# find inexes at which CIS is not feasible
infeasible_indices = np.where(results_CIS == False)[0]
print(f'Indices that are not safe for CIS: {infeasible_indices}')

# find indices at which N-step CIS is not feasible for any of the tested horizons
infeasible_N_step_indices = np.where(results_N_stepCIS.any(axis=1)==False)[0]
print(f'Indices that are not safe for N-step CIS for any of the tested horizons: {infeasible_N_step_indices}')

# find indices at which back-and-forth CIS is not feasible for any of the tested horizons
infeasible_indices_within = np.where(results_N_stepCIS == False)[0]
# print(f'Indices that are not safe for back-and-forth CIS: {infeasible_indices_within}')

feasible_indices_within = np.where(results_back_and_forth.any(axis=(1, 2)) == True)[0]
feasible_back_and_forth_states = states_tested_back_and_forth[feasible_indices_within]

indexes_to_test_posteriori = []
for feasible_index in feasible_indices_within:
    indexes_to_test_posteriori.append(np.argwhere(results_back_and_forth[feasible_index])[0])
indexes_to_test_posteriori = np.array(indexes_to_test_posteriori)
np.save('data_results/4dof_indexes_to_test_posteriori_2nd_run.npy', indexes_to_test_posteriori)

print('\n\n\n')
# horizons_result = np.zeros(results_back_and_forth.shape[1:])
r_j = np.sum(results_back_and_forth, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_pos, z_pos = np.meshgrid(np.arange(r_j.shape[0]), np.arange(r_j.shape[1]), indexing='ij')

y_pos = y_pos.flatten()
z_pos = z_pos.flatten()
values = r_j.flatten()

ax.bar3d(y_pos, z_pos + 1, np.zeros_like(values), 1, 1, values)

ax.set_xlabel('r')
ax.set_ylabel('j')
ax.set_zlabel('Number of successful states')

fig, ax = plt.subplots()
plt.title('Number of Successful States for Back-and-Forth CIS')
im = ax.imshow(r_j.T, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(im, ax=ax, label='Number of successful states')

ax.set_xlabel('r_steps backward')
ax.set_ylabel('j_steps forward')
ax.set_yticks(np.arange(0,r_j.shape[1],2))
ax.set_yticklabels(np.arange(1, r_j.shape[1] + 1,2))  # shift by 1

plt.savefig(f'data_results/4dof_heatmap_successful_states_back_and_forth_within_N_CIS_ipopt_2nd_run_2.png')
plt.show()

best_index_x, best_index_y = np.unravel_index(np.argmax(r_j), r_j.shape)
print(f'Max number of returns: {np.max(r_j)} Number of max values: {np.argwhere(r_j == np.max(r_j))}')
print(f'Maximum number of successful states is {np.max(r_j)} at r-j pair {best_index_x}, {best_index_y + 1}')

np.save(f'data_results/4dof_feasible_states_back_and_forth_2nd_run_2.npy', feasible_back_and_forth_states)
