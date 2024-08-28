import pickle
import numpy as np
from functools import reduce
from safe_mpc.parser import Parameters


def open_pickle(name):
    return pickle.load(open(conf.DATA_DIR + name + '_results.pkl', 'rb'))


def state_traj_cost(x, x_ref, Q):
    n = x.shape[0]
    cost = 0
    for i in range(n):
        cost += (x[i] - x_ref).T @ Q @ (x[i] - x_ref)
    return cost


def mean_cost(x_traj, x_ref):
    n = len(x_traj)
    Q = np.eye(6) * 1e-4
    Q[0, 0] = 500
    cost = 0
    for i in range(n):
        temp = x_traj[i]
        temp = temp[~np.isnan(x_traj[i]).any(axis=1)]
        cost += state_traj_cost(temp, x_ref, Q)
    return cost/n


conf = Parameters('triple_pendulum', 'naive')
controllers = ['naive', 'st', 'stwa', 'htwa', 'receding']
x_dict, idx_dict = dict(), dict()
for c in controllers:
    data = open_pickle(c)
    x_dict[c] = data['x_sim']
    idx_dict[c] = data['idx_abort']

x_r = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
# Find the tests for which at least one controller fails to solve the task
all_aborts = reduce(np.union1d, ([idx_dict[c] for c in controllers]))
# Thus find the tests for which all the controllers can solve the task
idx_common = np.setdiff1d(np.arange(conf.test_num), all_aborts)

# Compute the mean cost for the common tests
costs = dict()
print('Mean cost for the common tests:')
for c in controllers:
    costs[c] = mean_cost(x_dict[c][idx_common], x_r)
    print(c + ': ' + str(costs[c]))

# Compute the percentage increase in cost wrt naive
print('\nPercentage increase in cost wrt naive:')
for c in controllers[1:]:
    print(c + ': ' + str(100 * (costs[c] - costs['naive']) / costs['naive']))
