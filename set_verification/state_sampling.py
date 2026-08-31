import numpy as np 
from safe_mpc.controller import SafeBackupController
from safe_mpc.set_verification_ocp import ControlInvarianceOCP

import time
import pickle
import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_ocp
from safe_mpc.robot_visualizer import RobotVisualizer
from tqdm import tqdm
from copy import deepcopy

args = parse_args()
model_name = args['system']
params = Parameters(args,model_name, rti=False)
model = AdamModel(params)
set_OCP = ControlInvarianceOCP(model)

alpha = 0.995

n_samples = 100_000
sampled_states = np.zeros((n_samples, model.nx))
sampled_states[:, :model.nq] = np.random.uniform(model.x_min[:model.nq], model.x_max[:model.nq], (n_samples, model.nq))
for i in range(n_samples):
    while not(model.checkCollision(sampled_states[i])):
        sampled_states[i, :model.nq] = np.random.uniform(model.x_min[:model.nq], model.x_max[:model.nq], model.nq)
    velocity_direction_normalized = np.random.uniform(-1, 1, model.nv)
    velocity_direction_normalized /= np.linalg.norm(velocity_direction_normalized)
    sampled_states[i, model.nq:] = velocity_direction_normalized
    sampled_states[i, model.nq:] = alpha * velocity_direction_normalized * np.array(set_OCP.safe_set.pure_nn_output(sampled_states[i])).squeeze()  # fix this to sample on the border of the set
    if set_OCP.safe_set.nn_func(sampled_states[i], 0) < -1e-5:
        print(f'Value constraint state {i}: {set_OCP.safe_set.nn_func(sampled_states[i], 0)}')

np.save('data_results/sampled_states_large.npy', sampled_states)
print(sampled_states)