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

samples = np.load('data_results/sampled_states_large.npy')
values=[]
for i in range(samples.shape[0]):
    val = set_OCP.safe_set.nn_func_x(samples[i])
    values.append(val)
    print(f'Value constraint state {i}: {val}')
print(f'Mean value constraint: {np.mean(values)}')