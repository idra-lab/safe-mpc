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
states_to_verify = np.load('data_results/sampled_states_large.npy')
results = np.zeros(states_to_verify.shape[0],dtype=bool)

indx_failure = []

for i in tqdm(range(states_to_verify.shape[0])):
    x_init = states_to_verify[i]
    result = set_OCP.solveProblem(x_init)[0]
    results[i] = result
    print(f"State {i}_{x_init}: {'Safe' if result else 'Unsafe'}")   
    if not result:
        indx_failure.append(i)  

print(f'safe states: {np.sum(results)} / {results.shape[0]}')
CIS_Unsafe_states = states_to_verify[indx_failure]
np.save('data_results/CIS_Unsafe_states_large.npy', CIS_Unsafe_states)
np.save('data_results/verification_results_CIS_large.npy', results)