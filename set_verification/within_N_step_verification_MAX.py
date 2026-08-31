import numpy as np 
from safe_mpc.set_verification_ocp import NStepControlInvarianceOCP_MAX
import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_ocp
from safe_mpc.robot_visualizer import RobotVisualizer
from tqdm import tqdm
from copy import deepcopy

args = parse_args()
start = args['start_array']
end = args['end_array']
model_name = args['system']
params = Parameters(args,model_name, rti=False)
model = AdamModel(params)
set_OCP = NStepControlInvarianceOCP_MAX(model)
states_to_verify = np.load('data_results/sampled_states.npy')[start:end]
states_to_verify = np.load('data_results/CIS_Unsafe_states_large.npy')[start:end]
horizon_to_test = list(np.arange(int(params.N/2)+1, params.N+1))
horizon_to_test = list(np.arange(1, params.N+1))
# horizon_to_test = list(np.arange(1,10))
results = np.zeros((states_to_verify.shape[0], len(horizon_to_test)),dtype=bool)

threshold_vel_success = 1e-2

for i in tqdm(range(states_to_verify.shape[0])):
    for (j,horizon) in enumerate(reversed(horizon_to_test)):
        x_init = states_to_verify[i]
        result, solution = set_OCP.solveProblem(x_init, horizon)
        if result:
            if (solution[model.nq:] - x_init[model.nq:]) <= 0 and (x_init[model.nq:] - solution[model.nq:]) <= threshold_vel_success:
                results[i, -(j+1)] = result
            else:
                results[i, -(j+1)] = False
        else:
            results[i, -(j+1)] = False
        print(f"State {x_init[:model.nq]} {i} at horizon {horizon}: {'Safe' if result else 'Unsafe'}")     
        # print(f"Solution: {len(set_OCP.X)}")
    if (results[i,:] == False).all():
        print(f'State {i}_{x_init}:  is Unsafe for all horizons')
np.save(f'data_results/verification_results_N_stepCIS_MAX_large_start{start}_end{end}.npy', results)
print(f'succesful states = {np.sum(results.any(axis=1))} / {results.shape[0]}')
