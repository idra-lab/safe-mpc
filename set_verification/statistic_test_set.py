import numpy as np 
from safe_mpc.controller import SafeBackupController
from safe_mpc.set_verification_ocp import ControlInvarianceOCP, UpToNStepControlInvariance, BackAndForthNStepControlInvariance, BackAndForthWithinNStepControlInvariance

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
model = Parameters(model_name, rti=False)

set_OCPS = {}
set_OCPS['CIS'] = ControlInvarianceOCP(model)
set_OCPS['UpToNCis'] = UpToNStepControlInvariance(model)
set_OCPS['BFNCIS'] = BackAndForthNStepControlInvariance(model)
set_OCPS['BFWithNCIS'] = BackAndForthWithinNStepControlInvariance(model)

# sampling states from the set and place it on the border

n_points = 100_000

sampled_points = np.random.uniform(model.x_min,model.x_max,(n_points,model.nx))

for i in range(n_points):
    val_net = set_OCPS['CIS'].safe_set.nn_func_x(sampled_points[i])
    while val_net > 0.05 or val_net < 0:
        if val_net > 0:
            sampled_points[i,model.nq:] *= 0.97
        elif val_net < 0:
            sampled_points[i,model.nq:] *= 1.03
        val_net = set_OCPS['CIS'].safe_set.nn_func_x(sampled_points[i])

