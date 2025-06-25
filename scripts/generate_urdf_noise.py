import pickle
import numpy as np
from functools import reduce
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_controller, randomize_model
from safe_mpc.controller import SafeBackupController
from safe_mpc.cost_definition import *


np.random.seed(10)

args = parse_args()
model_name = args['system']
params = Parameters(args,model_name, rti=True)
params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']

NOISES=[0.0]#,0.1,1.3,2.5,3.7 ,5]

for noise in NOISES:
    for i in range(params.test_num):
        randomize_model(params.robot_urdf, noise_mass = params.noise_mass, noise_inertia = params.noise_inertia, noise_cm_position = params.noise_cm, controller_name=(f'noise{noise}_{i}'))
        print(f'Iteration {i}, noise {noise}')