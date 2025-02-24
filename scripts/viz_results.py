import pickle
import time
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.utils import obstacles, ee_ref, RobotVisualizer, capsules, capsule_pairs
from safe_mpc.abstract import AdamModel
from safe_mpc.controller import NaiveController


args = parse_args()
nq = args['dofs']
cont_name = args['controller']
alpha = args['alpha']
horizon = args['horizon']
params = Parameters('z1', True)
model = AdamModel(params, nq)
model.ee_ref = ee_ref
robot = NaiveController(model, obstacles, capsules, capsule_pairs)
rviz = RobotVisualizer(params, nq)
rviz.setTarget(ee_ref)
if params.obs_flag:
    rviz.addObstacles(obstacles)

data = pickle.load(open(f'{params.DATA_DIR}z1_{cont_name}_{horizon}hor_{int(alpha)}sm_mpc.pkl', 'rb'))
x = data['x']

for capsule in robot.capsules:
    rviz.init_capsule(capsule)

time.sleep(5)
for j in range(params.test_num):
    print(f"Trajectory {j + 1}")
    rviz.display(x[j][0, :nq])
    time.sleep(1)
    for i in range(params.n_steps):
        # rviz.display(x[j][i, :nq])
        rviz.displayWithEESphere(x[j][i, :nq], robot.capsules)
        time.sleep(params.dt)