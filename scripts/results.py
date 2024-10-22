import pickle
import time
import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.utils import obstacles, ee_ref, RobotVisualizer


params = Parameters('z1', True)
rviz = RobotVisualizer(params, 4)
rviz.setTarget(ee_ref)
# rviz.setInitialBox()
if params.obs_flag:
    rviz.addObstacles(obstacles)

data = pickle.load(open(f'{params.DATA_DIR}z1_naive_mpc.pkl', 'rb'))
x = data['x'][0]

time.sleep(5)
for i in range(params.n_steps):
    rviz.display(x[i, :4])
    time.sleep(params.dt)