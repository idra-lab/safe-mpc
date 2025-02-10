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

data = pickle.load(open(f'{params.DATA_DIR}z1_receding_45hor_50sm_mpc.pkl', 'rb'))
x = data['x']

time.sleep(5)
for j in range(params.test_num):
    print(f"Trajectory {j + 1}")
    rviz.display(x[j][0, :4])
    time.sleep(1)
    for i in range(params.n_steps):
        rviz.display(x[j][i, :4])
        time.sleep(params.dt)