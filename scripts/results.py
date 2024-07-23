import pickle
import adam
import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.utils import obstacles, ee_ref, RobotVisualizer


params = Parameters('z1', True)
rviz = RobotVisualizer(params, 4)
rviz.setTarget(ee_ref)
rviz.setInitialBox()
if params.obs_flag:
    rviz.addObstacles(obstacles)