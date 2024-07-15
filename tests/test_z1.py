import time
import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.abstract import AdamModel
from safe_mpc.controller import NaiveController 
import pinocchio as pin


rti = False
N = 30
params = Parameters('z1', rti)
# params.build = True
model = AdamModel(params, n_dofs=4)
ee_ref = np.array([0.35, 0.2, 0.12])
controller = NaiveController(model)
controller.resetHorizon(N)
controller.setReference(ee_ref)


q0 = np.array([ -0.5, 0.9 , -0.4,  0.5])
x0 = np.zeros((model.nx,))
x0[:model.nq] = q0
u0 = np.zeros(model.nq)        

flag = controller.initialize(x0, u0)    
controller.ocp_solver.print_statistics()
if flag:
    print('\nSuccess!\n')
    x_guess, u_guess = controller.getGuess()
else:
    print('\nFailed!\n')


description_dir = params.ROBOTS_DIR + 'z1_description'
rmodel, collision, visual = pin.buildModelsFromUrdf(description_dir + '/urdf/z1.urdf',
                                                    package_dirs=params.ROOT_DIR)
geom = [collision, visual]

lockIDs = []
lockNames = ['joint5', 'joint6', 'jointGripper']
for name in lockNames:
    lockIDs.append(rmodel.getJointId(name))

rmodel_red, geom_red = pin.buildReducedModel(rmodel, geom, lockIDs, np.zeros(7))

viz = pin.visualize.MeshcatVisualizer(rmodel_red, geom_red[0], geom_red[1])
viz.initViewer(loadModel=True, open=True)
viz.display(q0)

time.sleep(3)
for i in range(1, N+1):
    viz.display(x_guess[i, :model.nq])
    time.sleep(params.dt)
