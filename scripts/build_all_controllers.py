from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_controller
from safe_mpc.controller import SafeBackupController
from safe_mpc.cost_definition import *


args = parse_args()
model_name = args['system']

controllers = ['naive', 'zerovel', 'st', 'htwa', 'receding', 'parallel']
for cont in controllers:
    # Build each controller
    print(f'\n*** Building {cont.upper()} controller *** \n')
    params = Parameters(model_name, rti=True)
    model = AdamModel(params)
    model.ee_ref = params.ee_ref
    controller = get_controller(cont, model)
    cost_controller = ReachTargetNLS(model, params.Q_weight, params.R_weight)
    cost_controller.set_solver_cost(controller)
    controller.build_controller(True)
    del params, model, controller

print('\n*** Building SAFE BACKUP controller *** \n')
param_backup = Parameters(model_name, rti=True)
param_backup.use_net = None
param_backup.N = args['back_hor']
param_backup.solver_type = 'SQP_RTI'
model_backup = AdamModel(param_backup)
safe_ocp = SafeBackupController(model_backup)
cost_controller_backup = ZeroCost(model_backup)
cost_controller_backup.set_solver_cost(safe_ocp)
safe_ocp.build_controller(True)

print('DONE!')