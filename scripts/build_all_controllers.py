from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.controller import SafeBackupController
from safe_mpc.utils import obstacles, ee_ref, get_controller, capsules, capsule_pairs


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True)
params.build = True
params.act = args['activation']
model = AdamModel(params, args['dofs'])
model.ee_ref = ee_ref

controllers = ['naive', 'zerovel', 'st', 'htwa', 'receding']#, 'real']
for cont in controllers:
    # Build each controller
    print(f'\n*** Building {cont.upper()} controller *** \n')
    controller = get_controller(cont, model, obstacles, capsules, capsule_pairs)
    del controller

print('\n*** Building SAFE BACKUP controller *** \n')
params.solver_type = 'SQP'
safe_ocp = SafeBackupController(model, obstacles, capsules, capsule_pairs)

print('DONE!')