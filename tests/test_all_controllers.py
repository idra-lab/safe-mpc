import inspect
import sys
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
import safe_mpc.controller as controllers


conf = Parameters('../config/params.yaml')
model = TriplePendulumModel(conf)
simulator = SimDynamics(conf, model)

# Test all the controllers inside the imported module
module_attributes = dir(controllers)
classes = [cls for cls in module_attributes if inspect.isclass(getattr(controllers, cls))]
for cls in classes:
    print('Testing controller:', cls, file=sys.stderr)
    try:
        model = TriplePendulumModel(conf)
        ocp = getattr(controllers, cls)(conf, model)
        print('Initialization successful', file=sys.stderr)
        del ocp, model                      # Clean up
    except Exception as e:
        print('Initialization failed', file=sys.stderr)
        print(e, file=sys.stderr)
        continue
