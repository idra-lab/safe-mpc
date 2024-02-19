import inspect
import sys
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
import safe_mpc.controller as controllers


# Test all the controllers inside the imported module
module_attributes = dir(controllers)
classes = [cls for cls in module_attributes if inspect.isclass(getattr(controllers, cls))]
for cls in classes:
    print('Testing controller:', cls, file=sys.stderr)
    try:
        conf = Parameters('triple_pendulum', cls)
        model = TriplePendulumModel(conf)
        simulator = SimDynamics(model)
        ocp = getattr(controllers, cls)(simulator)
        print('Initialization successful', file=sys.stderr)
        # Clean up
        del model, simulator, ocp
    except Exception as e:
        print('Initialization failed', file=sys.stderr)
        print(e, file=sys.stderr)
        continue
