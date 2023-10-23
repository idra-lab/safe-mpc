import numpy as np
from casadi import SX
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver


class AbstractModel:
    def __init__(self, params):
        self.model = AcadosModel()
        self.addDynamicsModel(params)
        self.model.f_expl_expr = self.f_expl
        f_impl = self.x_dot - self.f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = self.x_dot
        self.model.u = self.u
        self.model.p = self.p

        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = int(self.nx / 2)

        # Joint limits
        self.u_min = -params.u_max * np.ones(self.nu)
        self.u_max = params.u_max * np.ones(self.nu)

        self.x_min = np.hstack([params.q_min * np.ones(self.nq), -params.dq_max * np.ones(self.nq)])
        self.x_max = np.hstack([params.q_max * np.ones(self.nq), params.dq_max * np.ones(self.nq)])

    def addDynamicsModel(self, params):
        """ Define a dummy dynamics model (simple integrator). """
        self.model.name = "model"
        self.x = SX.sym("x")
        self.x_dot = SX.sym("x_dot")
        self.u = SX.sym("u")
        self.f_expl = self.u 
        self.p = SX.sym("p")

    def checkStateConstraints(self, x):
        return np.all((x >= self.x_min) & (x <= self.x_max))

    def checkControlConstraints(self, u):
        return np.all((u >= self.u_min) & (u <= self.u_max))
    
    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) and self.checkControlConstraints(u)

    def checkDynamicsConstraints(self, x, u):
        pass

    def checkSafeSetConstraints(self, x):
        pass

    def getNNmodel(self, params):
        pass


class SimDynamics:
    def __init__(self, params, model):
        self.model = model
        sim = AcadosSim()
        sim.model = model.model
        sim.solver_options.T = params.dt
        sim.solver_options.num_stages = 4
        sim.parameter_values = np.array([0.])
        self.integrator = AcadosSimSolver(sim, build=params.regenerate)

    def simulate(self, x, u):
        self.integrator.set("x", x)
        self.integrator.set("u", u)
        self.integrator.solve()
        x_next = self.integrator.get("x")
        return x_next 
