import numpy as np
from casadi import MX, dot, vertcat, sin, cos
from acados_template import AcadosOcp, AcadosOcpSolver
from .abstract import AbstractController


class SafeBackupController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.Q = np.zeros((self.model.nx, self.model.nx))
        self.Q[self.model.nq:, self.model.nq:] = np.eye(self.model.nv) * 1e4

        q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(3)])
        q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(3)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5])


class VbocLikeController:
    def __init__(self, params, model):
        self.N = params.N
        self.model = model

        # Fix the dimension of the parameters field
        model.p =  MX.sym("p", model.nv)
        model.amodel.p = model.p

        # OCP model
        self.ocp = AcadosOcp()
        self.ocp.solver_options.tf = params.T 
        self.ocp.dims.N = self.N 
        self.ocp.model = model.amodel  

        # OCP cost
        self.ocp.cost.cost_type_0 = 'EXTERNAL'

        self.ocp.model.cost_expr_ext_cost_0 = - dot(model.p, model.x[model.nq:])
        self.ocp.parameter_values = np.array([0., 0., 0.])

        # OCP constraints
        self.ocp.constraints.lbx_0 = model.x_min
        self.ocp.constraints.ubx_0 = model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(model.nx)

        self.ocp.constraints.lbu = model.u_min
        self.ocp.constraints.ubu = model.u_max
        self.ocp.constraints.idxbu = np.arange(model.nu)
        self.ocp.constraints.lbx = model.x_min
        self.ocp.constraints.ubx = model.x_max
        self.ocp.constraints.idxbx = np.arange(model.nx)

        # Final constraint -> zero velocity
        q_fin_lb = np.hstack([model.x_min[:model.nq], np.zeros(3)])
        q_fin_ub = np.hstack([model.x_max[:model.nq], np.zeros(3)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.arange(model.nx)

        # Linear constraint
        self.C = np.zeros((self.model.nv + 1, self.model.nx))
        self.ocp.constraints.C = self.C
        self.ocp.constraints.D = np.zeros((self.model.nv + 1, self.model.nu))
        self.ocp.constraints.lg = np.zeros((self.model.nv + 1,))
        self.ocp.constraints.ug = np.zeros((self.model.nv + 1,))

        # Option for external cost function
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        # OCP solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=params.regenerate)


    def solve(self, x0, x_guess, u_guess):
        self.ocp_solver.reset()

        v_norm = np.linalg.norm(x0[self.model.nq:])
        # p -> direction of velocity
        p = x0[self.model.nq:] / v_norm

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', x_guess[i])
            self.ocp_solver.set(i, 'u', u_guess[i])
            self.ocp_solver.set(i, 'p', p)
        self.ocp_solver.set(self.N, 'x', x_guess[-1])
        self.ocp_solver.set(self.N, 'p', p)

        # Set the initial constraint
        d = np.array([p.tolist()])
        self.C[:self.model.nv, self.model.nq:] = np.eye(self.model.nv) - np.matmul(d.T, d)
        self.C[self.model.nv, self.model.nq:] = d
        self.ocp_solver.constraints_set(0, "C", self.C, api='new')

        # norm(v0) <= v_norm only for time step 0
        ug = np.zeros((self.model.nv + 1,))
        ug[-1] = v_norm
        self.ocp_solver.constraints_set(0, "ug", ug)

        # Set initial bounds -> x0_pos = q0, x0_vel free; (final bounds already set)
        q_init_lb = np.hstack([x0[:self.model.nq], self.model.x_min[self.model.nq:]])
        q_init_ub = np.hstack([x0[:self.model.nq], self.model.x_max[self.model.nq:]])
        self.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        self.ocp_solver.constraints_set(0, "ubx", q_init_ub)

        # Solve the OCP
        status = self.ocp_solver.solve()
        self.time = self.ocp_solver.get_stats('time_tot')
        return status