import numpy as np
from casadi import SX, dot
from .abstract import AbstractController


class SafeBackupController(AbstractController):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.Q = np.eye(model.nx)
        self.Q[model.nq:, model.nq:] = np.eye(model.nv) * 1e4

        q_fin_lb = np.hstack([model.x_min[:model.nq], np.zeros(3)])
        q_fin_ub = np.hstack([model.x_max[:model.nq], np.zeros(3)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5])


class VbocLikeController(SafeBackupController):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.C = np.zeros((self.model.nv + 1, self.model.nx))

    def additionalSetting(self, params):
        # Define the external cost for t = 0
        p = SX.sym("p", self.model.nv)
        self.model.p = p
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_0 = dot(p, self.model.x[:self.model.nv])
        self.ocp.parameter_values = np.zeros((self.model.nv,))

        # Set the cost to zero for LINEAR_LS
        self.ocp.cost.W = np.zeros((self.model.ny, self.model.ny))
        self.ocp.cost.W_e = np.zeros((self.model.nx, self.model.nx))

        # Linear constraint
        self.ocp.constraints.C = self.C
        self.ocp.constraints.D = np.zeros((self.model.nv + 1, self.model.nu))
        self.ocp.constraints.lg = np.zeros((self.model.nv + 1,))
        self.ocp.constraints.ug = np.zeros((self.model.nv + 1,))

    def solve(self, x0):
        self.ocp_solver.reset()

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', self.model.p)
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', self.model.p)

        # Set the initial constraint
        d = np.array([self.model.p.tolist()])
        self.C[:self.model.nv, self.model.nq:] = np.eye(self.model.nv) - np.matmul(d.T, d)
        self.C[self.model.nv, self.model.nq:] = d
        self.ocp_solver.constraints_set(0, "C", self.C, api='new')

        # norm(v0) <= v_norm only for time step 0
        ug = np.zeros((self.model.nv + 1,))
        ug[-1] = np.linalg.norm(x0[self.model.nq:])
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
