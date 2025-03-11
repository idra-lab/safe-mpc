import re
import numpy as np
import scipy.linalg as lin
from casadi import Function, norm_2
from acados_template import AcadosOcp, AcadosOcpSolver
from copy import deepcopy
import casadi as cs
import sympy as sym

class AbstractController:
    def __init__(self, model):
        self.model = model

        self.N = self.model.params.N
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.model.params.dt * self.N
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.Q = self.model.params.Q if not(self.model.params.track_traj) else 5e2 * np.eye(3)
        self.R = self.model.params.R * np.eye(self.model.nu) 

        self.ocp.cost.cost_type = 'EXTERNAL'
        self.ocp.cost.cost_type_e = 'EXTERNAL'

        t_glob = self.model.t_glob
        delta = t_glob - self.model.p[:3]    #self.model.ee_ref
        track_ee = delta.T @ self.Q @ delta 
        self.ocp.model.cost_expr_ext_cost = track_ee + self.model.u.T @ self.R @ self.model.u if not(self.model.params.track_traj) else track_ee
        self.ocp.model.cost_expr_ext_cost_e = track_ee
        self.ocp.parameter_values = np.hstack([self.model.ee_ref, [self.model.params.alpha, 1.]])

        # Bound constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)
        
        # Nonlinear constraints
        self.nl_con_0, self.nl_con, self.nl_con_e = self.model.NL_external
        self.idxhs_0,self.idxhs,self.idxhs_e = np.zeros(0),np.zeros(0),np.zeros(0)
        self.zl_0,self.zu_0,self.Zl_0,self.Zu_0 = np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)
        self.zl,self.zu,self.Zl,self.Zu = np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)
        self.zl_e,self.zu_e,self.Zl_e,self.Zu_e = np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)


        # Additional settings, in general is an empty method
        self.additionalSetting()

        self.model.amodel.con_h_expr_0 = cs.vertcat(*[constr[0] for constr in self.nl_con_0])           
        self.ocp.constraints.lh_0 = np.array(cs.vertcat(*[constr[1] for constr in self.nl_con_0]))
        self.ocp.constraints.uh_0 = np.array(cs.vertcat(*[constr[2] for constr in self.nl_con_0]))  
        self.ocp.constraints.idxsh_0 = self.idxhs_0
        self.ocp.cost.zl_0,self.ocp.cost.zu_0,self.ocp.cost.Zl_0,self.ocp.cost.Zu_0 = self.zl_0,self.zu_0,self.Zl_0,self.Zu_0

        self.model.amodel.con_h_expr = cs.vertcat(*[constr[0] for constr in self.nl_con])
        self.ocp.constraints.lh = np.array(cs.vertcat(*[constr[1] for constr in self.nl_con]))  
        self.ocp.constraints.uh = np.array(cs.vertcat(*[constr[2] for constr in self.nl_con]))  
        self.ocp.constraints.idxsh = self.idxhs
        self.ocp.cost.zl,self.ocp.cost.zu,self.ocp.cost.Zl,self.ocp.cost.Zu = self.zl,self.zu,self.Zl,self.Zu
        
        if len(self.nl_con_e) > 0:
            self.model.amodel.con_h_expr_e = cs.vertcat(*[constr[0] for constr in self.nl_con_e])
            self.ocp.constraints.lh_e = np.array(cs.vertcat(*[constr[1] for constr in self.nl_con_e]))
            self.ocp.constraints.uh_e = np.array(cs.vertcat(*[constr[2] for constr in self.nl_con_e]))
            self.ocp.constraints.idxsh_e = self.idxhs_e
            self.ocp.cost.zl_e,self.ocp.cost.zu_e,self.ocp.cost.Zl_e,self.ocp.cost.Zu_e = self.zl_e,self.zu_e,self.Zl_e,self.Zu_e 

        # Solver options
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_dyn = 0   
        self.ocp.solver_options.nlp_solver_type = self.model.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.model.params.solver_mode
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        self.ocp.solver_options.nlp_solver_max_iter = self.model.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.model.params.qp_max_iter
        self.ocp.solver_options.globalization = self.model.params.globalization
        self.ocp.solver_options.levenberg_marquardt = self.model.params.levenberg_marquardt
        self.ocp.solver_options.ext_fun_compile_flags = self.model.params.ext_flag

        self.reset_controller()
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower() + self.model.params.solver_type + self.model.net_name
        gen_name = self.model.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.model.params.build)

        # Reference and fails counter
        self.fails = 0

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

        # Viable state (None for Naive and ST controllers)
        self.x_viable = None

        # Time stats
        self.time_fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
                            'time_glob', 'time_reg', 'time_tot']
        self.last_status = 4

        self.track_traj = self.model.params.track_traj
        if self.track_traj:
            self.a_shape_8 = self.model.params.dim_shape_8
            self.offset_traj = np.array(self.model.params.offset_traj).reshape(3,1)
            a_sym,t_sym = sym.symbols('a t')
            x_sym = (a_sym*sym.cos(t_sym))/(1+sym.sin(t_sym)**2)
            y_sym = (a_sym*sym.cos(t_sym)*sym.sin(t_sym))/(1+sym.sin(t_sym)**2)

            dx=sym.simplify(sym.diff(x_sym,t_sym))
            dy=sym.simplify(sym.diff(y_sym,t_sym))

            self.dx_fun = sym.lambdify(t_sym,dx.subs({a_sym:self.a_shape_8}))
            self.dy_fun = sym.lambdify(t_sym,dy.subs({a_sym:self.a_shape_8})) 

            rot_mat__traj_x = np.array([[1,0,0],
                                        [0,np.cos(self.model.params.theta_rot_traj[0]),-np.sin(self.model.params.theta_rot_traj[0])],
                                        [0,np.sin(self.model.params.theta_rot_traj[0]),np.cos(self.model.params.theta_rot_traj[0])]])

            rot_mat__traj_y = np.array([[np.cos(self.model.params.theta_rot_traj[1]),0,np.sin(self.model.params.theta_rot_traj[1])],
                                        [0,1,0],
                                        [-np.sin(self.model.params.theta_rot_traj[1]),0,np.cos(self.model.params.theta_rot_traj[1])]])

            rot_mat__traj_z = np.array([[np.cos(self.model.params.theta_rot_traj[2]),-np.sin(self.model.params.theta_rot_traj[2]),0],
                                        [np.sin(self.model.params.theta_rot_traj[2]), np.cos(self.model.params.theta_rot_traj[2]),0],
                                        [0,0,1]])
            self.rot_mat__traj = rot_mat__traj_x@rot_mat__traj_y@rot_mat__traj_z

            self.traj_to_track = np.zeros((3,self.N+1))
            self.theta_traj = 0
            self.vel_max_traj = self.model.params.vel_max_traj
            self.acc_traj = self.vel_max_traj/(self.model.params.n_steps*0.9)
            self.vel_traj = self.vel_max_traj if self.model.params.vel_const else 0

            self.generate_trajectory(self.vel_traj)

    def generate_trajectory(self,velocity):
        traj = np.zeros((3,self.N+1))
        theta = self.theta_traj
        self.theta_traj = self.theta_next_from_vel(self.theta_traj,velocity)
        for i in range(self.N+1):
            traj[:,i] = self.get_point_from_theta(theta) 
            theta = self.theta_next_from_vel(theta,velocity)
        self.traj_to_track = self.rot_mat__traj @ traj + self.offset_traj
        if not(self.model.params.vel_const) and (self.vel_traj < self.vel_max_traj):
            self.vel_traj += self.acc_traj

    def get_point_from_theta(self,theta):
        return np.array([(self.a_shape_8*np.cos(theta))/(1+np.sin(theta)**2),(self.a_shape_8*np.cos(theta)*np.sin(theta))/(1+np.sin(theta)**2),0] )
    
    def theta_next_from_vel(self,theta,vel_set):
        return theta+(vel_set/(np.sqrt(self.dx_fun(theta)**2+self.dy_fun(theta)**2)))*self.model.params.dt

    def additionalSetting(self):
        pass

    def terminalSetConstraint(self, soft=True):
        # Get the actual number of nl_constraints --> will be the index for the soft constraint
        num_nl_e = np.sum([c[0].shape[0] for c in self.nl_con_e])
        safe_set_constraints = self.model.safe_set.get_constraints()
        bounds = self.model.safe_set.get_bounds()
        constraint_num = 0
        for i, constr in enumerate(safe_set_constraints):
            self.nl_con_e.append([constr,bounds[i][0],bounds[i][1]])
            constraint_num += constr.shape[0]

        if self.model.params.use_net:
            self.ocp.solver_options.model_external_shared_lib_dir = self.model.safe_set.l4c_model.shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = self.model.safe_set.l4c_model.name

        if soft:
            self.idxhs_e = np.hstack((self.idxhs_e,np.arange(num_nl_e, num_nl_e + constraint_num)))

            self.zl_e = np.hstack((self.zl_e,self.model.params.ws_t*np.ones(self.idxhs_e.size)))
            self.zu_e = np.hstack((self.zu_e,np.zeros(self.idxhs_e.size)))
            self.Zl_e = np.hstack((self.Zl_e,np.zeros(self.idxhs_e.size)))
            self.Zu_e = np.hstack((self.Zu_e,np.zeros(self.idxhs_e.size)))

    def runningSetConstraint(self, soft=True):
        # Suppose that the NN model is already set (same for external model shared lib)
        num_nl = np.sum([c[0].shape[0] for c in self.nl_con])
        safe_set_constraints = self.model.safe_set.get_constraints()
        bounds = self.model.safe_set.get_bounds()
        constraint_num = 0
        for i, constr in enumerate(safe_set_constraints):
            self.nl_con.append([constr,bounds[i][0],bounds[i][1]])
            constraint_num += constr.shape[0]

        if soft:
            self.idxhs = np.hstack((self.idxhs,np.arange(num_nl, num_nl + constraint_num)))

            # Set zl initially to zero, then apply receding constraint in the step method
            self.zl = np.hstack((self.zl,self.model.params.ws_r*np.ones(self.idxhs_e.size)))
            self.zu = np.hstack((self.zu,np.zeros(self.idxhs.size)))
            self.Zl = np.hstack((self.Zl,np.zeros(self.idxhs.size)))
            self.Zu = np.hstack((self.Zu,np.zeros(self.idxhs.size)))

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            
        self.ocp_solver.set(0, 'x', x0)
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])

        if self.track_traj:
            for i in range(self.N+1):
                self.ocp_solver.set(i,'p',np.array([self.traj_to_track[0,i],self.traj_to_track[1,i],self.traj_to_track[2,i],self.model.params.alpha,self.ocp_solver.get(i,'p')[-1]]))
            self.generate_trajectory(self.vel_traj)
        # Solve the OCP
        status = self.ocp_solver.solve()

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.ocp_solver.get(i, "x")
            self.u_temp[i] = self.ocp_solver.get(i, "u")
        self.x_temp[-1] = self.ocp_solver.get(self.N, "x")

        self.last_status = status
        return status

    def provideControl(self):
        """ Save the guess for the next MPC step and return u_opt[0] """
        if self.fails > 0:
            u = self.u_guess[0]
            # Rollback the previous guess
            self.x_guess = np.roll(self.x_guess, -1, axis=0)
            self.u_guess = np.roll(self.u_guess, -1, axis=0)
        else:
            u = self.u_temp[0]
            # Save the current temporary solution
            self.x_guess = np.roll(self.x_temp, -1, axis=0)
            self.u_guess = np.roll(self.u_temp, -1, axis=0)
        # Copy the last values
        self.x_guess[-1] = np.copy(self.x_guess[-2])
        self.u_guess[-1] = np.copy(self.u_guess[-2])
        return u, False

    def step(self, x0):
        pass

    def setReference(self, ee_ref):
        self.model.ee_ref = ee_ref

    def getTime(self):
        return np.array([self.ocp_solver.get_stats(field) for field in self.time_fields])

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def getLastViableState(self):
        return np.copy(self.x_viable)
    
    def resetHorizon(self, N):
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.model.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)

        self.x_temp = np.zeros((self.N + 1, self.model.nx))
        self.u_temp = np.zeros((self.N, self.model.nu))

        if self.track_traj:
            self.generate_trajectory(self.vel_traj)
            self.theta_traj=0
            self.vel_traj = 0 if not(self.model.params.vel_const) else self.vel_max_traj

    def safeGuess(self, x, u, n_safe):
        for i in range(n_safe):
            x, _ = self.model.integrate(x, u[i])
            if not self.model.checkStateConstraints(x) or not self.checkCollision(x):
                return False, None
        return self.model.checkSafeConstraints(x), x
    
    def guessCorrection(self):
        x_sim = np.copy(self.x_guess)
        u_sim = np.copy(self.u_guess)
        for i in range(self.N):
            x_sim[i + 1], u_sim[i] = self.model.integrate(x_sim[i], self.u_guess[i])
        if self.model.checkStateConstraints(x_sim) and np.all([self.checkCollision(x) for x in x_sim]):
            self.x_guess = np.copy(x_sim)
            self.u_guess = np.copy(u_sim)
            return True
        return False
    
    def reset_controller(self):
        self.fails = 0
        self.model.params.use_net = None
        self.model.net_name = ''

class NaiveController(AbstractController):
    def __init__(self, model):
        super().__init__(model)

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.model.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               np.all([self.model.checkCollision(x) for x in self.x_temp])

    def initialize(self, x0, u0=None):
        # Trivial guess
        self.x_guess = np.full((self.N + 1, self.model.nx), x0)
        if u0 is None:
            u0 = np.zeros(self.model.nu)
        self.u_guess = np.full((self.N, self.model.nu), u0)
        # Solve the OCP
        status = self.solve(x0)
        if status == 0 and self.checkGuess():
            self.x_guess = np.copy(self.x_temp)
            self.u_guess = np.copy(self.u_temp)
            return 1
        return 0

    def step(self, x):
        status = self.solve(x)
        if status == 0:
            self.fails = 0
        else:
            self.fails += 1
        return self.provideControl()
    
    
class TerminalZeroVelocity(NaiveController):
    """ Naive MPC with zero terminal velocity as constraint """
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        x_min_e = np.hstack((self.model.x_min[:self.model.nq], np.zeros(self.model.nv)))
        x_max_e = np.hstack((self.model.x_max[:self.model.nq], np.zeros(self.model.nv)))

        self.ocp.constraints.lbx_e = x_min_e
        self.ocp.constraints.ubx_e = x_max_e
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)


class STController(NaiveController):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        self.terminalSetConstraint()

    def reset_controller(self):
        self.fails = 0
        

class STWAController(STController):
    def __init__(self, model):
        super().__init__(model)
        self.x_viable = None

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.model.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               self.model.checkSafeConstraints(self.x_temp[-1]) and \
               np.all([self.model.checkCollision(x) for x in self.x_temp])

    def step(self, x):
        status = self.solve(x)
        if status == 0 and self.model.checkStateConstraints(self.x_temp) and \
                np.all([self.model.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
        else:
            if self.fails == 0:
                self.x_viable = np.copy(self.x_guess[-2])       
            if self.fails == self.N - 1:
                return self.u_guess[0], True
            self.fails += 1
        return self.provideControl()

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess
        self.x_viable = x_guess[-1]


class HTWAController(STWAController):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        self.terminalSetConstraint(soft=False)


class RecedingController(STWAController):
    def __init__(self, model):
        super().__init__(model)
        self.r = self.N
        self.r_last = self.N
        self.abort_flag = self.model.params.abort_flag

    def additionalSetting(self):
        # Terminal constraint before, since it construct the nn model
        self.terminalSetConstraint(soft=True)
        self.runningSetConstraint(soft=False)

    def reset_controller(self):
        super().reset_controller()
        self.r = self.N
        
    def step(self, x):
        # Terminal constraint
        self.ocp_solver.cost_set(self.N, 'zl', self.model.params.ws_t * np.ones((self.zl_e.size,)))
        self.ocp_solver.set(self.N, "p", np.hstack([self.model.ee_ref, [self.model.params.alpha, 1.]]))
        
        # Receding constraint
        if self.r < self.N:
            self.ocp_solver.cost_set(self.r, 'zl', self.model.params.ws_r * np.ones((self.zl.size,)))
            self.ocp_solver.set(self.r, "p", np.hstack([self.model.ee_ref, [self.model.params.alpha, 1.]]))
        for i in range(1,self.N):
            if i != self.r:
                # No constraints on other running states
                self.ocp_solver.cost_set(i,'zl', np.zeros((self.zl.size,)))
                # self.ocp_solver.constraints_set(i, "lh", lh)
                self.ocp_solver.set(i, "p", np.hstack([self.model.ee_ref, [self.model.params.alpha, -1.]]))
        # Solve the OCP
        status = self.solve(x)

        if self.abort_flag:
            self.r -= 1          
        else:
            if self.r > 0:
                self.r -= 1

        if self.r == 0 and self.abort_flag:
            self.x_viable = np.copy(self.x_guess[1])
            # Put back the receding constraint on last state for next iteration after abort
            self.r = self.N
            return self.u_guess[0], True

        if status == 0 and self.model.checkStateConstraints(self.x_temp)  \
                and np.all([self.model.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
            for i in range(self.r + 2, self.N + 1):
                if self.model.checkSafeConstraints(self.x_temp[i]):
                    self.r = i - 1
        else:
            self.fails += 1
            
        return self.provideControl()
    
    def resetHorizon(self, N):
        super().resetHorizon(N)
        self.r=N    

class RealReceding(STWAController):
    def __init__(self, model, obstacles):
        super().__init__(model, obstacles)
        self.r = self.N
        self.x_viable = np.copy(self.x_guess[-1])
        self.abort_flag = self.model.params.abort_flag

    def additionalSetting(self):
        self.terminalSetConstraint()
        
    def step(self, x):
        # Terminal constraint
        self.ocp_solver.cost_set(self.N, "zl", self.model.params.ws_t * np.ones((1,)))
        # Receding constraint --> bound the receding state
        if self.r < self.N:
            self.ocp_solver.constraints_set(self.r, "lbx", self.x_viable)
            self.ocp_solver.constraints_set(self.r, "ubx", self.x_viable)
        for i in range(self.N):
            if i != self.r:
                # No constraints on other running states
                self.ocp_solver.constraints_set(i, "lbx", self.model.x_min) 
                self.ocp_solver.constraints_set(i, "ubx", self.model.x_max)

        # Solve the OCP
        status = self.solve(x)

        if self.abort_flag:
            self.r -= 1          
        else:
            if self.r > 0:
                self.r -= 1

        if self.r == 0 and self.abort_flag:
            self.x_viable = np.copy(self.x_guess[1])
            # Put back the receding constraint on last state for next iteration after abort
            self.r = self.N
            return self.u_guess[0], True
        
        if status == 0 and self.model.checkStateConstraints(self.x_temp)  \
                and np.all([self.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
            for i in range(self.r + 2, self.N + 1):
                if self.model.checkSafeConstraints(self.x_temp[i]):
                    self.r = i - 1
                    self.x_viable = np.copy(self.x_temp[i])
        else:
            self.fails += 1
            
        return self.provideControl()
    
class ParallelController(RecedingController):
    def __init__(self, model):
        super().__init__(model)
        self.r = self.N
        self.constraints = np.linspace(1,self.N,self.N).round().astype(int).tolist()
        self.ocp_solver.set(self.N, "p", np.hstack([self.model.ee_ref, [self.model.params.alpha, 1.]]))

    def additionalSetting(self):
        # Terminal constraint before, since it construct the nn model
        self.terminalSetConstraint(soft=False)
        self.runningSetConstraint(soft=False)

    def constrain_n(self,n_constr):
        self.ocp_solver.cost_set(n_constr, 'zl' , self.model.params.ws_r * np.ones((self.zl.size,)))
        self.ocp_solver.set(n_constr, "p", np.hstack([self.model.ee_ref, [self.model.params.alpha, 1.]]))
        for i in range(1,self.N+1):
            if i != n_constr:
                # No constraints on other running states
                self.ocp_solver.cost_set(i, 'zl' , np.ones((self.zl.size,)))
                self.ocp_solver.set(i, "p", np.hstack([self.model.ee_ref, [self.model.params.alpha, -1.]]))

    def check_safe_n(self):
        r=0
        for i in range(self.r, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                r = i
        return r

    def sing_step(self, x, n_constr):
        success=False
        constr_ver = 0
        self.constrain_n(n_constr)
        # Solve the OCP
        status = self.solve(x)
        checked_r = self.check_safe_n()
        if (status==0) and \
             np.all([self.model.checkCollision(x) for x in self.x_temp]):
        
            constr_ver = checked_r if checked_r >= self.r else self.r

            if (constr_ver-self.r>=0) and\
                self.model.checkStateConstraints(self.x_temp):
                success = True

        return constr_ver if success else 0

    def step(self,x):
        node_success = 0
        for i in reversed(self.constraints):
            result = self.sing_step(x,i)
            if result > node_success:
                node_success = result
                tmp_x = np.copy(self.x_temp)
                tmp_u = np.copy(self.u_temp)
                if result==self.N:
                    break
        if node_success > 1:
            #print(f'Node success:{node_success}')
            #self.core_solution = core
            #self.step_old_solution = 0
            #print(f'rrrrrrrr:{self.r}')
            self.r = node_success
            self.x_temp = tmp_x
            self.u_temp = tmp_u
            self.fails = 0
        else:
            #self.core_solution = None
            #self.step_old_solution +=1
            self.fails += 1
            if self.r ==1:
                self.x_viable = np.copy(self.x_guess[1])
                self.r = self.N
                print("NOT SOLVED")
                #_,self.x_viable = self.simulator.checkSafeIntegrate([x],self.u_guess,self.r)
                if self.model.params.use_net:
                    print(f'is x viable:{self.model.safe_set.nn_func(self.x_viable,self.model.params.alpha)}')
                return self.u_guess[0], True
        self.r -= 1

        return self.provideControl()
    
    def resetHorizon(self, N):
        super().resetHorizon(N)
        self.constraints = np.linspace(1,self.N,self.N).round().astype(int).tolist()
        


class SafeBackupController(AbstractController):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        self.N = 45
        self.ocp.solver_options.tf = self.model.params.dt * self.N
        self.ocp.dims.N = self.N

        # Linear LS cost (zero cost)
        self.ocp.cost.cost_type = 'LINEAR_LS'        
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # self.Q = np.zeros((self.model.nx, self.model.nx))
        # self.R = np.zeros((self.model.nu, self.model.nu))

        # self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        # self.ocp.cost.W_e = self.Q

        # self.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        # self.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        # self.ocp.cost.Vx_e = np.zeros((self.model.nx, self.model.nx))

        # self.ocp.cost.yref = np.zeros(self.model.ny)
        # self.ocp.cost.yref_e = np.zeros(self.model.nx)

        # Terminal constraint --> zero velocity
        q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(self.model.nv)])
        q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(self.model.nv)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Options
        self.ocp.solver_options.ext_fun_compile_flags = '-O3'
        self.ocp.solver_options.levenberg_marquardt = 0.   # Set Default
        self.ocp.solver_options.nlp_solver_max_iter = 20
