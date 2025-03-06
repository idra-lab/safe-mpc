import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
import casadi as cs
import l4casadi as l4c

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU()):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )
    
class AbstractSafeSet():
    def __init__(self,model):
        self.model = model
        self.constraints = []
        self.constraints_fun = []
        self.bounds = []
    
    def get_constraints(self):
        return self.constraints
    
    def get_constraints_fun(self):
        return self.constraints_fun
    
    def get_bounds(self):
        return self.bounds

    def check_constraint(self,x):
        if self.constraints == None:
            return True
        for i,constraint in enumerate(self.constraints_fun):
            constraint_val =constraint(x)
            if not(np.array(np.multiply(((self.bounds[i][0]-self.model.params.tol_safe_set)<=constraint_val),(constraint_val<=(self.bounds[i][1]+self.model.params.tol_safe_set)))).all()):  
                return False
        return True
    
class NetSafeSet(AbstractSafeSet):
    def __init__(self,model):
        super().__init__(model)

        # constraints  bounded
        self.constraint_bound = 'zl'

        act_fun =nn.GELU(approximate='tanh')

        model_net = NeuralNetwork(8, 256, 1, act_fun)
        nn_data = torch.load(f'{self.model.params.NN_DIR}4dof_gelu_obs.pt',
                             map_location=torch.device('cpu'))
        model_net.load_state_dict(nn_data['model'])

        n_dof = self.model.params.nq
        
        x_cp = deepcopy(self.model.x)
        x_cp[n_dof] += self.model.params.eps
        vel_norm = cs.norm_2(x_cp[n_dof:n_dof + self.model.params.n_dof_safe_set])
        pos = (x_cp[:self.model.params.n_dof_safe_set] - nn_data['mean']) / nn_data['std']
        vel_dir = x_cp[n_dof:n_dof + self.model.params.n_dof_safe_set] / vel_norm
        state = cs.vertcat(pos, vel_dir)

        self.l4c_model = l4c.L4CasADi(model_net.linear_stack,
                                      device='cpu',
                                      name=f'{self.model.params.urdf_name}_model',
                                      build_dir=f'{self.model.params.GEN_DIR}nn_{self.model.params.urdf_name}')
        self.nn_model = cs.if_else(self.model.p[4] > 0, 
                                self.l4c_model(state) * (100 - self.model.p[3]) / 100 - vel_norm, 
                                1., True)
        
        self.constraints.append(self.nn_model)
        
        self.nn_func = cs.Function('nn_func', [self.model.x, self.model.p], [self.nn_model])
        
        nn_model_alpha_fixed = self.l4c_model(state) * (100 - self.model.params.alpha) / 100 - vel_norm
        self.nn_func_x = cs.Function('nn_func_x', [self.model.x], [nn_model_alpha_fixed])
        self.constraints_fun.append(self.nn_func_x)

        self.bounds.append([0,1e6])

class AnalyticSafeSet(AbstractSafeSet):
    # possibilità di aggiungere nymero di gradi safe_set e non necessariamente nq gradi di libertà
    def __init__(self,model):
        super().__init__(model)
        self.reg_term = 1e-6 
        self.n_dof = self.model.params.nq
        
        # constraints upper bounded
        self.constraint_bound = 'zu'

        for i,obs in enumerate(self.model.params.obstacles):
            if obs['name'] == 'floor':
                self.constraints.append(self.floor_con(obs))
                self.bounds.append([-1e6,0])
                self.constraints_fun.append(cs.Function(f'obs_{i}',[self.model.x],[self.constraints[-1]]))
                self.constraints[-1] = self.casadi_if_else(self.constraints[-1],self.bounds[-1])

            elif obs['name'] == 'ball':
                self.constraints.append(self.ball_con(obs))
                self.bounds.append([-1e6,0])    
                self.constraints_fun.append(cs.Function(f'obs_{i}',[self.model.x],[self.constraints[-1]]))
                self.constraints[-1] = self.casadi_if_else(self.constraints[-1],self.bounds[-1])

        self.constraints.append(self.ddq_min_expr())
        self.bounds.append([-1e6*np.ones(self.n_dof),np.sqrt(2*self.model.params.ddq_max[:self.n_dof])])
        self.constraints_fun.append(cs.Function('ddq_min',[self.model.x],[self.constraints[-1]]))
        self.constraints[-1] = (self.casadi_if_else(self.constraints[-1],self.bounds[-1]))

        self.constraints.append(self.ddq_max_expr())
        self.bounds.append([-1e6*np.ones(self.n_dof),np.sqrt(2*self.model.params.ddq_max[:self.n_dof])])
        self.constraints_fun.append(cs.Function('ddq_max',[self.model.x],[self.constraints[-1]]))
        self.constraints[-1] = (self.casadi_if_else(self.constraints[-1],self.bounds[-1]))




    def ddq_min_expr(self):
        return -self.model.x[self.n_dof:]/ \
                cs.sqrt(self.model.x[:self.n_dof]-self.model.x_min[:self.n_dof]+self.reg_term)

    def ddq_max_expr(self):
        return self.model.x[self.n_dof:]/ \
                cs.sqrt(self.model.x_max[:self.n_dof] - self.model.x[:self.n_dof]+self.reg_term)

    def floor_con(self,obs):
        distance_fl = obs['position'][2] - self.model.t_glob[2]
        dx_max_fl = cs.sqrt(2*self.model.params.ddx_max[2]*cs.fabs(distance_fl+1e-6))   #  [i['axis']]
        floor_expr = (((self.model.jac(np.eye(4),self.model.x[:self.n_dof])[:3,6:]@self.model.x[self.n_dof:])[2])*cs.sign(distance_fl)) - dx_max_fl
        return floor_expr
    
    def ball_con(self,obs):
        dist_vec_ball = (obs['position'])-self.model.t_glob
        dx_max_ball = cs.sqrt(cs.dot(2*self.model.params.ddx_max,cs.fabs(dist_vec_ball+1e-6)))
        ball_expr = cs.dot((self.model.jac(np.eye(4),self.model.x[:self.n_dof])[:3,6:]@self.model.x[self.n_dof:]),dist_vec_ball/cs.norm_2(dist_vec_ball)) - dx_max_ball  #-cs.fabs(cs.dot(dx_max_ball,dist_vec_ball))
        return ball_expr
    
    def casadi_if_else(self,expression,bounds):
        return(cs.if_else(self.model.p[4] > 0, 
                          expression, 
                          (bounds[0] + bounds[1])/2, True))