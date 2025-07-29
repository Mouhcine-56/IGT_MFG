import math
import numpy as np
import torch

class Analytic(object):
    """
    Example 4.2.
    """
    def __init__(self, G_NN_list, Round, n, x0_initial, device, VV):
        
        self.dim = 1
        self.TT = 1
        self.X0_ub = 2
        self.ODE_solver = 'RK23'
        self.data_tol = 1e-07
        self.max_nodes = 5000
        self.tseq = np.linspace(0., self.TT, 20+1)[1:]
        self.X0_lb = - self.X0_ub
        self.device = device
        self.x0_initial = x0_initial
        self.mean_0 = torch.mean(self.x0_initial) *0 + 1
        self.G_NN_list = G_NN_list
        self.Round = Round
        self.n = n
        self.sigma = np.sqrt(0.105)
        self.mu = 0.1
        self.VV = VV
       

    def sample_x0(self, num_samples):
        """
        The initial distribution rho_0 of the agents.
        """
        X0 = torch.rand(num_samples, self.dim, device=self.device)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb
       
        return X0
        

    def gen_x0(self, num_samples, Torch=False):

        samples = np.random.normal(loc=self.mu, scale=self.sigma, size=(num_samples, self.dim))

        if Torch:
            return torch.tensor(samples, dtype=torch.float32, device=self.device)
        else:
            return samples
        
    def update_mean(self, t, x):
        
        Means = []
        t_expanded = t.repeat_interleave(self.x0_initial.shape[0]).view(-1,1)
        x0_expanded = self.x0_initial.repeat(t.shape[0],1)
        
        if self.VV == 1:
            if self.Round == 0:
                Means.append(self.mean_0)
                if self.n == 0:
                    return Means[self.n]
                else:
                    for i in range(1, self.n + 1):
                        with torch.no_grad(): 
                            mean_new = torch.mean(
                                self.G_NN_list[i-1](t_expanded, x0_expanded).reshape(t.shape[0], self.x0_initial.shape[0]),
                                dim=1
                            )

                        Means.append((1 / (i + 1)) * mean_new + (i / (i + 1)) * Means[i-1])

                    return Means[self.n]
                
            else:
                with torch.no_grad(): 
                        mean_new = torch.mean(
                                self.G_NN_list[0](t_expanded, x0_expanded).reshape(t.shape[0], self.x0_initial.shape[0]),
                                dim=1
                            )
                        Means.append(mean_new)
                if self.n == 0: 
                    return Means[self.n]
                else:
                    for i in range(1, self.n + 1):
                        with torch.no_grad(): 
                            mean_new = torch.mean(
                                self.G_NN_list[i](t_expanded, x0_expanded).reshape(t.shape[0], self.x0_initial.shape[0]),
                                dim=1
                            )


                        Means.append((1 / (i + 1)) * mean_new + (i / (i + 1)) * Means[i-1])

                    return Means[self.n]
        else:
            with torch.no_grad(): 
                        mean_new = torch.mean(
                            self.G_NN_list[-1](t_expanded, x0_expanded).reshape(t.shape[0], self.x0_initial.shape[0]),
                            dim=1
                        )
            return mean_new
                    
            
        


    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)
    
    def _prod(self, x, y):
        return torch.sum(x * y, dim=1, keepdim=True)
    
    def F(self, t, x):
        
        up_mean = self.update_mean(t, x)
        
        return 0.5 * (x-up_mean.view(-1,1))**2
        


    def d_F(self, t, x):
        
        up_mean = self.update_mean(t, x)
        
        return x-up_mean.view(-1,1)
    
    

    def ham(self, tt, xx, pp):
        
        """ The Hamiltonian."""

        out = -0.5 * self._sqeuc(pp) + self.F( tt, xx)

        return out
    
    def U_star(self, X_aug):
        
        '''Control as a function of the costate.'''
        Ax = X_aug[self.dim:2*self.dim]        
        U =  -Ax
        return U
    
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]

            # Derivative of the terminal cost with respect to X(T)
            dFdXT = 0 

            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc
    
    def dynamics_torch(self, t, x, V_NN):
        
        '''Evaluation of the dynamics at a single time instance for closed-loop ODE integration.'''
        
        U = - V_NN.get_grad(t, x)

        return U
    
    def dynamics(self, t, X, U_fun):
        
        '''Evaluation of the dynamics at a single time instance for closed-loop ODE integration.'''
        
        U = U_fun([[t]], X.reshape((1,-1))).flatten()

        return U
    
    def terminal_cost(self, X):

        return  0
    
    def running_cost(self,t, X, U):
        
        FF = self.F(torch.tensor(t, dtype=torch.float32, device=self.device),torch.tensor(X.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
   
        return 0.5 * np.sum(U * U, axis=0, keepdims=True) +  FF.T

    
    def aug_dynamics(self, t, X_aug):
        
        '''Evaluation of the augmented dynamics at a vector of time instances for solution of the two-point BVP.'''
        
        U = self.U_star(X_aug)

        x = X_aug[:self.dim]
        
        dFF = self.d_F(torch.tensor(t, dtype=torch.float32, device=self.device),torch.tensor(x.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        
        # Costate
        Ax = X_aug[self.dim:2*self.dim]
        
        dxdt =  U
        
        dAxdt = - dFF.T

        L = self.running_cost(t, x, U)
       
        fun = np.vstack((dxdt, dAxdt, -L))


        return fun

    
    
