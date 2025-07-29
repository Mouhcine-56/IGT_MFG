import math
import numpy as np
import torch

class Analytic(object):
    """
    Example 4.2.
    """
    def __init__(self, device):
        
        self.dim = 2
        self.TT = 1
        self.X0_ub = 1
        self.device = device
        self.TT = 1
        self.X0_ub = 1
        self.c = 6
        self.gamma_obst = 5
        self.psi_scale = 1
        self.target = [[0.75, 0.5]]
        self.ODE_solver = 'RK23'
        self.data_tol = 1e-05
        self.max_nodes = 5000
        self.tseq = np.linspace(0., self.TT, 20+1)[1:]



    
    def sample_x0(self, num_samples):

        valid_samples = []
        max_attempts = num_samples * 10  # Avoid infinite loops, limit attempts
        attempts = 0

        while len(valid_samples) < num_samples and attempts < max_attempts:
            # Generate a random sample in [-1,1] x [-1,1] - obstacle 
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)


            # Reject points inside the obstacle (out < 0 means inside)
            if self.FF_obstacle_func(x, y) <= 0:  # Accept only if outside the obstacle
                point = [x, y] + [0.0] * (self.dim - 2)
                valid_samples.append(point)

            attempts += 1

        valid_samples = np.array(valid_samples)

        if len(valid_samples) < num_samples:
            print(f"Warning: Only {len(valid_samples)} samples generated out of {num_samples} requested.")

        return torch.tensor(valid_samples, dtype=torch.float32, device=self.device)
    
    def gen_x0(self, num_samples, Torch=False):

        mu = np.array([[-0.75, -0.75] + [0] * (self.dim - 2)], dtype=np.float32)
        samples = np.sqrt(0.0025) * np.random.randn(num_samples, self.dim) + mu

        if Torch:
            return torch.tensor(samples, dtype=torch.float32, device=self.device)
        else:
            return samples

    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)
    
    def _prod(self, x, y):
        return torch.sum(x * y, dim=1, keepdim=True)
    
    def circular_obstacle_torch(self, x, y, center, radius):
        """
        Defines a circular obstacle in PyTorch.
        """
        dist_squared = (x - center[0])**2 + (y - center[1])**2
        return radius**2 - dist_squared  
    
    def FF_obstacle_func(self, x, y):
        """
        Combine two circular obstacles with smooth maximum.
        """
        obstacle1 = self.circular_obstacle_torch(x, y, center=[0.1, 0.6], radius=0.5)
        obstacle2 = self.circular_obstacle_torch(x, y, center=[0.1, -0.7], radius=0.7)
        return np.maximum(obstacle1, obstacle2)


#     def smooth_max_torch(self, f1, f2, smoothness=100.0):
#         """
#         Smooth max in PyTorch (differentiable).
#         """
#         return (1 / smoothness) * torch.log(torch.exp(smoothness * f1) + torch.exp(smoothness * f2))

    def smooth_max_torch(self, f1, f2, smoothness=50.0):

        exp_f1 = torch.exp(smoothness * f1)
        exp_f2 = torch.exp(smoothness * f2)
        return (f1 * exp_f1 + f2 * exp_f2) / (exp_f1 + exp_f2)
    
    def smooth_clamp_min(self, x, smoothness=50.0):
        """
        Smooth approximation of torch.clamp_min(x, 0).

        Parameters:
        - x: Input tensor.
        - smoothness: Controls the sharpness of the transition (higher is sharper).
        """
        return (1 / smoothness) * torch.log(1 + torch.exp(smoothness * x))

    def F_obstacle_func_loss(self, xx_inp, smoothness=50.0, scale=1):
        """
        Computes the differentiable obstacle loss for a batch of input points.
        """
        # Extract x and y coordinates from the input tensor
        x = xx_inp[:, 0]
        y = xx_inp[:, 1]

        # Define obstacle 1 centers and radii
#         center1 = torch.tensor([-0.5, 0.5], dtype=torch.float, device=self.device)
#         radius1 = torch.tensor(1 / 3, dtype=torch.float, device=self.device)
#         center2 = torch.tensor([2 / 7, -2 / 7], dtype=torch.float, device=self.device)
#         radius2 = torch.tensor(2 / 3, dtype=torch.float, device=self.device)
        

        # Define circular obstacles 3       
        center1 = torch.tensor([0.1, 0.6], dtype=torch.float, device=self.device)
        radius1 = torch.tensor(0.5, dtype=torch.float, device=self.device)
        center2 = torch.tensor([0.1, -0.7], dtype=torch.float, device=self.device)
        radius2 = torch.tensor(0.7, dtype=torch.float, device=self.device)

        # Compute obstacle fields
        obstacle1 = self.circular_obstacle_torch(x, y, center1, radius1)
        obstacle2 = self.circular_obstacle_torch(x, y, center2, radius2)

        # Combine obstacle fields with smooth max (fully differentiable)
        combined_obstacle = self.smooth_max_torch(obstacle1, obstacle2)

        # Scale the result and clamp minimum value to zero (differentiable)
        loss = scale * self.smooth_clamp_min(combined_obstacle)

        return loss.view(-1,1) 

#     def F_obstacle_func_loss(self, xx_inp, scale=1):
#         """
#         Calculate interaction term. Calculates F(x), where F is the forcing term in the HJB equation.
#         """
#         batch_size = xx_inp.size(0)
#         xx = xx_inp[:, 0:2]
#         dim = xx.size(1)
#         assert (dim == 2), f"Require dim=2 but, got dim={dim} (BAD)"

#         # Two diagonal obstacles
#         # Rotation matrix
#         theta = torch.tensor(np.pi / 5)
#         rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
#                                 [torch.sin(theta), torch.cos(theta)]]).expand(batch_size, dim, dim).to(self.device)

#         # Bottom/Left obstacle  # TODO: Clean it up
#         center1 = torch.tensor([-2, 0.5], dtype=torch.float).to(self.device)
#         xxcent1 = xx - center1
#         xxcent1 = xxcent1.unsqueeze(1).bmm(rot_mat).squeeze(1)
#         covar_mat1 = torch.eye(dim, dtype=torch.float)
#         covar_mat1[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
#         covar_mat1 = covar_mat1.expand(batch_size, dim, dim).to(self.device)
#         bb_vec1 = torch.tensor([0, 2], dtype=torch.float).expand(xx.size()).to(self.device)
#         xxcov1 = xxcent1.unsqueeze(1).bmm(covar_mat1)
#         quad1 = torch.bmm(xxcov1, xxcent1.unsqueeze(2)).view(-1, 1)
#         lin1 = torch.sum(xxcent1 * bb_vec1, dim=1, keepdim=True)
#         out1 = (-1) * ((quad1 + lin1) + 1)
#         out1 = scale * out1.view(-1, 1)
#         out1 = torch.clamp_min(out1, min=0)

#         # Top/Right obstacle
#         center2 = torch.tensor([2, -0.5], dtype=torch.float).to(self.device)
#         xxcent2 = xx - center2
#         xxcent2 = xxcent2.unsqueeze(1).bmm(rot_mat).squeeze(1)
#         covar_mat2 = torch.eye(dim, dtype=torch.float)
#         covar_mat2[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
#         covar_mat2 = covar_mat2.expand(batch_size, dim, dim).to(self.device)
#         bb_vec2 = torch.tensor([0, -2], dtype=torch.float).expand(xx.size()).to(self.device)
#         xxcov2 = xxcent2.unsqueeze(1).bmm(covar_mat2)
#         quad2 = torch.bmm(xxcov2, xxcent2.unsqueeze(2)).view(-1, 1)
#         lin2 = torch.sum(xxcent2 * bb_vec2, dim=1, keepdim=True)
#         out2 = (-1) * ((quad2 + lin2) + 1)
#         out2 = scale * out2.view(-1, 1)
#         out2 = torch.clamp_min(out2, min=0)

#         out = out1 + out2

#         return out
    
#     def FF_obstacle_func_loss_derivatives(self, xx_inp, smoothness=50.0, scale=1.0):
#         """
#         Computes the derivatives of the obstacle loss w.r.t. x1 and x2.
#         """
#         # Extract x and y coordinates
#         x = xx_inp[:, 0]
#         y = xx_inp[:, 1]

#         # Define obstacle centers and radii
#         center1 = torch.tensor([-0.5, 0.5], dtype=torch.float, device=self.device)
#         radius1 = torch.tensor(1 / 3, dtype=torch.float, device=self.device)
#         center2 = torch.tensor([2 / 7, -2 / 7], dtype=torch.float, device=self.device)
#         radius2 = torch.tensor(2 / 3, dtype=torch.float, device=self.device)

#         # Compute obstacle fields
#         f1 = self.circular_obstacle_torch(x, y, center1, radius1)
#         f2 = self.circular_obstacle_torch(x, y, center2, radius2)
        
#         # Cap f1 and f2 to prevent extreme values



#         # Compute gradients of f1 and f2
#         df1_dx1 = -2 * (x - center1[0])
#         df1_dx2 = -2 * (y - center1[1])
#         df2_dx1 = -2 * (x - center2[0])
#         df2_dx2 = -2 * (y - center2[1])

#         # Smooth max weight terms
#         exp_f1 = torch.exp(smoothness * f1)
#         exp_f2 = torch.exp(smoothness * f2)
#         denominator = exp_f1 + exp_f2 


#         # Combined obstacle gradient w.r.t x1 and x2
#         d_combined_dx1 = (smoothness * exp_f1 * df1_dx1 + smoothness * exp_f2 * df2_dx1) / denominator
#         d_combined_dx2 = (smoothness * exp_f1 * df1_dx2 + smoothness * exp_f2 * df2_dx2) / denominator

#         # Compute derivatives of the loss
#         combined_obstacle = self.smooth_max_torch(f1, f2, smoothness)
#         loss_gradient_x1 = torch.where(combined_obstacle > 0, scale * d_combined_dx1, torch.zeros_like(d_combined_dx1))
#         loss_gradient_x2 = torch.where(combined_obstacle > 0, scale * d_combined_dx2, torch.zeros_like(d_combined_dx2))


#         return np.array([loss_gradient_x1.view(-1).cpu().numpy(),loss_gradient_x2.view(-1).cpu().numpy()])


    def compute_loss_gradients(self, xx_inp):

        # Ensure the input tensor requires gradient
        xx_inp = xx_inp.clone().detach().requires_grad_(True)

        # Compute the loss
        loss = self.F_obstacle_func_loss(xx_inp)

        # Compute gradients using autograd
        loss.backward(torch.ones_like(loss))

        # Extract gradients for x and y
        loss_gradient_x1 = xx_inp.grad[:, 0]  # Gradient w.r.t. x
        loss_gradient_x2 = xx_inp.grad[:, 1]  # Gradient w.r.t. y
        
        #print(np.array([loss_gradient_x1.view(-1).cpu().numpy(), loss_gradient_x2.view(-1).cpu().numpy()]))
        #print(stp)

        # Convert gradients to NumPy arrays and return them
       
        return np.array([loss_gradient_x1.view(-1).cpu().numpy(), loss_gradient_x2.view(-1).cpu().numpy()])

    def ham(self, tt, xx, pp):
        
        """ The Hamiltonian."""

        out = -self.c * self._sqeuc(pp)  + self.gamma_obst*self.F_obstacle_func_loss(xx)

        return out
    
    def U_star(self, X_aug):
        
        '''Control as a function of the costate.'''
        Ax = X_aug[self.dim:2*self.dim]
        U =  Ax
        return U
    
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            # Extract components
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]

            # Debugging: Print shapes
            #print(f"X0 shape: {X0.shape}, XT shape: {XT.shape}, AT shape: {AT.shape}, vT shape: {vT.shape}")

            # Assertions to validate shapes
            assert X0.shape == (self.dim,), f"X0 shape mismatch: {X0.shape} vs {(self.dim,)}"
            assert XT.shape == (self.dim,), f"XT shape mismatch: {XT.shape} vs {(self.dim,)}"
            assert AT.shape == (self.dim,), f"AT shape mismatch: {AT.shape} vs {(self.dim,)}"
            assert vT.shape == (1,), f"vT shape mismatch: {vT.shape} vs (1,)"

            # Derivative of the terminal cost with respect to the final state
            target_array = np.array(self.target).squeeze()
            #assert target_array.shape == (self.dim,), f"Target shape mismatch: {target_array.shape} vs {(self.dim,)}"
            dFdXT = np.zeros_like(AT)
            dFdXT[:2] = 2 * self.psi_scale * (XT[:2] - target_array)

            # Compute boundary condition residuals
            residuals = np.concatenate((X0 - X0_in, AT - dFdXT, vT))
            #print("Boundary condition residuals:", residuals)

            return residuals
        return bc
    
    def dynamics_torch(self, t, x, V_NN):
        
        '''Evaluation of the dynamics at a single time instance for closed-loop ODE integration.'''
        
        U =  V_NN.get_grad(t, x)

        return - 2 * self.c * U
    
    def dynamics(self, t, X, U_fun):
        
        '''Evaluation of the dynamics at a single time instance for closed-loop ODE integration.'''
        
        U = U_fun([[t]], X.reshape((1,-1))).flatten()

        return  -2 * self.c * U
    
    def terminal_cost(self, X):

        z = (X[:2]- np.array(self.target).squeeze())

        return self.psi_scale*np.sum(z * z, axis=0, keepdims=True)

    
    
    def running_cost(self,t, X, U):
        
        FF = self.F_obstacle_func_loss(torch.tensor(X.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        
        return self.c * np.sum(U * U, axis=0, keepdims=True) +  self.gamma_obst * FF.T

    
    def aug_dynamics(self, t, X_aug):
        
        '''Evaluation of the augmented dynamics at a vector of time instances for solution of the two-point BVP.'''
        
        U = self.U_star(X_aug)
       
        x = X_aug[:self.dim]

        
        dFF =  self.compute_loss_gradients(torch.tensor(x.T, dtype=torch.float32, device=self.device))

         
        Ax = X_aug[self.dim:2*self.dim]

        
        dxdt =  -2 * self.c * U
        
        dAxdt = np.zeros_like(dxdt)
        
        
        dAxdt[0:2,:] = - self.gamma_obst * dFF
        

        L = self.running_cost(t, x, U)

       
        fun = np.vstack((dxdt, dAxdt, -L))


        return fun
    
    



    def get_trace(self, grad, xx, batch_size, dim, grad_outputs_vec):
        """
        Computation of the second-order term in the HJB equation.
        """
        hess_stripes = torch.autograd.grad(outputs=grad, inputs=xx,
                                           grad_outputs=grad_outputs_vec,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]
        pre_laplacian = torch.stack([hess_stripes[i * batch_size: (i + 1) * batch_size, i]
                                     for i in range(0, dim)], dim=1)
        laplacian = torch.sum(pre_laplacian, dim=1)
        laplacian_sum_repeat = laplacian.repeat((1, dim))

        return laplacian_sum_repeat.T
    
    def terminal(self, xx_inp):
        
        z = (xx_inp[:,0:2]-torch.tensor(self.target, device=self.device))
        
        return self.psi_scale * torch.sum(z*z, dim=1, keepdim=True)
    
    

    def psi_func(self, xx_inp):
        """
        The final-time cost function.
        """
        
        return self.terminal( xx_inp)