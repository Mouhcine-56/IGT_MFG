import math
import numpy as np
import torch

class Obstacle(object):
    """
    Obstacle avoidance optimal control problem (e.g., Section 5.1.3, dim = 2).
    """

    def __init__(self, device):
        self.dim = 10
        self.TT = 1
        self.X0_ub = 1
        self.device = device

        self.c = 6                      # Control penalty
        self.gamma_obst = 5            # Obstacle penalty weight
        self.psi_scale = 1             # Terminal cost scaling
        self.target = [[0.75, 0.5]]    # Target position at final time

        self.ODE_solver = 'RK23'
        self.data_tol = 1e-5
        self.max_nodes = 5000
        self.tseq = np.linspace(0., self.TT, 21)[1:]

    # ===============================
    #       Initial Sampling
    # ===============================

    def sample_x0(self, num_samples):
        """
        Sample initial points outside obstacles.
        """
        valid_samples = []
        max_attempts = num_samples * 10
        attempts = 0

        while len(valid_samples) < num_samples and attempts < max_attempts:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if self.FF_obstacle_func(x, y) <= 0:
                point = [x, y] + [0.0] * (self.dim - 2)
                valid_samples.append(point)
            attempts += 1

        if len(valid_samples) < num_samples:
            print(f"Warning: Only {len(valid_samples)} samples generated out of {num_samples} requested.")

        return torch.tensor(valid_samples, dtype=torch.float32, device=self.device)

    def gen_x0(self, num_samples, Torch=False):
        """
        Generate initial Gaussian-distributed states centered at (-0.75, -0.75).
        """
        mu = np.array([[-0.75, -0.75] + [0] * (self.dim - 2)], dtype=np.float32)
        samples = np.sqrt(0.0025) * np.random.randn(num_samples, self.dim) + mu
        return torch.tensor(samples, dtype=torch.float32, device=self.device) if Torch else samples

    # ===============================
    #        Basic Utilities
    # ===============================

    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)

    def _prod(self, x, y):
        return torch.sum(x * y, dim=1, keepdim=True)

    # ===============================
    #    Obstacle Loss Functions
    # ===============================
        
    def circular_obstacle_torch(self, x, y, center, radius):
        """
        Defines a circular obstacle in PyTorch.
        """
        dist_squared = (x - center[0])**2 + (y - center[1])**2
        return radius**2 - dist_squared  

    def FF_obstacle_func(self, x, y):
        """Non-differentiable combination (used in sampling)."""
        o1 = self.circular_obstacle_torch(x, y, center=[0.1, 0.6], radius=0.5)
        o2 = self.circular_obstacle_torch(x, y, center=[0.1, -0.7], radius=0.7)
        return np.maximum(o1, o2)

    def smooth_max_torch(self, f1, f2, smoothness=50.0):
        """Smooth maximum operator."""
        exp_f1 = torch.exp(smoothness * f1)
        exp_f2 = torch.exp(smoothness * f2)
        return (f1 * exp_f1 + f2 * exp_f2) / (exp_f1 + exp_f2)
        
    def smooth_clamp_min(self, x, smoothness=50.0):
        """Smooth approximation of clamp(x, min=0)."""
        return (1 / smoothness) * torch.log(1 + torch.exp(smoothness * x))
        

    def F_obstacle_func_loss(self, xx_inp, smoothness=50.0, scale=1):
        """
        Differentiable obstacle loss used in Hamiltonian and cost.
        """
        x = xx_inp[:, 0]
        y = xx_inp[:, 1]
        center1 = torch.tensor([0.1, 0.6], device=self.device)
        center2 = torch.tensor([0.1, -0.7], device=self.device)
        radius1 = 0.5
        radius2 = 0.7

        o1 = self.circular_obstacle_torch(x, y, center1, radius1)
        o2 = self.circular_obstacle_torch(x, y, center2, radius2)

        combined = self.smooth_max_torch(o1, o2)
        loss = scale * self.smooth_clamp_min(combined)

        return loss.view(-1, 1)

    def compute_loss_gradients(self, xx_inp):
        """Compute ∇F_obstacle_loss using autograd."""
        xx_inp = xx_inp.clone().detach().requires_grad_(True)
        loss = self.F_obstacle_func_loss(xx_inp)
        loss.backward(torch.ones_like(loss))
        grad_x = xx_inp.grad[:, 0]
        grad_y = xx_inp.grad[:, 1]
        return np.array([grad_x.view(-1).cpu().numpy(), grad_y.view(-1).cpu().numpy()])


    # ===============================
    #      Dynamics & Hamiltonian
    # ===============================

    def ham(self, tt, xx, pp):
        """
        Hamiltonian: H = -c ||p||² + γ · F_obstacle(x)
        """
        return -self.c * self._sqeuc(pp) + self.gamma_obst * self.F_obstacle_func_loss(xx)

    def U_star(self, X_aug):
        """Optimal control u = ∂H/∂p = p"""
        return X_aug[self.dim:2*self.dim]

    def dynamics_torch(self, t, x, V_NN):
        """Closed-loop torch dynamics."""
        U = V_NN.get_grad(t, x)
        return -2 * self.c * U

    def dynamics(self, t, X, U_fun):
        """Closed-loop dynamics used in scipy.solve_ivp."""
        U = U_fun([[t]], X.reshape((1, -1))).flatten()
        return -2 * self.c * U

    # ===============================
    #     Boundary Conditions
    # ===============================

    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]

            target_array = np.array(self.target).squeeze()
            dFdXT = np.zeros_like(AT)
            dFdXT[:2] = 2 * self.psi_scale * (XT[:2] - target_array)

            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc

    # ===============================
    #     Costs and Augmented ODE
    # ===============================

    def terminal_cost(self, X):
        z = X[:2] - np.array(self.target).squeeze()
        return self.psi_scale * np.sum(z * z, axis=0, keepdims=True)
        

    def running_cost(self, t, X, U):
        FF = self.F_obstacle_func_loss(torch.tensor(X.T, dtype=torch.float32, device=self.device))
        return self.c * np.sum(U * U, axis=0, keepdims=True) + self.gamma_obst * FF.cpu().detach().numpy().T

    def aug_dynamics(self, t, X_aug):
        """
        Augmented dynamics:
          - dx/dt = f(x, u)
          - dp/dt = -∂H/∂x
          - dv/dt = -L(x, u)
        """
        U = self.U_star(X_aug)
        x = X_aug[:self.dim]
        Ax = X_aug[self.dim:2*self.dim]

        dxdt = -2 * self.c * U
        dAxdt = np.zeros_like(dxdt)

        dFF = self.compute_loss_gradients(torch.tensor(x.T, dtype=torch.float32, device=self.device))
        dAxdt[:2, :] = -self.gamma_obst * dFF

        L = self.running_cost(t, x, U)
        return np.vstack((dxdt, dAxdt, -L))

    # ===============================
    #       Final Cost Wrapper
    # ===============================
    
    def terminal(self, xx_inp):
        
        z = (xx_inp[:,0:2]-torch.tensor(self.target, device=self.device))
        
        return self.psi_scale * torch.sum(z*z, dim=1, keepdim=True)

    def psi_func(self, xx_inp):
        """Terminal cost wrapper for NN compatibility."""
        return self.terminal(xx_inp)

