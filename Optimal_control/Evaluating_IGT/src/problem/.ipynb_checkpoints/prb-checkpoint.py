import math
import numpy as np
import torch

class Analytic(object):
    """
    Analytic problem setup for evaluting IGT section 5.1.1:.
    Defines dynamics, costs, Hamiltonian,  and value function.
    """
    def __init__(self, device):
        self.dim = 2                   # Dimension of state space
        self.TT = 1                    # Final time
        self.X0_ub = 1                 # Upper bound for initial state sampling
        self.X0_lb = -self.X0_ub       # Lower bound for initial state sampling
        self.ODE_solver = 'RK23'      # ODE solver for closed-loop simulation
        self.data_tol = 1e-8          # Tolerance for BVP solver
        self.max_nodes = 5000         # Max nodes in BVP solver
        self.tseq = np.linspace(0., self.TT, 21)[1:]  # Exclude t=0
        self.device = device

    # ==================================
    #       Sampling Functions
    # ==================================

    def sample_x0(self, num_samples):
        """Sample torch tensors from uniform initial distribution."""
        X0 = torch.rand(num_samples, self.dim, device=self.device)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb
        return X0
        

    def gen_x0(self, num_samples, Torch=False):
        """Generate numpy array of initial points (for BVP)."""
        X0 = np.random.rand(num_samples, self.dim)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb
        
        if Torch:
            return torch.tensor(X0, dtype=torch.float32, device=self.device)
        else:
            return X0

    # ==================================
    #        Cost and Dynamics
    # ==================================

    def _sqeuc(self, x):
        """Squared Euclidean norm."""
        return torch.sum(x * x, dim=1, keepdim=True)

    def _prod(self, x, y):
        """Dot product."""
        return torch.sum(x * y, dim=1, keepdim=True)

    def ham(self, tt, xx, pp):
        """Hamiltonian H = -1/4 ||p||² + <p, x>."""
        return -0.25 * self._sqeuc(pp) + self._prod(pp, xx)

    def U_star(self, X_aug):
        """Optimal control: u = -1/2 * p."""
        Ax = X_aug[self.dim:2*self.dim]
        return -0.5 * Ax

    def dynamics(self, t, X, U_fun):
        """
        Closed-loop system dynamics: dx/dt = x + u(t, x)
        """
        U = U_fun([[t]], X.reshape((1, -1))).flatten()
        return X + U

    def aug_dynamics(self, t, X_aug):
        """
        Augmented dynamics for BVP:
            dx/dt = x + u
            dp/dt = -∂H/∂x = -p
            dv/dt = -L(x, u)
        """
        U = self.U_star(X_aug)
        x = X_aug[:self.dim]
        Ax = X_aug[self.dim:2*self.dim]
        dxdt = x + U
        dAxdt = -Ax
        L = self.running_cost(x, U)
        return np.vstack((dxdt, dAxdt, -L))

    # ==================================
    #         Boundary Conditions
    # ==================================

    def make_bc(self, X0_in):
        """
        Boundary condition: x(0) = X0_in, p(T) = ∂Φ/∂x, v(T) = -Φ(x(T))
        """
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]

            dFdXT = 2 * XT  # ∂Φ/∂x = 2x

            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc

    # ==================================
    #         Cost Functions
    # ==================================

    def terminal_cost(self, X):
        """Terminal cost Φ(x) = ||x||²."""
        return np.sum(X.reshape(-1, 1)**2, axis=0, keepdims=True)

    def running_cost(self, X, U):
        """Running cost L(x, u) = ||u||²."""
        return np.sum(U**2, axis=0, keepdims=True)

    def psi_func(self, xx_inp):
        """Terminal value function ψ(x) = ||x||²."""
        return self._sqeuc(xx_inp)

    def V_exact(self, x, t):
        """
        Exact solution: V(x, t) = 2||x||² / (1 + exp(2(t-1)))
        """
        return 2 * self._sqeuc(x) / (1 + math.exp(2 * (t - 1)))
