import math
import numpy as np
import torch

class Analytic(object):
    """
    Evaluating IGT-MFG 5.2.1 — Analytic formulation.
    Includes value function cost, generator mean tracking, dynamics, and Hamiltonian.
    """

    def __init__(self, G_NN_list, Round, n, x0_initial, device, VV):
        self.dim = 1
        self.TT = 1
        self.X0_ub = 2
        self.X0_lb = -self.X0_ub
        self.device = device

        self.ODE_solver = 'RK23'
        self.data_tol = 1e-7
        self.max_nodes = 5000
        self.tseq = np.linspace(0., self.TT, 21)[1:]

        self.G_NN_list = G_NN_list
        self.Round = Round
        self.n = n
        self.VV = VV

        self.x0_initial = x0_initial
        self.mean_0 = torch.mean(self.x0_initial) * 0 + 1
        self.sigma = np.sqrt(0.105)
        self.mu = 0.1

    # ========================
    #     Sampling Methods
    # ========================
    def sample_x0(self, num_samples):
        """Uniform initial distribution in [-2, 2]."""
        X0 = torch.rand(num_samples, self.dim, device=self.device)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb
        return X0

    def gen_x0(self, num_samples, Torch=False):
        """Gaussian initial distribution centered at mu."""
        samples = np.random.normal(loc=self.mu, scale=self.sigma, size=(num_samples, self.dim))
        if Torch:
            return torch.tensor(samples, dtype=torch.float32, device=self.device)
        return samples

    # ========================
    #     Population Mean
    # ========================
    def update_mean(self, t, x):
        """
        Computes empirical mean trajectory based on generator history.
        Handles both Round 0 and later rounds differently.
        """
        Means = []
        t_expanded = t.repeat_interleave(self.x0_initial.shape[0]).view(-1, 1)
        x0_expanded = self.x0_initial.repeat(t.shape[0], 1)

        if self.VV == 1:
            if self.Round == 0:
                Means.append(self.mean_0)
                if self.n == 0:
                    return Means[0]
                for i in range(1, self.n + 1):
                    with torch.no_grad():
                        mean_i = torch.mean(
                            self.G_NN_list[i-1](t_expanded, x0_expanded).reshape(t.shape[0], -1),
                            dim=1
                        )
                    avg = (1 / (i + 1)) * mean_i + (i / (i + 1)) * Means[i - 1]
                    Means.append(avg)
                return Means[self.n]

            else:
                with torch.no_grad():
                    mean_0 = torch.mean(
                        self.G_NN_list[0](t_expanded, x0_expanded).reshape(t.shape[0], -1),
                        dim=1
                    )
                Means.append(mean_0)
                if self.n == 0:
                    return Means[0]
                for i in range(1, self.n + 1):
                    with torch.no_grad():
                        mean_i = torch.mean(
                            self.G_NN_list[i](t_expanded, x0_expanded).reshape(t.shape[0], -1),
                            dim=1
                        )
                    avg = (1 / (i + 1)) * mean_i + (i / (i + 1)) * Means[i - 1]
                    Means.append(avg)
                return Means[self.n]

        else:
            with torch.no_grad():
                mean_final = torch.mean(
                    self.G_NN_list[-1](t_expanded, x0_expanded).reshape(t.shape[0], -1),
                    dim=1
                )
            return mean_final

    # ========================
    #      Cost Functions
    # ========================
    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)

    def _prod(self, x, y):
        return torch.sum(x * y, dim=1, keepdim=True)

    def F(self, t, x):
        """Population interaction cost."""
        mu_t = self.update_mean(t, x)
        return 0.5 * (x - mu_t.view(-1, 1)) ** 2

    def d_F(self, t, x):
        """Gradient of interaction cost."""
        mu_t = self.update_mean(t, x)
        return x - mu_t.view(-1, 1)

    # ========================
    #     Hamiltonian & BCs
    # ========================
    def ham(self, tt, xx, pp):
        """Hamiltonian H = -1/2 ||p||² + F(x, t)."""
        return -0.5 * self._sqeuc(pp) + self.F(tt, xx)

    def U_star(self, X_aug):
        """Control u = -p."""
        Ax = X_aug[self.dim:2*self.dim]
        return -Ax

    def make_bc(self, X0_in):
        """Boundary conditions for BVP."""
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]
            dFdXT = 0  # terminal gradient is 0
            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc

    # ========================
    #       Dynamics
    # ========================
    def dynamics_torch(self, t, x, V_NN):
        """Torch version of closed-loop dynamics."""
        return -V_NN.get_grad(t, x)

    def dynamics(self, t, X, U_fun):
        """Used for scipy ODE solver."""
        U = U_fun([[t]], X.reshape(1, -1)).flatten()
        return U

    # ========================
    #   Augmented Dynamics
    # ========================
    def terminal_cost(self, X):
        return 0

    def running_cost(self, t, X, U):
        FF = self.F(torch.tensor(t, dtype=torch.float32, device=self.device),
                    torch.tensor(X.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        return 0.5 * np.sum(U * U, axis=0, keepdims=True) + FF.T

    def aug_dynamics(self, t, X_aug):
        """
        Augmented dynamics:
        dx/dt = u,
        dp/dt = -∂F/∂x,
        dv/dt = -L(x, u)
        """
        U = self.U_star(X_aug)
        x = X_aug[:self.dim]

        dFdx = self.d_F(
            torch.tensor(t, dtype=torch.float32, device=self.device),
            torch.tensor(x.T, dtype=torch.float32, device=self.device)
        ).cpu().detach().numpy()

        Ax = X_aug[self.dim:2*self.dim]
        dxdt = U
        dAxdt = -dFdx.T
        L = self.running_cost(t, x, U)

        return np.vstack((dxdt, dAxdt, -L))

