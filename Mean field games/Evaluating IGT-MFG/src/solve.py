import torch
import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
import copy
from scipy.stats import ks_2samp
import torch.nn as nn
from scipy.stats import wasserstein_distance
#from geomloss import SamplesLoss

#================================ DGM_HJB =========================================#

def Solve_HJB(an, V_NN, num_epoch, t, lr, num_samples, device):
    """
    Train the value network V_NN to solve the Hamilton-Jacobi-Bellman (HJB) equation
    using the PDE residual loss.

    Args:
        an          : Analytic object with ham() and sampling methods
        V_NN        : Neural network model to approximate V(t, x)
        num_epoch   : Number of training epochs
        t           : Time grid (tensor of shape [num_samples, 1])
        lr          : Learning rate
        num_samples : Number of spatial samples per epoch
        device      : Torch device (cuda or cpu)

    Returns:
        V_NN        : Trained neural network
    """
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    # Sample initial x points (requires grad for autograd)
    x_rand = an.sample_x0(num_samples).requires_grad_(True)

    for epoch in range(num_epoch + 1):
        # Ensure t has gradients for ∂V/∂t
        t = t.requires_grad_(True)

        # Forward pass: V(t, x)
        V_nn = V_NN(t, x_rand)

        # Compute ∂V/∂t
        V_nn_t = torch.autograd.grad(
            outputs=V_nn, inputs=t,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Compute ∇V
        V_nn_x = torch.autograd.grad(
            outputs=V_nn, inputs=x_rand,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # HJB residual loss: (∂V/∂t + H)^2
        Loss = torch.mean((V_nn_t + an.ham(t, x_rand, V_nn_x)) ** 2)

        # Backward pass and optimization step
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Iteration {epoch:5d}: Loss = {Loss.item():.4e}")

    print('\n')
    return V_NN

#================================  TPBVP + HJB =========================================#

def Approximate_v(an, V_NN, data, num_epoch, t, lr, num_samples, Round, device):
    """
    Train the value network V_NN to fit data value and gradient data,
    while minimizing the HJB residual.

    Args:
        an          : Analytic problem instance
        V_NN        : Neural network model for value function V(t,x)
        data        : Dictionary with keys 't', 'X', 'V', 'A'
        num_epoch   : Number of training epochs
        t           : Time grid (torch tensor)
        lr          : Learning rate
        num_samples : Number of HJB sample points
        Round       : Round number (unused but kept for interface)
        device      : PyTorch device

    Returns:
        V_NN        : Updated model
    """
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    # Initial spatial samples for HJB residual
    x_rand = an.sample_x0(num_samples).requires_grad_(True)

    for epoch in range(num_epoch + 1):
        t = t.requires_grad_(True)

        # Forward pass: V(t, x_rand)
        V_nn = V_NN(t, x_rand)

        # ∂V/∂t
        V_nn_t = torch.autograd.grad(
            outputs=V_nn, inputs=t,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # ∇V
        V_nn_x = torch.autograd.grad(
            outputs=V_nn, inputs=x_rand,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Loss components
        Loss_hjb = torch.mean((V_nn_t + an.ham(t, x_rand, V_nn_x)) ** 2)
        Loss_v   = torch.mean((V_NN(data['t'], data['X']) - data['V']) ** 2)
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A']) ** 2)

        # Total loss
        Loss_total = Loss_hjb + Loss_v + Loss_v_x

        # Backpropagation
        optimizer.zero_grad()
        Loss_total.backward()
        optimizer.step()

        # Print progress every 1000 iterations
        if epoch % 1000 == 0:
            print(f"Iteration {epoch:5d}: Loss_V = {Loss_v.item():.4e}, "
                  f"Loss_V_x = {Loss_v_x.item():.4e}, "
                  f"Loss_HJB = {Loss_hjb.item():.4e}, "
                  f"Loss_total = {Loss_total.item():.4e}")

    print('\n')
    return V_NN

#================================ solve BVP ===========================================#

def generate_data(an, V_NN, num_samples, device):
    
    def eval_u(t, x):
        u = - V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
    def bvp_guess(t, x):
        V_NN.eval()
        V = V_NN(torch.tensor(t, dtype=torch.float32, device=device), 
                 torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        V_x = V_NN.get_grad(t, x).detach().cpu().numpy()
        return V, V_x

    print('Generating data_OC...')

    dim = an.dim

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0
    start_time = time.time()
    x0_int = an.gen_x0(num_samples, Torch=False)

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        #print('Solving BVP #', Ns_sol+1, '...', end='\r')

        X0 = x0_int[Ns_sol,:]
        bc = an.make_bc(X0)

        # Integrates the closed-loop system (NN controller)

        SOL = solve_ivp(an.dynamics, [0., an.TT], X0,
                        method=an.ODE_solver,
                        args=(eval_u,),
                        rtol=1e-08)

        V_guess, A_guess = bvp_guess(SOL.t.reshape(1,-1).T, SOL.y.T)

        try:
            # Solves the two-point boundary value problem
            
            X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))

            SOL = solve_bvp(an.aug_dynamics, bc, SOL.t, X_aug_guess,
                            verbose=0,
                            tol=an.data_tol,
                            max_nodes=an.max_nodes)
            if not SOL.success:
                warnings.warn(Warning())

            Ns_sol += 1

            V = SOL.y[-1:] + an.terminal_cost(SOL.y[:dim,-1])

            t_OUT = np.vstack((t_OUT, SOL.x.reshape(1,-1).T))
            X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
            A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
            V_OUT = np.vstack((V_OUT, V.T))

        except Warning:
            pass

    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'BVP solutions in %.1f' % (time.time() - start_time), 'sec \n')

    data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
            'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
            'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}

    return data
   
#==========================   Simulate points   ==========================================#   


def sim_points(an, V_NN, num_samples, N, t0, tf, device):
    """
    Simulate closed-loop trajectories using the current V_NN policy.

    Args:
        an          : Analytic problem instance
        V_NN        : Trained neural network for value function V(t,x)
        num_samples : Number of initial samples to simulate
        N           : Number of time steps per trajectory
        t0, tf      : Initial and final times
        device      : torch device (CPU or CUDA)

    Returns:
        t_tr   : Flattened time vector for training (N+1) * num_samples x 1
        x_tr   : Initial positions repeated (N+1) times, shape: (N+1)*num_samples x dim
        t_OUT  : Original (unflattened) time grid, shape: num_samples x (N+1)
        X_OUT  : Flattened trajectory, shape: (N+1)*num_samples x dim
    """

    def eval_u(t, x):
        """Closed-loop control: u = -∇V"""
        u = -V_NN.get_grad(t, x).detach().cpu().numpy()
        return u

    X_OUT = np.empty((0, an.dim))
    t_OUT = np.empty((0, 1))

    data = an.gen_x0(num_samples, Torch=False)
    Ns_sol = 0
    start_time = time.time()

    print('Generating data_MFG...')

    while Ns_sol < num_samples:
        X0 = data[Ns_sol, :]

        # Solve the closed-loop ODE for one trajectory
        SOL = solve_ivp(
            fun=an.dynamics,
            t_span=[t0, tf],
            y0=X0,
            method='RK23',
            t_eval=np.linspace(t0, tf, N + 1),
            args=(eval_u,),
            rtol=1e-8
        )

        # Stack results
        t_OUT = np.vstack((t_OUT, SOL.t.reshape(-1, 1)))
        X_OUT = np.vstack((X_OUT, SOL.y.T))

        Ns_sol += 1

    # Reshape for return
    t_train = t_OUT.reshape(num_samples, N + 1)
    t_tr = t_train.T.flatten().reshape(-1, 1)                       # (N+1)*num_samples x 1
    x_tr = np.tile(data[0:num_samples, :], (N + 1, 1))             # (N+1)*num_samples x dim

    X_OUT = X_OUT.reshape(num_samples, N + 1, an.dim)
    X_OUT = X_OUT.transpose(1, 0, 2).reshape((N + 1) * num_samples, an.dim)

    print(f'Generated {X_OUT.shape[0]} data points from {Ns_sol} IVP solutions '
          f'in {time.time() - start_time:.1f} sec\n')

    return t_tr, x_tr, t_OUT, X_OUT


#================================  Train Generator =========================================#

def Train_Gen(an, G_NN, V_NN, t_tr, x_tr, X_OUT, num_epoch, t, lr, num_samples, Round, device):
    """
    Train the generator network G_NN to match simulated trajectory data
    and satisfy the dynamics constraint.

    Args:
        an          : Analytic problem instance
        G_NN        : Generator neural network G(t, x)
        V_NN        : Value function network (for ∇V in dynamics)
        t_tr        : Flattened training time vector
        x_tr        : Initial points for generator input
        X_OUT       : True points from BVP trajectories
        num_epoch   : Number of training iterations
        t           : Time samples for enforcing ODE constraint
        lr          : Learning rate
        num_samples : Number of samples for ODE matching
        Round       : Current round (unused)
        device      : torch.device (CPU or CUDA)

    Returns:
        G_NN_original : Untrained copy (before update)
        G_NN          : Trained generator
    """

    # Keep a copy of original G_NN
    G_NN.eval()
    G_NN_original = copy.deepcopy(G_NN)

    # Prepare training data
    x_rand = an.gen_x0(num_samples, Torch=True).requires_grad_(True)
    t = t.requires_grad_(True)
    t_train = torch.tensor(t_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_train = torch.tensor(x_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_OUT = torch.tensor(X_OUT, dtype=torch.float32, device=device).requires_grad_(True)

    G_NN.train()
    optimizer = optim.Adam(G_NN.parameters(), lr)

    best_loss = float('inf')

    for epoch in range(num_epoch + 1):
        # Forward pass: simulate new samples and compute gradients
        gen_samples = G_NN(t, x_rand)
        G_nn_t = G_NN.grad_t(t, x_rand)

        # Dynamics consistency loss
        Loss_ode = torch.mean((G_nn_t - an.dynamics_torch(t, G_NN(t, x_rand), V_NN)) ** 2)

        # Trajectory matching loss
        Loss_G = torch.mean((G_NN(t_train, X_train) - X_OUT) ** 2)

        # Total loss (weighted sum)
        Loss_total = Loss_G + 0.5 * Loss_ode

        # Optimization step
        optimizer.zero_grad()
        Loss_total.backward()
        optimizer.step()

        # Logging every 1000 iterations
        if epoch % 1000 == 0:
            print(f"Iteration {epoch:5d}: "
                  f"Loss_G = {Loss_G.item():.4e}, "
                  f"Loss_ODE = {Loss_ode.item():.4e}, "
                  f"Loss_total = {Loss_total.item():.4e}")

            new_loss = Loss_total.item()
            if new_loss > best_loss:
                pass  # keep current x_rand
            else:
                x_rand = an.gen_x0(num_samples, Torch=True).requires_grad_(True)
                best_loss = new_loss

    print('\n')
    return G_NN_original, G_NN



#==================================Compute J and V ==============================#
    
def Comp_J0(an, t, x, V_NN):
    """
    Compute the average cost functional J₀ for the current population trajectory
    using the generator stored in an.G_NN_list[-1].

    Args:
        an      : Analytic instance (contains G_NN_list)
        t       : Time grid (tensor of shape [T, 1])
        x       : Initial samples (tensor of shape [N, dim])
        V_NN    : Trained value network

    Returns:
        J (float): Estimated population cost J₀
    """

    # Expand t and x to create full batch of (t, x0) for generator
    t_expanded = t.repeat_interleave(x.shape[0], dim=0)     # [T*N, 1]
    x_expanded = x.repeat(t.shape[0], 1)                    # [T*N, dim]

    # Compute generator trajectory
    G = an.G_NN_list[-1]
    Xn = G(t_expanded, x_expanded)                          # [T*N, dim]

    # Compute population mean at each time step
    Xn_reshaped = Xn.view(t.shape[0], x.shape[0], -1)       # [T, N, dim]
    mean = torch.mean(Xn_reshaped, dim=1, keepdim=True)     # [T, 1, dim]

    # Compute squared deviation: F = 0.5 * (x - mean)^2
    F = 0.5 * torch.sum((Xn_reshaped - mean) ** 2, dim=2)   # [T, N]

    # Compute control u = -∇V(t, x)
    u = -V_NN.get_grad(t_expanded, Xn)                      # [T*N, dim]
    u_squared = 0.5 * torch.sum(u ** 2, dim=1)              # [T*N]
    u_squared = u_squared.view(t.shape[0], x.shape[0])      # [T, N]

    # Total cost = running cost (control + population interaction)
    l = u_squared + F                                       # [T, N]

    # Compute average cost J = (1/T) * mean over time and samples
    J = torch.mean(torch.sum(l, dim=0) / t.shape[0])

    return J.item()


def Comp_V(x, V_NN):
    """
    Compute the average value function V(0, x) over the initial distribution.

    Args:
        x     : Tensor of initial positions [N, dim]
        V_NN  : Trained value network V(t, x)

    Returns:
        V0 (float) : Mean value at time t=0 over all samples
    """
    t0 = torch.zeros_like(x[:, :1])  # Shape [N, 1], matching batch size
    V0 = torch.mean(V_NN(t0, x))     # Evaluate V at t = 0

    return V0.item()

    
  

#==================================Wasserstein==============================#

def wasserstein_fp(an, G_NN_list,              # BR passés (longueur k)
                   G_NN_new,               # BR courant   (G_k)
                   x0, m0,                    # (N,d)  échantillon initial
                   t_grid,                 # (T,1)  instants
                   p          = 1,
                   blur       = 0.05,
                   device     = "cpu"):
    """
    W_p(  m̄_k(t) , BR_k(t) )  pour t ∈ t_grid,   avec
        m̄_k = (1/(k+1)) [ δ_{x0} + Σ_{i=0}^{k-1} δ_{G_i} ].
    Retourne  np.ndarray shape (T,)  des distances.
    """

    T       = t_grid.shape[0]
    N, d    = x0.shape
    k       = len(G_NN_list)
    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, backend="tensorized")

    # ---------- pré-calcul des trajectoires BR_k -------------------
    with torch.no_grad():
        t_big = t_grid.repeat_interleave(N, dim=0)          # (T*N,1)
        x_rep = x0.repeat(T, 1)                             # (T*N,d)
        br_k  = G_NN_new(t_big, x_rep).view(T, N, d)        # (T,N,d)

    # ---------- pré-calcul des trajectoires passées ----------------
    past_traj = []  
    for G in G_NN_list:
        with torch.no_grad():
            past = G(t_big, x_rep).view(T, N, d)
        past_traj.append(past)
    if past_traj:
        past_traj = torch.stack(past_traj, dim=0)           # (k,T,N,d)

    # ---------- distances -----------------------------------------
    dists = torch.empty(T, device="cpu")
    for j in range(T):
        # population : x0 + BR passés (chacun N pts)
        if an.Round == 0:
            parts = [m0]                                        # (N,d)
        else:
            parts = [] 
        if k:
            parts.append(past_traj[:, j, :, :].reshape(k*N, d))
        X = torch.cat(parts, dim=0)                         # ((k+1)*N,d)
        Y = br_k[j]                                         # (N,d)
        dists[j] = loss_fn(X, Y).cpu()

    return dists.numpy()        # shape (T,)


    
